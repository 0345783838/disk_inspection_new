import time
import cv2
import numpy as np
import onnxruntime
import math


def nms(boxes, scores, iou_threshold):
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []

    while sorted_indices.size > 0:
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        if sorted_indices.size == 1:
            break

        ious = compute_iou(boxes[box_id], boxes[sorted_indices[1:]])
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-6)


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class OnnxSegmentor:
    def __init__(
        self,
        path: str,
        label: list,
        conf_thres: float = 0.7,
        iou_thres: float = 0.5,
        num_masks=32,
        num_splits: int = 6,
        overlap_ratio: float = 0.05
    ):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.labels = label
        self.num_masks = num_masks
        self.num_splits = num_splits
        self.overlap_ratio = overlap_ratio

        self.session = onnxruntime.InferenceSession(
            path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        self.get_input_details()
        self.get_output_details()

    def __call__(self, image):
        return self.segment_large_image(image)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [i.name for i in model_inputs]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [i.name for i in model_outputs]

    def prepare_input(self, image):
        img_h, img_w = image.shape[:2]

        input_img = cv2.resize(image, (self.input_width, self.input_height))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis].astype(np.float32)

        return input_tensor, img_h, img_w

    def inference(self, input_tensor):
        return self.session.run(self.output_names, {self.input_names[0]: input_tensor})

    def process_box_output(self, box_output, img_h, img_w):
        preds = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        scores = np.max(preds[:, 4:4 + num_classes], axis=1)
        keep = scores > self.conf_threshold

        if not np.any(keep):
            return np.empty((0, 4)), [], [], np.empty((0, self.num_masks))

        preds = preds[keep]
        scores = scores[keep]

        class_ids = np.argmax(preds[:, 4:4 + num_classes], axis=1)
        boxes = preds[:, :4]
        mask_preds = preds[:, 4 + num_classes:]

        boxes = self.rescale_boxes(
            boxes,
            (self.input_height, self.input_width),
            (img_h, img_w)
        )
        boxes = xywh2xyxy(boxes)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h)

        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices], mask_preds[indices]

    def process_box_output_debug(self, box_output, img_h, img_w, conf_threshold, iou_threshold):
        preds = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        scores = np.max(preds[:, 4:4 + num_classes], axis=1)
        keep = scores > conf_threshold

        if not np.any(keep):
            return np.empty((0, 4)), [], [], np.empty((0, self.num_masks))

        preds = preds[keep]
        scores = scores[keep]

        class_ids = np.argmax(preds[:, 4:4 + num_classes], axis=1)
        boxes = preds[:, :4]
        mask_preds = preds[:, 4 + num_classes:]

        boxes = self.rescale_boxes(
            boxes,
            (self.input_height, self.input_width),
            (img_h, img_w)
        )
        boxes = xywh2xyxy(boxes)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h)

        indices = nms(boxes, scores, iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices], mask_preds[indices]

    def process_mask_output(self, mask_preds, mask_output, boxes, img_h, img_w):
        if len(mask_preds) == 0:
            return []

        mask_output = np.squeeze(mask_output)
        num_mask, mh, mw = mask_output.shape

        masks = sigmoid(mask_preds @ mask_output.reshape(num_mask, -1))
        masks = masks.reshape(-1, mh, mw)

        scale_boxes = self.rescale_boxes(
            boxes,
            (img_h, img_w),
            (mh, mw)
        )

        final_masks = np.zeros((len(scale_boxes), img_h, img_w), dtype=np.uint8)

        for i in range(len(scale_boxes)):
            sx1, sy1, sx2, sy2 = scale_boxes[i].astype(int)
            x1, y1, x2, y2 = boxes[i].astype(int)

            crop = masks[i][sy1:sy2, sx1:sx2]
            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (x2 - x1, y2 - y1))
            crop = (crop > 0.5).astype(np.uint8)

            final_masks[i, y1:y2, x1:x2] = crop

        return final_masks

    def segment_objects(self, image):
        input_tensor, img_h, img_w = self.prepare_input(image)
        outputs = self.inference(input_tensor)

        boxes, scores, class_ids, mask_preds = self.process_box_output(
            outputs[0], img_h, img_w
        )

        masks = self.process_mask_output(
            mask_preds, outputs[1], boxes, img_h, img_w
        )

        return boxes, scores, class_ids, masks

    def segment_objects_debug(self, image, conf_threshold, iou_threshold):
        input_tensor, img_h, img_w = self.prepare_input(image)
        outputs = self.inference(input_tensor)

        boxes, scores, class_ids, mask_preds = self.process_box_output_debug(
            outputs[0], img_h, img_w, conf_threshold, iou_threshold
        )

        masks = self.process_mask_output(
            mask_preds, outputs[1], boxes, img_h, img_w
        )

        return boxes, scores, class_ids, masks

    def split_image(self, image):
        H, W = image.shape[:2]
        step = W // self.num_splits
        overlap = int(step * self.overlap_ratio)

        crops, positions = [], []
        x = 0
        while x < W and len(crops) < self.num_splits:
            x_start = max(0, x - overlap)
            x_end = min(W, x + step + overlap)

            crops.append(image[:, x_start:x_end])
            positions.append((x_start, x_end))
            x += step

        return crops, positions

    def merge_masks(self, masks):
        full = np.zeros(masks[0].shape, dtype=np.uint8)
        for m in masks:
            full = np.maximum(full, m)
        return full

    def segment_large_image(self, image):
        H, W = image.shape[:2]
        crops, positions = self.split_image(image)
        full_image_mask = np.zeros((H, W), dtype=np.uint8)

        for crop, (x1, x2) in zip(crops, positions):
            _, _, _, masks = self.segment_objects(crop)
            if len(masks) == 0:
                continue

            merged = self.merge_masks(masks)
            merged = cv2.resize(
                merged, (x2 - x1, H), interpolation=cv2.INTER_NEAREST
            ) * 255

            full_image_mask[:, x1:x2] = np.maximum(
                full_image_mask[:, x1:x2], merged
            )

        return full_image_mask, None

    def segment_large_image_debug(self, image, conf_threshold, iou_threshold):
        H, W = image.shape[:2]
        crops, positions = self.split_image(image)
        full_image_mask = np.zeros((H, W), dtype=np.uint8)

        for crop, (x1, x2) in zip(crops, positions):
            _, _, _, masks = self.segment_objects_debug(crop, conf_threshold, iou_threshold)
            if len(masks) == 0:
                continue

            merged = self.merge_masks(masks)
            merged = cv2.resize(
                merged, (x2 - x1, H), interpolation=cv2.INTER_NEAREST
            ) * 255

            full_image_mask[:, x1:x2] = np.maximum(
                full_image_mask[:, x1:x2], merged
            )

        return full_image_mask, None

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        input_shape = np.array([input_shape[1], input_shape[0],
                                input_shape[1], input_shape[0]])
        boxes = boxes / input_shape
        boxes *= np.array([image_shape[1], image_shape[0],
                           image_shape[1], image_shape[0]])
        return boxes
