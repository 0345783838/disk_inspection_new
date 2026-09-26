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
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]
    return keep_boxes


def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    unique_class_ids = np.unique(class_ids)
    keep_boxes = []
    for class_id in unique_class_ids:
        idx = np.where(class_ids == class_id)[0]
        if len(idx) == 0:
            continue
        keep = nms(boxes[idx], scores[idx], iou_threshold)
        keep_boxes.extend(idx[keep])
    return keep_boxes


def compute_iou(box, boxes):
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter
    return inter / np.clip(union, 1e-6, None)


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

        self.initialize_model(path)

    def __call__(self, image):
        return self.segment_large_image(image)

    # ================= INIT =================

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(
            path, providers=['CPUExecutionProvider']
        )
        self.get_input_details()
        self.get_output_details()

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [i.name for i in model_inputs]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [o.name for o in model_outputs]

    # ================= CORE =================

    def segment_large_image(self, image):
        H, W = image.shape[:2]

        crops, positions = self.split_image(image)

        all_boxes = []
        all_scores = []
        all_class_ids = []
        all_masks = []

        for crop, position in zip(crops, positions):
            boxes, scores, class_ids, masks = self.segment_objects(crop)

            if len(masks) == 0:
                continue

            full_mask = self.merge_masks(masks)


            all_boxes.append(boxes)
            all_scores.append(scores)
            all_class_ids.append(class_ids)
            all_masks.append(full_masks)

        if len(all_boxes) == 0:
            return [], [], [], np.zeros((H, W), dtype=np.uint8)

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        class_ids = np.concatenate(all_class_ids, axis=0)

        # ===== FIX CHỖ LỖI =====
        full_mask = np.zeros((H, W), dtype=np.uint8)
        for m in all_masks:
            full_mask = np.maximum(full_mask, np.max(m, axis=0))

        keep = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[keep], scores[keep], class_ids[keep], full_mask

    # ================= SPLIT =================

    def split_image(self, image):
        H, W = image.shape[:2]
        step = W // self.num_splits
        overlap = int(step * self.overlap_ratio)

        crops = []
        positions = []

        x = 0
        while x < W and len(crops) < self.num_splits:
            x_start = max(0, x - overlap)
            x_end = min(W, x + step + overlap)

            crop = image[:, x_start:x_end]
            crops.append(crop)
            positions.append((x_start, x_end))

            x += step

        return crops, positions

    # ================= MERGE MASK =================

    def merge_masks(self, masks):
        """
        masks: list[np.ndarray]  -> mỗi mask shape (h_i, w_i)
        """
        full_mask = np.zeros(masks[0].shape, dtype=np.uint8)
        for mask in masks:
            # and logic to merge
            full_mask = np.maximum(full_mask, mask)

        return full_mask

    # ================= SINGLE INFER =================

    def segment_objects(self, image):
        inp = self.prepare_input(image)
        outputs = self.inference(inp)

        boxes, scores, class_ids, mask_pred = self.process_box_output(outputs[0])
        mask_maps = self.process_mask_output(mask_pred, outputs[1], boxes)

        return boxes, scores, class_ids, mask_maps

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        return img[np.newaxis]

    def inference(self, input_tensor):
        return self.session.run(
            self.output_names, {self.input_names[0]: input_tensor}
        )

    # ================= POST =================

    def process_box_output(self, box_output):
        preds = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        scores = np.max(preds[:, 4:4 + num_classes], axis=1)
        keep = scores > self.conf_threshold

        preds = preds[keep]
        scores = scores[keep]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_preds = preds[..., :num_classes + 4]
        mask_preds = preds[..., num_classes + 4:]

        class_ids = np.argmax(box_preds[:, 4:], axis=1)

        boxes = self.extract_boxes(box_preds)

        keep = nms(boxes, scores, self.iou_threshold)

        return boxes[keep], scores[keep], class_ids[keep], mask_preds[keep]

    def process_mask_output(self, mask_preds, mask_output, boxes):
        if len(mask_preds) == 0:
            return []

        proto = np.squeeze(mask_output)
        c, mh, mw = proto.shape

        masks = sigmoid(mask_preds @ proto.reshape(c, -1))
        masks = masks.reshape(-1, mh, mw)

        scale_boxes = self.rescale_boxes(
            boxes, (self.img_height, self.img_width), (mh, mw)
        )

        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))

        for i in range(len(scale_boxes)):
            sx1, sy1, sx2, sy2 = scale_boxes[i].astype(int)
            x1, y1, x2, y2 = boxes[i].astype(int)

            crop = masks[i][sy1:sy2, sx1:sx2]
            crop = cv2.resize(crop, (x2 - x1, y2 - y1))

            mask_maps[i, y1:y2, x1:x2] = crop > 0.5

        return mask_maps.astype(np.uint8)

    def extract_boxes(self, box_preds):
        boxes = box_preds[:, :4]

        boxes = self.rescale_boxes(
            boxes,
            (self.input_height, self.input_width),
            (self.img_height, self.img_width),
        )

        boxes = xywh2xyxy(boxes)

        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        input_shape = np.array([
            input_shape[1], input_shape[0],
            input_shape[1], input_shape[0]
        ])
        boxes = boxes / input_shape
        boxes *= np.array([
            image_shape[1], image_shape[0],
            image_shape[1], image_shape[0]
        ])
        return boxes
