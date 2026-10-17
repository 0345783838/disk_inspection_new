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
        overlap_ratio: float = 0.05,
        preprocess_mode: str = "stretch",
        rgb_input: bool = False,
        mask_decode: str = "legacy",
        tile_merge: str = "max",
    ):
        if preprocess_mode not in {"stretch", "letterbox"}:
            raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")
        if mask_decode not in {"legacy", "native"}:
            raise ValueError(f"Unsupported mask_decode: {mask_decode}")
        if tile_merge not in {"max", "core"}:
            raise ValueError(f"Unsupported tile_merge: {tile_merge}")
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.labels = label
        self.num_masks = num_masks
        self.num_splits = num_splits
        self.overlap_ratio = overlap_ratio
        self.preprocess_mode = preprocess_mode
        self.rgb_input = rgb_input
        self.mask_decode = mask_decode
        self.tile_merge = tile_merge

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

    def _letterbox_transform(self, img_h, img_w):
        """Derive geometry from this image, never from another in-flight request."""
        if self.preprocess_mode != "letterbox":
            return None
        ratio = min(self.input_width / img_w, self.input_height / img_h)
        return (ratio,
                (self.input_width - round(img_w * ratio)) // 2,
                (self.input_height - round(img_h * ratio)) // 2)

    def prepare_input(self, image):
        img_h, img_w = image.shape[:2]
        transform = self._letterbox_transform(img_h, img_w)
        if transform is not None:
            ratio, pad_x, pad_y = transform
            resized_w = round(img_w * ratio)
            resized_h = round(img_h * ratio)
            input_img = np.full((self.input_height, self.input_width, 3),
                                114, dtype=np.uint8)
            input_img[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = (
                cv2.resize(image, (resized_w, resized_h)))
        else:
            input_img = cv2.resize(image, (self.input_width, self.input_height))
        if self.rgb_input:
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis].astype(np.float32)

        return input_tensor, img_h, img_w

    def inference(self, input_tensor):
        return self.session.run(self.output_names, {self.input_names[0]: input_tensor})

    def decode_boxes(self, boxes, img_h, img_w):
        transform = self._letterbox_transform(img_h, img_w)
        if transform is None:
            boxes = self.rescale_boxes(
                boxes, (self.input_height, self.input_width), (img_h, img_w))
            boxes = xywh2xyxy(boxes)
        else:
            ratio, pad_x, pad_y = transform
            boxes = xywh2xyxy(boxes)
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h)
        return boxes

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

        boxes = self.decode_boxes(boxes, img_h, img_w)

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

        boxes = self.decode_boxes(boxes, img_h, img_w)

        indices = nms(boxes, scores, iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices], mask_preds[indices]

    def process_mask_output(self, mask_preds, mask_output, boxes, img_h, img_w):
        if len(mask_preds) == 0:
            return []

        if self.mask_decode == "native":
            return self.process_mask_native(mask_preds, mask_output, boxes, img_h, img_w)

        mask_output = np.squeeze(mask_output)
        num_mask, mh, mw = mask_output.shape

        masks = sigmoid(mask_preds @ mask_output.reshape(num_mask, -1))
        masks = masks.reshape(-1, mh, mw)

        transform = self._letterbox_transform(img_h, img_w)
        if transform is None:
            scale_boxes = self.rescale_boxes(
                boxes, (img_h, img_w), (mh, mw))
        else:
            ratio, pad_x, pad_y = transform
            scale_boxes = boxes.copy()
            scale_boxes[:, [0, 2]] = (
                scale_boxes[:, [0, 2]] * ratio + pad_x) * mw / self.input_width
            scale_boxes[:, [1, 3]] = (
                scale_boxes[:, [1, 3]] * ratio + pad_y) * mh / self.input_height

        final_masks = np.zeros((len(scale_boxes), img_h, img_w), dtype=np.uint8)

        for i in range(len(scale_boxes)):
            sx1, sy1, sx2, sy2 = scale_boxes[i].astype(int)
            x1, y1, x2, y2 = boxes[i].astype(int)

            crop = masks[i][sy1:sy2, sx1:sx2]
            if crop.size == 0 or x2 <= x1 or y2 <= y1:
                continue

            crop = cv2.resize(crop, (x2 - x1, y2 - y1))
            crop = (crop > 0.5).astype(np.uint8)

            final_masks[i, y1:y2, x1:x2] = crop

        return final_masks

    def process_mask_native(self, mask_preds, mask_output, boxes, img_h, img_w):
        """Resize full logits before box cropping; preserve thin edge geometry.

        Matches the training library's native mask path without requiring torch.
        The legacy path remains the default for existing production weights.
        """
        prototype = np.squeeze(mask_output, axis=0) if mask_output.ndim == 4 else mask_output
        channels, mh, mw = prototype.shape
        logits = (mask_preds @ prototype.reshape(channels, -1)).reshape(-1, mh, mw)
        if self.preprocess_mode == "letterbox":
            gain = min(mh / img_h, mw / img_w)
            pad_w = (mw - img_w * gain) / 2
            pad_h = (mh - img_h * gain) / 2
            left, top = int(round(pad_w - .1)), int(round(pad_h - .1))
            right = mw - int(round(pad_w + .1))
            bottom = mh - int(round(pad_h + .1))
            logits = logits[:, top:bottom, left:right]
        result = np.zeros((len(boxes), img_h, img_w), dtype=np.uint8)
        for index, (plane, box) in enumerate(zip(logits, boxes)):
            # Integer pixel coordinates satisfy x >= floating x1, x < x2.
            x1, y1, x2, y2 = np.ceil(box).astype(int)
            x1, x2 = np.clip([x1, x2], 0, img_w)
            y1, y2 = np.clip([y1, y2], 0, img_h)
            if x2 <= x1 or y2 <= y1:
                continue
            scaled = cv2.resize(plane, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            result[index, y1:y2, x1:x2] = (scaled[y1:y2, x1:x2] > 0).astype(np.uint8)
        return result

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

    def paste_tile_mask(self, full_mask, tile_mask, x1, x2, tile_index):
        """Use overlap as context only in core mode; retain legacy max by default."""
        if self.tile_merge == "core":
            width = full_mask.shape[1]
            step = width // self.num_splits
            start = max(x1, tile_index * step)
            end = min(x2, width if tile_index == self.num_splits - 1
                      else (tile_index + 1) * step)
            if end > start:
                full_mask[:, start:end] = tile_mask[:, start-x1:end-x1]
        else:
            full_mask[:, x1:x2] = np.maximum(full_mask[:, x1:x2], tile_mask)

    def segment_large_image(self, image):
        H, W = image.shape[:2]
        crops, positions = self.split_image(image)
        full_image_mask = np.zeros((H, W), dtype=np.uint8)

        for tile_index, (crop, (x1, x2)) in enumerate(zip(crops, positions)):
            _, _, _, masks = self.segment_objects(crop)
            if len(masks) == 0:
                continue

            merged = self.merge_masks(masks)
            merged = cv2.resize(
                merged, (x2 - x1, H), interpolation=cv2.INTER_NEAREST
            ) * 255

            self.paste_tile_mask(full_image_mask, merged, x1, x2, tile_index)

        return full_image_mask, None

    def segment_large_image_debug(self, image, conf_threshold, iou_threshold):
        H, W = image.shape[:2]
        crops, positions = self.split_image(image)
        full_image_mask = np.zeros((H, W), dtype=np.uint8)

        for tile_index, (crop, (x1, x2)) in enumerate(zip(crops, positions)):
            _, _, _, masks = self.segment_objects_debug(crop, conf_threshold, iou_threshold)
            if len(masks) == 0:
                continue

            merged = self.merge_masks(masks)
            merged = cv2.resize(
                merged, (x2 - x1, H), interpolation=cv2.INTER_NEAREST
            ) * 255

            self.paste_tile_mask(full_image_mask, merged, x1, x2, tile_index)

        return full_image_mask, None

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        input_shape = np.array([input_shape[1], input_shape[0],
                                input_shape[1], input_shape[0]])
        boxes = boxes / input_shape
        boxes *= np.array([image_shape[1], image_shape[0],
                           image_shape[1], image_shape[0]])
        return boxes
