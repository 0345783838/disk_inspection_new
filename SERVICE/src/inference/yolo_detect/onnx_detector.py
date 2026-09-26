import time
import cv2
import numpy as np
import onnxruntime
import os
import glob
from tqdm import tqdm


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def multiclass_nms(boxes, scores, class_ids, iou_threshold):

    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices,:]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


class OnnxDetector:
    def __init__(self, path: str, label: list, conf_thres : float = 0.7, iou_thres: float = 0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.labels = label

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=onnxruntime.get_available_providers())
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        img_height, img_width = image.shape[:2]
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        boxes, scores, class_ids = self.process_output(outputs, img_width, img_height)

        return boxes, scores, class_ids

    def detect_objects_debug(self, image, conf_threshold, iou_threshold):
        img_height, img_width = image.shape[:2]
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        boxes, scores, class_ids = self.process_output_debug(outputs, img_width, img_height, conf_threshold, iou_threshold)

        return boxes, scores, class_ids

    def prepare_input(self, image):

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output, img_width, img_height):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions, img_width, img_height)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def process_output_debug(self, output, img_width, img_height, conf_threshold, iou_threshold):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > conf_threshold, :]
        scores = scores[scores > conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions, img_width, img_height)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = multiclass_nms(boxes, scores, class_ids, iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions, img_width, img_height):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes, img_width, img_height)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes, img_width, img_height):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([img_width, img_height, img_width, img_height])
        return boxes

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        

if __name__ == "__main__":
    model = OnnxDetector(r"D:\huynhvc\OTHERS\disk_checking\disk_checking\SERVICE\config\models\detect_point.onnx",
                         ["1"], conf_thres=0.5, iou_thres=0.1)
    image_path = r"D:\huynhvc\OTHERS\disk_checking\disk_checking\07_12"
    output_path = r"D:\huynhvc\OTHERS\disk_checking\disk_checking\testing\out_test"
    paths = glob.glob(f"{image_path}/*")
    for path in tqdm(paths):
        img = cv2.imread(path)
        boxes, confs, cls_idxs = model(img)
        for box, conf, cls in zip(boxes, confs, cls_idxs):
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            predict_class = model.labels[int(cls)]
            object_conf = float(conf)

            # Draw the bounding box
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Prepare the label with class name and confidence
            label = f"{object_conf:.2f}"

            # Calculate the position for the label
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top_left = (
            xmin, ymin - label_size[1] - 10 if ymin - label_size[1] - 10 > 10 else ymin + label_size[1] + 10)

            # Draw the label background
            cv2.rectangle(img, (top_left[0] - 1, top_left[1] + base_line + 10),
                          (top_left[0] + label_size[0], top_left[1] - label_size[1] + 10), (0, 255, 0), cv2.FILLED)

            # Put the label text
            cv2.putText(img, label, (top_left[0], top_left[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv2.imwrite(f"{output_path}/{os.path.basename(path)}", img)


