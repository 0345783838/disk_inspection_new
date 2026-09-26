import onnxruntime as ort
import numpy as np
import cv2


class OnnxSegmentor:
    def __init__(
        self,
        onnx_path: str,
        conf_thres: float = 0.5,
        gpu_mode: int = 0,
        num_threads: int = 8,
        num_splits: int = 6,
        overlap_ratio: float = 0.05,
    ):
        """
        ONNX Inference Engine cho segmentation ảnh dài
        """
        self.conf_thres = conf_thres
        self.num_splits = num_splits
        self.overlap_ratio = overlap_ratio

        self._initialize_model(onnx_path, gpu_mode, num_threads)

    def __call__(self, image: np.ndarray):
        return self.infer(image)

    # =========================
    # INIT MODEL
    # =========================
    def _initialize_model(self, path, gpu_mode=0, num_threads=8):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if gpu_mode:
            self.session = ort.InferenceSession(
                path, providers=["CUDAExecutionProvider"]
            )
        else:
            so.intra_op_num_threads = num_threads
            so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            self.session = ort.InferenceSession(
                path,
                sess_options=so,
                providers=["CPUExecutionProvider"],
            )

        self._get_input_details()
        self._get_output_details()

        # Warmup
        dummy = np.zeros(
            (1, 3, self.input_height, self.input_width), dtype=np.float32
        )
        for _ in range(3):
            _ = self.session.run(self.output_names, {self.input_name: dummy})

    def _get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def _get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [out.name for out in model_outputs]

    # =========================
    # PREPROCESS
    # =========================
    def preprocess_batch(self, images):
        """
        images: list of (H, W, 3)
        return: (B, 3, H_in, W_in)
        """
        batch = []
        for img in images:
            img_r = cv2.resize(img, (self.input_width, self.input_height))
            img_r = img_r.astype(np.float32) / 255.0
            img_r = np.transpose(img_r, (2, 0, 1))
            batch.append(img_r)
        return np.stack(batch, axis=0)

    # =========================
    # INFER LARGE IMAGE
    # =========================
    def infer_large_image(self, image: np.ndarray):
        """
        Input: ảnh lớn (H, W, 3)
        Output:
            - full_mask (H, W) uint8
            - score (float)
        """
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

        # Preprocess batch
        batch_input = self.preprocess_batch(crops)

        # Inference
        outputs = self.session.run(
            None, {self.input_name: batch_input})[0]  # (B, 1, H_in, W_in)

        # Full probability map
        full_prob = np.zeros((H, W), dtype=np.float32)

        for i, (x_start, x_end) in enumerate(positions):
            crop_w = x_end - x_start

            logits = outputs[i, 0]
            prob = 1.0 / (1.0 + np.exp(-logits))

            prob_resized = cv2.resize(
                prob,
                (crop_w, H),
                interpolation=cv2.INTER_LINEAR,
            )

            full_prob[:, x_start:x_end] = np.maximum(
                full_prob[:, x_start:x_end],
                prob_resized,
            )

        # Threshold sau cùng
        full_mask = (full_prob > self.conf_thres).astype(np.uint8) * 255

        # Score
        if full_mask.sum() == 0:
            score = 1.0
        else:
            score = float(full_prob[full_prob > self.conf_thres].mean())

        return full_mask, score

    def infer_large_image_debug(self, image: np.ndarray, conf_thres):
        """
        Input: ?nh l?n (H, W, 3)
        Output:
            - full_mask (H, W) uint8
            - score (float)
        """
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

        # Preprocess batch
        batch_input = self.preprocess_batch(crops)

        # Inference
        outputs = self.session.run(
            None, {self.input_name: batch_input})[0]  # (B, 1, H_in, W_in)

        # Full probability map
        full_prob = np.zeros((H, W), dtype=np.float32)

        for i, (x_start, x_end) in enumerate(positions):
            crop_w = x_end - x_start

            logits = outputs[i, 0]
            prob = 1.0 / (1.0 + np.exp(-logits))

            prob_resized = cv2.resize(
                prob,
                (crop_w, H),
                interpolation=cv2.INTER_LINEAR,
            )

            full_prob[:, x_start:x_end] = np.maximum(
                full_prob[:, x_start:x_end],
                prob_resized,
            )

        # Threshold sau cùng
        full_mask = (full_prob > conf_thres).astype(np.uint8) * 255

        # Score
        if full_mask.sum() == 0:
            score = 1.0
        else:
            score = float(full_prob[full_prob > conf_thres].mean())

        return full_mask, score

    # =========================
    # PUBLIC API
    # =========================
    def infer(self, image: np.ndarray):
        """
        Giữ API cũ: truyền ảnh to → trả mask full
        """
        return self.infer_large_image(image)
