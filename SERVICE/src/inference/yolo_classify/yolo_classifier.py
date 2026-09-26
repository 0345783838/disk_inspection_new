import cv2
import numpy as np
import onnxruntime as ort


class YoloClassifier:
    def __init__(
        self,
        model_path: str,
        labels: list,
        input_size: tuple = (224, 224),
        conf_thres: float = 0.5,
        batch_size: int = 4  # ⭐ thêm batch_size
    ):
        self.conf_threshold = conf_thres
        self.labels = labels
        self.input_size = input_size
        self.batch_size = batch_size  # ⭐ lưu batch_size

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.ort_sess = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])

    @staticmethod
    def _center_crop(image, size=(224, 224)):
        h, w, _ = image.shape
        new_h, new_w = size
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        return image[top:top + new_h, left:left + new_w]

    @staticmethod
    def _normalize(image, mean=np.array([0., 0., 0.]), std=np.array([1., 1., 1.])):
        return (image - mean) / std

    def _preprocess(self, img, input_size):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)
        img = self._center_crop(img, input_size)
        img = img.astype(np.float32) / 255.0
        img = self._normalize(img)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = np.expand_dims(img, axis=0)
        return img

    def _preprocess_batch(self, imgs, input_size):
        batch = []
        for img in imgs:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)
            img = self._center_crop(img, input_size)
            img = img.astype(np.float32) / 255.0
            img = self._normalize(img)
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)
            batch.append(img)
        return np.array(batch, dtype=np.float32)

    def predict(self, img):
        try:
            imgs_numpy = self._preprocess(img, self.input_size)
            preds = self.ort_sess.run(None, {self.ort_sess.get_inputs()[0].name: imgs_numpy})[0]
            label = self.labels[np.argmax(preds)]
            conf = float(np.max(preds))
            return label, conf
        except Exception as ex:
            print(ex)
        return None, None

    # --- PREDICT BATCH CÓ CHIA NHỎ ---
    def predict_batch(self, imgs):
        try:
            all_labels = []
            all_confs = []

            # chia imgs thành từng batch nhỏ
            for i in range(0, len(imgs), self.batch_size):
                chunk = imgs[i:i + self.batch_size]
                imgs_numpy = self._preprocess_batch(chunk, self.input_size)

                preds = self.ort_sess.run(
                    None, {self.ort_sess.get_inputs()[0].name: imgs_numpy}
                )[0]

                labels = [self.labels[np.argmax(pred)] for pred in preds]
                confs = [float(np.max(pred)) for pred in preds]

                all_labels.extend(labels)
                all_confs.extend(confs)

            return all_labels, all_confs

        except Exception as ex:
            print(ex)

        return None, None


if __name__ == '__main__':
    import time

    model = YoloClassifier(r"D:\huynhvc\WORKING\PROJECT\SMT\DATA\working\training_cls\WLP_CHIP_PIN\wlp_chip_pin_yolo_11n_16_05\weights\best_old.onnx",
                           ['ng', 'ok'], (224, 224))
    # time_start = time.time()
    # a, b = model.predict_batch([img])
    # print(time.time() - time_start)

    time_start = time.time()

    img = cv2.imread(
        r"D:\huynhvc\WORKING\PROJECT\WLP_CHIP\data\CROP\SRG42AR80F05\PATTERN_0_-13_0_0_GOOD_0_1_2_3_4_5_6.png")
    img_2 = cv2.imread(r"D:\huynhvc\WORKING\PROJECT\WLP_CHIP\data\CROP\SH722ARB0F02\OK\PATTERN_-36_33_0_0_GOOD_0_1.png")
    img_3 = cv2.imread(r"D:\huynhvc\WORKING\PROJECT\WLP_CHIP\data\CROP\SH722ARB0F02\NG\PATTERN_-64_116_2_0_PAD_0_1.png")
    a, b = model.predict_batch([img, img_2, img_3, img])
    print(time.time() - time_start)
    print(a, b)
