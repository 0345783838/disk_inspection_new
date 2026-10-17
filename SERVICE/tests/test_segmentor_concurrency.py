"""A shared segmentor must preserve each camera's geometry across ONNX calls."""
from concurrent.futures import ThreadPoolExecutor
import threading
import unittest

import numpy as np

from src.inference.yolo_segment.onnx_segmentor import OnnxSegmentor


def make_model(mode, mask_decode):
    model = OnnxSegmentor.__new__(OnnxSegmentor)
    model.input_height = model.input_width = 320
    model.preprocess_mode = mode
    model.rgb_input = True
    model.mask_decode = mask_decode
    model.num_masks = 2
    model.conf_threshold = .5
    model.iou_threshold = .8
    return model


def prediction(height, width, mode):
    # Independent forward projection of a known box, plus a rejected anchor.
    box = np.array([10., 20., 80., 90.], np.float32)
    if mode == 'letterbox':
        gain = min(320 / width, 320 / height)
        projected = box * gain
        projected[[0, 2]] += (320 - round(width * gain)) // 2
        projected[[1, 3]] += (320 - round(height * gain)) // 2
    else:
        projected = box / [width, height, width, height] * 320
    x1, y1, x2, y2 = projected
    values = np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1, .99, 1., 0.], np.float32)
    anchors = np.stack((values, np.zeros_like(values)), axis=1)[None]
    prototype = np.full((1, 2, 80, 80), 10., np.float32)
    return anchors, prototype


class SegmentorConcurrencyTests(unittest.TestCase):
    def test_shared_model_matches_serial_boxes_and_masks_during_overlapping_inference(self):
        for mode in ('letterbox', 'stretch'):
            for decoder in ('native', 'legacy'):
                for debug in (False, True):
                    with self.subTest(mode=mode, decoder=decoder, debug=debug):
                        model = make_model(mode, decoder)
                        images = [np.zeros((100, 200, 3), np.uint8), np.zeros((200, 100, 3), np.uint8)]
                        outputs = [prediction(*image.shape[:2], mode) for image in images]
                        local = threading.local()
                        barrier = threading.Barrier(2)

                        def inspect(index):
                            local.index = index
                            if debug:
                                return model.segment_objects_debug(images[index], .5, .8)
                            return model.segment_objects(images[index])

                        model.inference = lambda tensor: outputs[local.index]
                        serial = [inspect(index) for index in (0, 1)]
                        for result in serial:
                            np.testing.assert_allclose(result[0], [[10, 20, 80, 90]], atol=1e-5)
                            self.assertTrue(result[3].any())

                        def overlapping_inference(tensor):
                            # Both preparations finish before either decode can run.
                            barrier.wait(timeout=5)
                            return outputs[local.index]

                        model.inference = overlapping_inference
                        with ThreadPoolExecutor(max_workers=2) as pool:
                            concurrent = list(pool.map(inspect, (0, 1)))
                        for expected, actual in zip(serial, concurrent):
                            for expected_array, actual_array in zip(expected, actual):
                                np.testing.assert_array_equal(actual_array, expected_array)


if __name__ == '__main__':
    unittest.main()
