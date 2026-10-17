"""ONNX preprocessing options; production defaults must remain unchanged."""
import unittest
import importlib.util

import numpy as np

from src.inference.yolo_segment.onnx_segmentor import OnnxSegmentor


def model_without_session(mode="stretch", rgb=False, mask_decode="legacy"):
    model = OnnxSegmentor.__new__(OnnxSegmentor)
    model.input_height = 320
    model.input_width = 320
    model.preprocess_mode = mode
    model.rgb_input = rgb
    model.mask_decode = mask_decode
    model.tile_merge = "max"
    model.num_splits = 6
    model.overlap_ratio = .05
    return model


class SegmentorPreprocessTests(unittest.TestCase):
    def test_stretch_keeps_existing_bgr_default(self):
        model = model_without_session()
        image = np.zeros((288, 326, 3), np.uint8)
        image[:] = (10, 20, 30)
        tensor, height, width = model.prepare_input(image)
        self.assertEqual((height, width), (288, 326))
        self.assertNotIn('_image_transform', model.__dict__)
        np.testing.assert_allclose(tensor[0, :, 0, 0],
                                   np.array([10, 20, 30]) / 255)

    def test_letterbox_rgb_roundtrips_box(self):
        model = model_without_session("letterbox", True)
        image = np.zeros((288, 326, 3), np.uint8)
        image[:] = (10, 20, 30)
        tensor, _, _ = model.prepare_input(image)
        # Independent forward letterbox geometry used to verify box roundtrip.
        ratio = min(320 / 326, 320 / 288)
        pad_x, pad_y = (320 - round(326 * ratio)) // 2, (320 - round(288 * ratio)) // 2
        self.assertEqual(tensor.shape, (1, 3, 320, 320))
        self.assertEqual(pad_x, 0)
        np.testing.assert_allclose(tensor[0, :, pad_y + 1, 1],
                                   np.array([30, 20, 10]) / 255)
        xyxy = np.array([100, 50, 120, 270], dtype=np.float32)
        input_xyxy = xyxy.copy()
        input_xyxy[[0, 2]] = input_xyxy[[0, 2]] * ratio + pad_x
        input_xyxy[[1, 3]] = input_xyxy[[1, 3]] * ratio + pad_y
        xywh = np.array([[(input_xyxy[0] + input_xyxy[2]) / 2,
                          (input_xyxy[1] + input_xyxy[3]) / 2,
                          input_xyxy[2] - input_xyxy[0],
                          input_xyxy[3] - input_xyxy[1]]], dtype=np.float32)
        decoded = model.decode_boxes(xywh, 288, 326)
        np.testing.assert_allclose(decoded[0], xyxy, atol=1e-4)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('ultralytics'),
                         'Optional training dependencies required for reference parity test')
    def test_native_matches_training_library_letterbox(self):
        import torch
        from ultralytics.utils.ops import process_mask_native
        model = model_without_session("letterbox", mask_decode="native")
        model.prepare_input(np.zeros((288, 326, 3), np.uint8))
        rng = np.random.default_rng(930)
        prototype = rng.normal(size=(1, 32, 80, 80)).astype(np.float32)
        coefficients = rng.normal(size=(3, 32)).astype(np.float32)
        boxes = np.array([[100.2, 0.1, 106.7, 287.8], [20.5, 45.2, 220.9, 284.1],
                          [310.1, 0, 326, 288]], np.float32)
        actual = model.process_mask_output(coefficients, prototype, boxes, 288, 326)
        expected = process_mask_native(torch.from_numpy(prototype[0]),
                        torch.from_numpy(coefficients), torch.from_numpy(boxes), (288, 326)).numpy()
        np.testing.assert_array_equal(actual, expected)

    def test_native_preserves_narrow_full_height_mask(self):
        model = model_without_session("letterbox", mask_decode="native")
        model.prepare_input(np.zeros((288, 326, 3), np.uint8))
        prototype = np.full((1, 1, 80, 80), -4, np.float32)
        prototype[0, 0, :, 24:26] = 4
        boxes = np.array([[98.1, 0, 104.5, 288]], np.float32)
        result = model.process_mask_output(np.ones((1, 1), np.float32), prototype, boxes, 288, 326)
        self.assertTrue(result[0].any(axis=1).all())
        self.assertFalse(result[0, :, :99].any())
        self.assertFalse(result[0, :, 105:].any())

    def test_legacy_default_is_pixel_identical_to_original_path(self):
        import cv2
        from src.inference.yolo_segment.onnx_segmentor import sigmoid
        model = model_without_session()
        rng = np.random.default_rng(12)
        prototype = rng.normal(size=(1, 32, 80, 80)).astype(np.float32)
        coefficients = rng.normal(size=(2, 32)).astype(np.float32)
        boxes = np.array([[10, 0, 30, 288], [100, 10, 122, 280]], np.float32)
        first = model.process_mask_output(coefficients, prototype, boxes, 288, 326)
        # Independent reference for the pre-change stretch decoder.
        planes = sigmoid(coefficients @ prototype[0].reshape(32, -1)).reshape(-1,80,80)
        scaled_boxes = boxes / np.array([326,288,326,288]) * np.array([80,80,80,80])
        expected = np.zeros((2,288,326),np.uint8)
        for index,(plane,scaled_box,box) in enumerate(zip(planes,scaled_boxes,boxes)):
            sx1,sy1,sx2,sy2 = scaled_box.astype(int)
            x1,y1,x2,y2 = box.astype(int)
            expected[index,y1:y2,x1:x2] = cv2.resize(plane[sy1:sy2,sx1:sx2],(x2-x1,y2-y1))>.5
        np.testing.assert_array_equal(first, expected)

    def test_native_stretch_does_not_crop_letterbox_padding(self):
        model = model_without_session(mask_decode="native")
        prototype = np.ones((1,1,80,80),np.float32)
        boxes = np.array([[0,0,326,288]],np.float32)
        actual = model.process_mask_output(np.ones((1,1),np.float32),prototype,boxes,288,326)
        self.assertTrue(actual.all())

    def test_core_merge_removes_halo_branch_without_erasing_true_curve(self):
        model = model_without_session()
        model.tile_merge = "core"
        full = np.zeros((20,600),np.uint8)
        image = np.zeros((20,600,3),np.uint8)
        _, positions = model.split_image(image)
        expected = np.zeros_like(full)
        for y in range(20):
            expected[y,96+y//3] = 255
        for index,(x1,x2) in enumerate(positions):
            mask = expected[:,x1:x2].copy()
            if index == 1:
                mask[:,0:3] = 255  # crop-left hallucination, outside owned core
            model.paste_tile_mask(full,mask,x1,x2,index)
        np.testing.assert_array_equal(full,expected)

    def test_default_max_merge_keeps_prior_overlap_union(self):
        model = model_without_session()
        full = np.zeros((10,600),np.uint8)
        expected = full.copy()
        _, positions = model.split_image(np.zeros((10,600,3),np.uint8))
        rng = np.random.default_rng(42)
        for index,(x1,x2) in enumerate(positions):
            mask = (rng.random((10,x2-x1))>.9).astype(np.uint8)*255
            expected[:,x1:x2] = np.maximum(expected[:,x1:x2],mask)
            model.paste_tile_mask(full,mask,x1,x2,index)
        np.testing.assert_array_equal(full,expected)

    def test_core_merge_covers_remainder_columns(self):
        model = model_without_session()
        model.tile_merge = "core"
        full = np.zeros((10,613),np.uint8)
        _, positions = model.split_image(np.zeros((10,613,3),np.uint8))
        for index,(x1,x2) in enumerate(positions):
            model.paste_tile_mask(full,np.ones((10,x2-x1),np.uint8),x1,x2,index)
        self.assertTrue(full.all())


if __name__ == "__main__":
    unittest.main()
