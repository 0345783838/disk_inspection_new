"""Regression tests run without cameras, PLC connections or ONNX inference."""
import io
import sys
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile
from pydantic import ValidationError

from src.dtos.meta import DataResponse, UvParams
from src.tools.caliper_advanced_new import AdvancedMultiEdgeCaliper

# Isolate hardware/model construction, while exercising the actual service and route code.
base = types.ModuleType("src.service.base_service")
base.BaseService = object
with patch.dict(sys.modules, {"src.service.base_service": base}):
    from src.service.check_disk_service_yolo import DiskCheckingService
    from src.controller import service_controller as routes


def caliper():
    return AdvancedMultiEdgeCaliper(min_edge_distance=4, subpixel=True, max_pairs=50,
                                    pair_max_gap=25, thickness_list=[3, 5, 7],
                                    length_rate=.95, polarity="both", angle_deg=0,
                                    return_profiles=True)


def stripe_mask(height, width, count):
    image = np.zeros((height, width), dtype=np.uint8)
    for x in range(count):
        left = 30 + 45 * x
        image[:, left:left + 10] = 255
    return image


class CaliperTests(unittest.TestCase):
    def test_shared_caliper_keeps_concurrent_results_independent(self):
        shared = caliper()
        masks = [stripe_mask(32, 400, n) for n in (4, 7)]
        barrier = threading.Barrier(2)
        original_pair = shared._pair_edges

        def pair(edges, gap):
            barrier.wait(timeout=5)
            return original_pair(edges, gap)

        with patch.object(shared, "_pair_edges", side_effect=pair):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda image: shared.measure(image, (200, 16)), masks))
        self.assertEqual([len(result["pairs"]) for result in results], [4, 7])
        self.assertIsNot(results[0]["profiles"], results[1]["profiles"])
        self.assertNotIn("edges", shared.__dict__)
        self.assertNotIn("pairs", shared.__dict__)
        shared.measure(masks[1], (200, 16))
        self.assertEqual(len(results[0]["pairs"]), 4)

    def test_debug_caliper_retains_pair_measurements(self):
        shared = caliper()
        mask = stripe_mask(32, 400, 4)
        prod = shared.measure(mask, (200, 16))
        debug = shared.measure_debug(mask, (200, 16), 4, 25, .95, [3, 5, 7])
        self.assertEqual([p["distance"] for p in prod["pairs"]],
                         [p["distance"] for p in debug["pairs"]])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.service = DiskCheckingService()
        s = self.service
        s.caliper = caliper()
        s.num_disk = 25
        s.min_disk_distance, s.max_disk_distance = 1, 9999
        s.min_disk_area = 0
        s.uv_disk_lower_threshold, s.uv_disk_upper_threshold = (107, 35, 40), (119, 150, 250)
        s.uv_min_disk_area = 0
        self.boxes = np.array([[40 + i * 28, 90, 50 + i * 28, 100] for i in range(25)], dtype=np.float32)
        s.disk_point_detect_model = types.SimpleNamespace(conf_threshold=.1, iou_threshold=.1,
            labels=['point'], detect_objects_debug=lambda *args: (self.boxes.copy(), np.ones(25), np.zeros(25, dtype=int)))
        s.disk_segmentor_yolo = types.SimpleNamespace(conf_threshold=.5, iou_threshold=.8,
            segment_large_image_debug=lambda image, *args: (stripe_mask(*image.shape[:2], 5), None))
        self.label = 'ok'
        s.point_classification_model = types.SimpleNamespace(
            predict_batch=lambda images: ([self.label] * len(images), [1.] * len(images)))
        self.image = np.zeros((200, 800, 3), dtype=np.uint8)

    def test_white_debug_matches_production_for_all_classification_states(self):
        for label, expected in [('ok', 0), ('ng', 1), ('no_disk', 2)]:
            with self.subTest(label=label):
                self.label = label
                prod = self.service.check_disk_white(self.image.copy())
                debug = self.service.check_disk_debug(self.image.copy(), self.service._default_white_params())
                self.assertEqual(prod.Result, expected)
                self.assertEqual(prod.Result, debug.Result)
                self.assertEqual(prod.ErrorCode, debug.ErrorCode)
                self.assertEqual(prod.ResImg, debug.FinalImg)
                for field in ['CropBox', 'UvBox1', 'UvBox2', 'Mid1', 'Mid2']:
                    self.assertEqual(getattr(prod, field), getattr(debug, field))

    def test_no_detections_rejects_without_uv_geometry(self):
        self.boxes = np.empty((0, 4), dtype=np.float32)
        prod = self.service.check_disk_white(self.image)
        debug = self.service.check_disk_debug(self.image, self.service._default_white_params())
        self.assertEqual(prod.Result, 1)
        self.assertEqual(debug.Result, 1)
        self.assertEqual(prod.ErrorCode, 'ERROR_002')
        self.assertIsNone(prod.CropBox)
        self.assertEqual(prod.MinDiskDistance, 0)
        self.assertTrue(prod.ResImg and debug.FinalImg)

    def test_debug_uses_requested_disk_count(self):
        params = self.service._default_white_params()
        self.assertEqual(self.service.check_disk_debug(self.image.copy(), params).Result, 0)
        params.disk_num = 3
        self.assertEqual(self.service.check_disk_debug(self.image.copy(), params).Result, 1)

    def test_uv_debug_and_production_agree_for_empty_homogeneous_and_mixed(self):
        crop = str([[0, 0], [400, 0], [400, 100], [0, 100]])
        top = str([[0, 10], [400, 10], [400, 40], [0, 40]])
        bottom = str([[0, 50], [400, 50], [400, 80], [0, 80]])
        mids = [str([[30 + 45*i, 10] for i in range(n)]) for n in (4, 7)]
        params = UvParams(crop_box=crop, uv_box_1=top, uv_box_2=bottom, mid_1=mids[0], mid_2=mids[1],
                          uv_disk_lower_threshold=[107, 35, 40], uv_disk_upper_threshold=[119, 150, 250], uv_disk_min_area=0)
        for counts, expected in [((0, 0), True), ((4, 7), True), ((1, 2), False)]:
            with self.subTest(counts=counts):
                hsv = np.zeros((100, 400, 3), dtype=np.uint8)
                for y1, y2, count in [(10, 40, counts[0]), (50, 80, counts[1])]:
                    mask = stripe_mask(y2-y1, 400, count)
                    hsv[y1:y2][mask > 0] = (113, 80, 150)
                image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                prod = self.service.check_disk_uv(image.copy(), crop, top, bottom, *mids)
                debug = self.service.check_disk_uv_debug(image.copy(), params)
                self.assertEqual(prod.Result, expected)
                self.assertEqual(prod.Result, debug.Result)
                self.assertEqual(prod.CountUvDisk, debug.CountUvDisk)
                self.assertEqual(prod.ResImg, debug.FinalImg)

    def test_invalid_uv_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            self.service.check_disk_uv(self.image, '[]', '[]', '[]', '[]', '[]')

    def test_uv_accepts_existing_white_crop_rounding(self):
        crop = np.array([[10.7, 0], [410.2, 0], [410.2, 100], [10.7, 100]])
        region = np.array([[0, 10], [400, 10], [400, 40], [0, 40]])
        self.service._validate_uv_geometry(self.image, crop, region, region, np.array([]), np.array([]))


class ContractTests(unittest.TestCase):
    def test_bool_cannot_silently_become_ok(self):
        with self.assertRaises(ValidationError):
            DataResponse(Result=False)
        self.assertEqual(DataResponse().Result, 1)

    def test_trigger_uses_one_read(self):
        with patch.object(routes.plc_controlling_service, 'read_trigger', side_effect=[(1, True), (-1, False)]) as read:
            self.assertEqual(routes.read_trigger(), {'Success': 1, 'Status': True})
            read.assert_called_once()

    def test_empty_or_invalid_image_returns_400(self):
        for data in [b'', b'not an image']:
            with self.subTest(data=data):
                with self.assertRaises(HTTPException) as caught:
                    routes.decode_image(UploadFile(file=io.BytesIO(data)))
                self.assertEqual(caught.exception.status_code, 400)

    def test_malformed_form_returns_400(self):
        with self.assertRaises(HTTPException) as caught:
            routes.parse_form(UvParams, '{')
        self.assertEqual(caught.exception.status_code, 400)


if __name__ == '__main__':
    unittest.main()
