"""Select the real UV inspection path without model inference or hardware."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import decouple
import numpy as np

base = types.ModuleType('src.service.base_service')
base.BaseService = object
with patch.dict(sys.modules, {'src.service.base_service': base}):
    from src.service.check_disk_service_yolo import DiskCheckingService
    uv_service_module = sys.modules[DiskCheckingService.__module__]
from src.dtos.meta import UvParams
from src.tools.caliper_advanced_new import AdvancedMultiEdgeCaliper
from src.tools.uv_rim_counter import RimSettings


class ActiveServiceConfigTests(unittest.TestCase):
    def test_missing_true_and_false_flags_and_active_rim_parameters(self):
        service_root = Path(__file__).resolve().parents[1]
        source = (service_root/'config/config.env').read_text(encoding='utf-8')
        source = '\n'.join(line for line in source.splitlines()
                           if not line.startswith(('UV_USE_RIM_TRACKING', 'UV_RIM_HUE_HIGH')))

        class NoModelOrHardware:
            def __init__(self, *args, **kwargs):
                pass

        modules = {}
        for name, symbol in [
            ('src.communication.modbus_client', 'MbClient'),
            ('src.inference.segment.unet_onnx_segmentor', 'OnnxSegmentor'),
            ('src.inference.yolo_segment.onnx_segmentor', 'OnnxSegmentor'),
            ('src.inference.yolo_classify.yolo_classifier', 'YoloClassifier'),
            ('src.inference.yolo_detect.onnx_detector', 'OnnxDetector')]:
            module = types.ModuleType(name)
            setattr(module, symbol, NoModelOrHardware)
            modules[name] = module
        with tempfile.TemporaryDirectory() as folder:
            config_path = Path(folder)/'config.env'
            for flag, expected in [(None, True), ('True', True), ('False', False)]:
                with self.subTest(flag=flag):
                    content = source+'\nUV_RIM_HUE_HIGH = 140\n'
                    if flag is not None:
                        content += 'UV_USE_RIM_TRACKING = '+flag+'\n'
                    config_path.write_text(content, encoding='utf-8')
                    config = decouple.Config(decouple.RepositoryEnv(str(config_path)))
                    spec = importlib.util.spec_from_file_location(
                        '_uv_base_config_test', service_root/'src/service/base_service.py')
                    module = importlib.util.module_from_spec(spec)
                    with patch.dict(sys.modules, modules), patch.object(decouple, 'config', config):
                        spec.loader.exec_module(module)
                        service = module.BaseService()
                    self.assertIs(service.uv_use_rim_tracking, expected)
                    if expected:
                        self.assertEqual(service.uv_rim_settings.hue_high, 140)
                    else:
                        self.assertIsNone(service.uv_rim_settings)


class UvAlgorithmSelectionTests(unittest.TestCase):
    def setUp(self):
        self.service = DiskCheckingService()
        self.service.uv_disk_lower_threshold = (107, 35, 40)
        self.service.uv_disk_upper_threshold = (119, 150, 250)
        self.service.uv_min_disk_area = 0
        self.service.uv_rim_settings = RimSettings()
        self.service.caliper = AdvancedMultiEdgeCaliper(
            min_edge_distance=4, subpixel=True, max_pairs=50, pair_max_gap=25,
            thickness_list=[3, 5, 7], length_rate=.95, polarity='both', angle_deg=0)
        self.crop = str([[0, 0], [1500, 0], [1500, 120], [0, 120]])
        self.top = str([[0, 10], [1500, 10], [1500, 50], [0, 50]])
        self.bottom = str([[0, 60], [1500, 60], [1500, 100], [0, 100]])

    def inspect(self, white, uv, hsv_color=(113, 80, 180)):
        hsv = np.zeros((120, 1500, 3), dtype=np.uint8)
        for y1, y2, count in [(10, 50, uv[0]), (60, 100, uv[1])]:
            for index in range(count):
                x = 50+45*index
                hsv[y1:y2, x:x+8] = hsv_color
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        original = image.copy()
        mids = [str([[54+45*i, 20] for i in range(n)]) for n in white]
        prod = self.service.check_disk_uv(image.copy(), self.crop, self.top, self.bottom, *mids)
        params = UvParams(crop_box=self.crop, uv_box_1=self.top, uv_box_2=self.bottom,
                          mid_1=mids[0], mid_2=mids[1],
                          uv_disk_lower_threshold=[0, 0, 0],  # legacy debug values must not select another algorithm
                          uv_disk_upper_threshold=[0, 0, 0], uv_disk_min_area=0)
        if not getattr(self.service, 'uv_use_rim_tracking', True):
            params.uv_disk_lower_threshold = list(self.service.uv_disk_lower_threshold)
            params.uv_disk_upper_threshold = list(self.service.uv_disk_upper_threshold)
        debug = self.service.check_disk_uv_debug(image.copy(), params)
        np.testing.assert_array_equal(image, original)
        for field in ('Result', 'CountUvDisk', 'UvAlgorithm', 'UvCounts', 'WhiteCounts',
                      'UvUncertainCounts', 'UvRequiresReview'):
            self.assertEqual(getattr(prod, field), getattr(debug, field))
        self.assertEqual(prod.ResImg, debug.FinalImg)
        return prod

    def test_default_uses_new_counter_and_ignores_legacy_caliper(self):
        with patch.object(self.service, 'get_caliper_result_debug', side_effect=AssertionError('legacy called')):
            result = self.inspect((25, 25), (25, 25))
        self.assertEqual(result.UvAlgorithm, 'rim_tracking')
        self.assertEqual(result.UvCounts, [25, 25])
        self.assertTrue(result.Result)
        self.assertEqual(result.ErrorCode, 'PASS')
        self.assertEqual(result.CountUvDisk, 0)

    def test_matching_counts_pass_without_requiring_25(self):
        for counts in [(20, 20), (18, 17), (0, 20), (7, 9)]:
            with self.subTest(counts=counts):
                result = self.inspect(counts, counts)
                self.assertEqual(result.UvCounts, list(counts))
                self.assertEqual(result.WhiteCounts, list(counts))
                self.assertTrue(result.Result)

    def test_equal_total_does_not_hide_a_per_band_mismatch(self):
        result = self.inspect((20, 20), (19, 21))
        self.assertEqual(result.UvCounts, [19, 21])
        self.assertFalse(result.Result)
        self.assertEqual(result.ErrorCode, 'ERROR_003')
        self.assertEqual(result.CountUvDisk, 40)

    def test_one_non_uv_band_and_one_uv_band_is_mixing(self):
        result = self.inspect((25, 25), (0, 25))
        self.assertEqual(result.UvCounts, [0, 25])
        self.assertFalse(result.Result)

    def test_all_non_uv_keeps_existing_homogeneous_policy(self):
        result = self.inspect((20, 20), (0, 0))
        self.assertTrue(result.Result)
        self.assertEqual(result.UvCounts, [0, 0])

    def test_false_selects_legacy_without_calling_new_tracker(self):
        self.service.uv_use_rim_tracking = False
        with patch.object(uv_service_module, 'count_uv_rims', side_effect=AssertionError('new called')):
            result = self.inspect((4, 7), (4, 7))
        self.assertEqual(result.UvAlgorithm, 'hsv_caliper')
        self.assertEqual(result.UvCounts, [4, 7])
        self.assertTrue(result.Result)


if __name__ == '__main__':
    unittest.main()
