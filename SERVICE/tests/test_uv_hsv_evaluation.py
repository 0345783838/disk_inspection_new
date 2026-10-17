"""Synthetic diagnostics: never substitute these checks for real UV recall/false-alarm ground truth."""
import unittest
from types import SimpleNamespace
import cv2
import numpy as np
from training.evaluate_uv_hsv import band_details, CANDIDATES
from src.tools.caliper_advanced_new import AdvancedMultiEdgeCaliper


class UvEvaluationTests(unittest.TestCase):
    def setUp(self):
        caliper = AdvancedMultiEdgeCaliper(min_edge_distance=4, subpixel=True, max_pairs=50,
                                          pair_max_gap=25, thickness_list=[3, 5, 7],
                                          length_rate=.95, polarity='both', angle_deg=0)
        def filtered(mask, area):
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < area:
                    cv2.drawContours(mask, [contour], -1, 0, -1)
            return mask
        self.service = SimpleNamespace(caliper=caliper, uv_min_disk_area=20,
            preprocess_uv_image_hsv=lambda band, lower, upper: cv2.inRange(cv2.cvtColor(band, cv2.COLOR_BGR2HSV), lower, upper),
            remove_mask_noise_uv=lambda mask, min_disk_area: filtered(mask, min_disk_area),
            get_caliper_result_debug=lambda mask, center, min_edge_distance, max_edge_distance, length_rate, thickness_list:
                caliper.measure_debug(mask, center, min_edge_distance, max_edge_distance, length_rate, thickness_list))

    def test_v_floor_suppresses_dark_blue_but_retains_bright_uv(self):
        hsv = np.zeros((40, 260, 3), np.uint8)
        hsv[:, 40:48] = (113, 60, 45)  # dark blue distractor
        hsv[:, 100:108] = (113, 90, 170)  # fluorescent edge
        hsv[:, 160:168] = (0, 0, 255)  # white metallic highlight
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        old, _, _ = band_details(self.service, bgr, *CANDIDATES['legacy'], 1)
        new, mask, _ = band_details(self.service, bgr, *CANDIDATES['h105_125_s35_v60'], 1)
        self.assertEqual(old['raw_pairs'], 2)
        self.assertEqual(new['raw_pairs'], 1)
        self.assertFalse(mask[:, 40:48].any())
        self.assertFalse(mask[:, 160:168].any())
        self.assertTrue(mask[:, 100:108].any())

    def test_uv_ablations_do_not_mutate_shared_white_caliper(self):
        hsv = np.zeros((40, 260, 3), np.uint8)
        for x in (40, 100, 160):
            hsv[:, x:x+3] = (113, 90, 170)
        c = self.service.caliper
        before = (c.min_edge_distance, c.pair_max_gap, c.length_rate, list(c.thickness_list))
        detail, _, _ = band_details(self.service, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), *CANDIDATES['legacy'], 1)
        self.assertGreater(detail['caliper_ablations_not_deployed']['thin_edges']['raw_pairs'], detail['raw_pairs'])
        self.assertEqual(before, (c.min_edge_distance, c.pair_max_gap, c.length_rate, c.thickness_list))

    def test_non_uv_color_does_not_become_positive_when_hue_is_widened(self):
        hsv = np.zeros((40, 260, 3), np.uint8)
        hsv[:, 40:48] = (25, 100, 160)
        detail, mask, _ = band_details(self.service, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), *CANDIDATES['h105_125_s35_v60'], 2)
        self.assertEqual(detail['raw_pairs'], 0)
        self.assertFalse(mask.any())


if __name__ == '__main__':
    unittest.main()
