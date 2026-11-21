"""Algorithmic fixtures only, not a substitute for labeled camera UV images."""
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from src.tools.uv_rim_counter import count_uv_rims, uv_masks, fan_maps, compare_bright_ridges, RimSettings


def image_with_rims(xs, width=320, height=100, thickness=1, value=180):
    hsv = np.zeros((height, width, 3), np.uint8)
    for x in xs:
        hsv[:, x:x+thickness] = (113, 75, value)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class UvRimCounterTests(unittest.TestCase):
    def setUp(self):
        # Mechanical fixtures must stay reproducible when users tune config.env.
        loader = patch('src.tools.uv_rim_counter.load_rim_settings', return_value=RimSettings())
        loader.start()
        self.addCleanup(loader.stop)

    def test_counts_single_pixel_and_thick_rims_once_each(self):
        for thickness in (1, 3, 7):
            result, _ = count_uv_rims(image_with_rims([50, 115, 200], thickness=thickness), 1)
            self.assertEqual(result['count'], 3)

    def test_bright_thin_secondary_ridge_is_rejected_between_broad_rims(self):
        image = image_with_rims([35, 95, 155, 215, 275], thickness=5)
        image[:, 63] = image[0, 36]
        baseline, _ = count_uv_rims(image, 1, RimSettings())
        filtered, _ = count_uv_rims(image, 1, RimSettings(min_ridge_width_px=2.5))
        rejected = [t for t in filtered['rejected'] if t['reasons'] == ['thin secondary ridge']]
        self.assertEqual((baseline['count'], filtered['count']), (6, 5))
        self.assertEqual(len(rejected), 1)
        self.assertLessEqual(abs(rejected[0]['center_x'] - 63), 2)
        self.assertAlmostEqual(rejected[0]['width_px'], 1.)

    def test_preserves_a_thin_rim_in_an_open_slot_or_sparse_band(self):
        settings = RimSettings(min_ridge_width_px=2.5)
        image = image_with_rims([35, 95, 215, 275, 335], width=380, thickness=5)
        image[:, 155] = image[0, 36]
        full, _ = count_uv_rims(image, 1, settings)
        sparse, _ = count_uv_rims(image_with_rims([80, 160]), 1, settings)
        self.assertEqual((full['count'], sparse['count']), (6, 2))

    def test_count_is_not_forced_to_25(self):
        for xs in ([], [80], [50, 150], [40, 100, 160, 220]):
            result, _ = count_uv_rims(image_with_rims(xs), 1)
            self.assertEqual(result['count'], len(xs))

    def test_close_distinct_rims_remain_two(self):
        result, _ = count_uv_rims(image_with_rims([80, 86]), 1)
        self.assertEqual(result['count'], 2)

    def test_shared_bright_ridge_can_have_different_sampled_columns(self):
        value = np.zeros((100, 40), np.uint8)
        value[:, 12:20] = 220
        first = np.full(100, 12)
        second = np.r_[np.full(70, 19), np.full(30, -1)]
        self.assertEqual(compare_bright_ridges(value, first, second, 12), (True, False))
        value[56:70, 14:17] = 130
        self.assertEqual(compare_bright_ridges(value, first, second, 12), (True, True))

    def test_crossing_rims_do_not_share_most_bright_rows(self):
        value = np.full((100, 40), 210, np.uint8)
        first = np.r_[np.full(40, 12), np.full(60, -1)]
        second = np.r_[np.full(30, -1), np.full(70, 19)]
        self.assertEqual(compare_bright_ridges(value, first, second, 12), (False, False))

    def test_neutral_core_requires_blue_halo(self):
        hsv = np.zeros((100, 320, 3), np.uint8)
        hsv[:, 60:67] = (113, 75, 150)
        hsv[:, 62:65] = (0, 0, 255)
        hsv[:, 200:205] = (0, 0, 255)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        mask, _, _ = uv_masks(image)
        self.assertTrue(mask[:, 62:65].all())
        self.assertFalse(mask[:, 200:205].any())
        result, _ = count_uv_rims(image, 1)
        self.assertEqual(result['count'], 1)

    def test_dark_blue_rim_is_not_a_bright_uv_positive(self):
        result, _ = count_uv_rims(image_with_rims([50, 115], value=60), 1)
        self.assertEqual(result['count'], 0)

    def test_mask_gaps_do_not_require_a_connected_component(self):
        image = image_with_rims([80])
        image[30:38] = 0
        image[60:67] = 0
        result, _ = count_uv_rims(image, 1)
        self.assertEqual(result['count'], 1)

    def test_short_reflection_is_not_counted_as_another_disk(self):
        image = image_with_rims([80, 160])
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[65:, 230:233] = (113, 75, 180)
        result, _ = count_uv_rims(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), 1)
        self.assertEqual(result['count'], 2)
        self.assertGreater(len(result['rejected']), 0)

    def test_opposed_curved_reflection_does_not_add_an_instance(self):
        image = image_with_rims([70, 140, 220])
        cv2.line(image, (110, 99), (125, 55), (190, 120, 75), 2)
        result, _ = count_uv_rims(image, 1)
        self.assertEqual(result['count'], 3)

    def test_input_image_is_unchanged(self):
        image = image_with_rims([80])
        before = image.copy()
        count_uv_rims(image, 2)
        np.testing.assert_array_equal(image, before)

    def test_more_than_25_instances_are_not_capped(self):
        result, _ = count_uv_rims(image_with_rims(list(range(20, 600, 20)), width=640), 1)
        self.assertEqual(result['count'], 29)

    def test_uniform_blue_background_is_not_a_rim(self):
        hsv = np.full((100, 320, 3), (113, 75, 180), np.uint8)
        result, _ = count_uv_rims(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), 1)
        self.assertEqual(result['count'], 0)

    def test_fan_shaped_primary_rims_survive_full_length_opposed_shadow(self):
        image = np.zeros((150, 600, 3), np.uint8)
        mx, _, _, _ = fan_maps(image.shape[:2], -.35, 0., -.04)
        for x in (80, 240, 460):
            points = np.column_stack((mx[:, x], np.arange(150))).round().astype(np.int32)
            cv2.polylines(image, [points], False, (190, 120, 75), 2)
        cv2.line(image, (180, 0), (125, 149), (190, 120, 75), 2)
        result, _ = count_uv_rims(image, 1)
        self.assertEqual(result['count'], 3)

    def test_broad_alignment_recovers_rims_from_a_bad_hough_fit(self):
        for band, slope, tilt, xs in (
                (1, -.39, -.06, [54, 124, 236, 390, 475, 535]),
                (2, .42, .05, [375, 448, 515, 558])):
            with self.subTest(band=band):
                image = np.zeros((170, 640, 3), np.uint8)
                mx, _, _, _ = fan_maps(image.shape[:2], slope, tilt, -.06)
                for x in xs:
                    points = np.column_stack((mx[:, x], np.arange(len(mx)))).round().astype(np.int32)
                    cv2.polylines(image, [points], False, (190, 120, 75), 2)
                with patch('src.tools.uv_rim_counter.estimate_fan',
                           return_value=(np.sign(slope) * .02, -.06, True)):
                    result, _ = count_uv_rims(image, band)
                self.assertEqual(result['count'], len(xs))

    def test_sparse_opposed_outer_arc_does_not_set_the_fan_tilt(self):
        image = np.zeros((180, 700, 3), np.uint8)
        cv2.line(image, (565, 0), (587, 179), (190, 120, 75), 2)
        result, _ = count_uv_rims(image, 1)
        self.assertEqual(result['count'], 0)


if __name__ == '__main__':
    unittest.main()
