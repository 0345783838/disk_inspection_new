"""Gap-difference boundaries and correspondence; no models or hardware."""
import unittest

import cv2
import numpy as np

from src.tools.caliper_gap_delta import compare_caliper_gaps, validate_gap_delta_threshold


def mids(xs, y):
    return [[x, y] for x in xs]


class GapDeltaTests(unittest.TestCase):
    def compare(self, xs, ys, limit=10):
        return compare_caliper_gaps(mids(xs, 10), mids(ys, 50), limit)

    def test_parallel_translation_does_not_count_as_crossing(self):
        check = self.compare([0, 50, 100], [8, 58, 108])
        self.assertEqual(check['status'], 'ok')
        self.assertEqual(check['max_delta_px'], 0)

    def test_boundary_passes_and_strictly_greater_fails(self):
        for limit in (10, 20, 30):
            for diff, expected in [(limit-.1, 'ok'), (limit, 'ok'), (limit+.1, 'ng')]:
                with self.subTest(limit=limit, diff=diff):
                    check = self.compare([0, 100, 200], [0, 100 + diff, 200], limit)
                    self.assertEqual(check['status'], expected)
                    self.assertAlmostEqual(check['max_delta_px'], diff)

    def test_same_pixel_delta_is_independent_of_mean_gap(self):
        for gap in [40, 80]:
            self.assertEqual(self.compare([0, gap], [0, gap+11])['status'], 'ng')

    def test_sort_by_position_not_by_gap_size(self):
        check = self.compare([100, 0, 40], [0, 60, 100])
        self.assertEqual(check['status'], 'ng')
        self.assertEqual([g['delta_px'] for g in check['gaps']], [20, 20])

    def test_unequal_counts_are_not_silently_truncated(self):
        check = self.compare([0, 50, 100], [0, 100])
        self.assertEqual(check['status'], 'invalid')
        self.assertEqual(check['reason'], 'different_disk_counts')
        self.assertEqual(check['gaps'], [])

    def test_equal_count_but_different_missing_disks_are_not_compared(self):
        check = self.compare([0, 50, 100], [0, 100, 150])
        self.assertEqual(check['status'], 'invalid')
        self.assertEqual(check['reason'], 'ambiguous_disk_correspondence')

    def test_common_tilt_larger_than_half_pitch_does_not_shift_identities(self):
        self.assertEqual(self.compare([0, 50, 100], [40, 90, 140])['status'], 'ok')

    def test_mask_components_keep_identity_despite_large_common_tilt(self):
        mask = np.zeros((80, 220), np.uint8)
        first, second = mids([20, 70, 120], 10), mids([60, 110, 160], 60)
        for a, b in zip(first, second):
            cv2.line(mask, tuple(a), tuple(b), 255, 9)
        check = compare_caliper_gaps(first, second, 20, mask)
        self.assertEqual(check['status'], 'ok')
        self.assertEqual(check['matching_method'], 'mask_components')
        self.assertEqual(check['max_delta_px'], 0)
        self.assertEqual(check['skipped_gaps'], [0, 0])

    def test_missing_mask_component_only_compares_truly_adjacent_pairs(self):
        mask = np.zeros((80, 180), np.uint8)
        for x in [20, 60, 100, 140]:
            mask[:, x-4:x+5] = 255
        check = compare_caliper_gaps(mids([20, 60, 100, 140], 10),
                                     mids([20, 100, 140], 60), 20, mask)
        self.assertEqual(check['status'], 'ok')
        self.assertEqual(check['unmatched_points'], [1, 0])
        self.assertEqual(check['skipped_gaps'], [2, 1])
        self.assertEqual([(g['gap_index'], g['other_gap_index']) for g in check['gaps']], [(3, 2)])

    def test_merged_or_disconnected_masks_do_not_invent_correspondence(self):
        first, second = mids([20, 70, 120], 10), mids([20, 70, 120], 60)
        for mask in (np.zeros((80, 180), np.uint8), np.full((80, 180), 255, np.uint8)):
            check = compare_caliper_gaps(first, second, 20, mask)
            self.assertEqual(check['status'], 'invalid')
            self.assertEqual(check['reason'], 'no_corresponding_adjacent_disks')

    def test_no_or_one_disk_is_not_a_successful_gap_measurement(self):
        for points in [[], [10]]:
            check = self.compare(points, points)
            self.assertEqual(check['status'], 'invalid')
            self.assertEqual(check['reason'], 'fewer_than_two_disks')

    def test_duplicate_or_nonfinite_coordinates_are_rejected(self):
        self.assertEqual(self.compare([0, 0], [0, 0])['status'], 'invalid')
        for value in [float('nan'), float('inf')]:
            check = self.compare([0, value], [0, 50])
            self.assertEqual(check['reason'], 'nonfinite_midpoints')

    def test_threshold_is_finite_nonnegative_and_zero_is_strict(self):
        for value in [-1, float('nan'), float('inf')]:
            with self.assertRaisesRegex(ValueError, 'MAX_CALIPER_GAP_DELTA_PX'):
                validate_gap_delta_threshold(value)
        self.assertEqual(self.compare([0, 50], [0, 50], 0)['status'], 'ok')
        self.assertEqual(self.compare([0, 50], [0, 50.01], 0)['status'], 'ng')


if __name__ == '__main__':
    unittest.main()
