import unittest

import numpy as np

from src.tools.caliper_advanced_new import AdvancedMultiEdgeCaliper
from src.tools.white_edge_mask import (WhiteEdgeMaskOptions, add_supported_white_pairs,
                                      connect_white_edge_endpoints, prepare_white_edge_mask)


class WhiteEdgeMaskTests(unittest.TestCase):
    def test_white_caliper_completes_a_cut_stripe_without_counting_outside_noise(self):
        mask = np.zeros((20, 300), np.uint8)
        mask[:, 22:27] = 255  # Entirely outside the original scan, inside its extension.
        mask[:, 140:145] = 255
        mask[:, 268:273] = 255  # Straddles the right endpoint of the original scan.
        caliper = AdvancedMultiEdgeCaliper(min_edge_distance=3, subpixel=True, max_pairs=50,
                                          pair_max_gap=25, thickness_list=[3, 5, 7],
                                          length_rate=.8, polarity='both', angle_deg=0)
        center = (150, 10)
        standard = caliper.measure(mask, center)
        recovered = caliper.measure_white_mask(mask, center, 3, 25, .8, [3, 5, 7])
        self.assertEqual(len(standard['pairs']), 1)
        self.assertEqual(recovered['pairs'][0], standard['pairs'][0])
        self.assertEqual(len(recovered['pairs']), 2)
        pair = recovered['pairs'][1]
        self.assertAlmostEqual((pair['e1']['point'][0] + pair['e2']['point'][0]) / 2, 270, delta=1)

    def test_bridges_short_break_without_joining_neighbor(self):
        mask = np.zeros((64, 80), np.uint8)
        mask[8:56, [20, 23]] = 255
        mask[29:32, 20] = 0
        mask[12, 46] = 255
        original = mask.copy()

        result = prepare_white_edge_mask(mask, WhiteEdgeMaskOptions(bridge=7))

        self.assertTrue(np.array_equal(mask, original))
        self.assertTrue((result[original > 0] == 255).all())
        self.assertTrue((result[29:32, 20] == 255).all())
        self.assertFalse(result[8:56, 21:23].any())
        self.assertEqual(result[12, 45:49].tolist(), [0, 255, 0, 0])

    def test_thickening_allows_caliper_to_read_a_broken_thin_stripe(self):
        mask = np.zeros((80, 300), np.uint8)
        for y in (*range(8, 38), *range(42, 72)):
            mask[y, 84 + y // 2] = 255
        caliper = AdvancedMultiEdgeCaliper(min_edge_distance=4, subpixel=True, max_pairs=50,
                                          pair_max_gap=25, thickness_list=[3, 5, 7],
                                          length_rate=.95, polarity='both', angle_deg=0)
        center = (150, 40)
        self.assertEqual(len(caliper.measure(mask, center)['pairs']), 0)

        result = prepare_white_edge_mask(mask, WhiteEdgeMaskOptions(width=7, bridge=7))
        pairs = caliper.measure(result, center)['pairs']
        self.assertEqual(len(pairs), 1)
        measured_center = (pairs[0]['e1']['point'][0] + pairs[0]['e2']['point'][0]) / 2
        self.assertAlmostEqual(measured_center, 104, delta=2)

    def test_empty_and_short_components_do_not_gain_foreground(self):
        empty = np.zeros((32, 32), np.uint8)
        self.assertFalse(prepare_white_edge_mask(empty).any())
        empty[15:19, 13:15] = 255
        self.assertTrue(np.array_equal(prepare_white_edge_mask(empty), empty))

    def test_rescue_keeps_existing_pairs_and_rejects_unsupported_pairs(self):
        def pair(x):
            return {'e1': {'point': (x - 2, 30)}, 'e2': {'point': (x + 2, 30)}}

        raw = np.zeros((60, 160), np.uint8)
        raw[10:50, 60] = 255
        raw[29:32, 60] = 0
        widened = prepare_white_edge_mask(raw, WhiteEdgeMaskOptions(bridge=7))
        widened[10:50, 100] = 255  # Deliberately simulate a candidate without raw support.
        standard = {'pairs': [pair(35)], 'edges': []}
        candidate = {'pairs': [pair(36), pair(60), pair(100)], 'edges': []}

        output = add_supported_white_pairs(standard, candidate, raw, widened, 30)

        self.assertEqual(output['pairs'], [pair(35), pair(60)])
        self.assertEqual(standard['pairs'], [pair(35)])

    def test_endpoint_connection_follows_skew_and_requires_both_ends(self):
        mask = np.zeros((72, 112), np.uint8)
        for y in (*range(10, 30), *range(38, 65)):
            mask[y, 23 + y // 2] = 255
        mask[8:65, 76] = 255
        mask[10:29, 10] = 255
        result, bridges = connect_white_edge_endpoints(mask)

        self.assertEqual(len(bridges), 1)
        self.assertTrue((result[mask > 0] == 255).all())
        self.assertTrue(all(result[y, 22 + y // 2:25 + y // 2].any() for y in range(30, 38)))
        self.assertFalse(result[29:, 10].any())

    def test_endpoint_connection_does_not_cross_neighbor_or_long_gap(self):
        mask = np.zeros((80, 80), np.uint8)
        mask[10:32, 40] = 255
        mask[42:70, 40] = 255
        mask[10:70, 43] = 255
        result, bridges = connect_white_edge_endpoints(mask)
        self.assertEqual(bridges, [])
        self.assertTrue(np.array_equal(result, mask))

        mask[:, 43] = 0
        result, bridges = connect_white_edge_endpoints(mask, max_gap=9)
        self.assertEqual(bridges, [])
        self.assertTrue(np.array_equal(result, mask))


if __name__ == '__main__':
    unittest.main()
