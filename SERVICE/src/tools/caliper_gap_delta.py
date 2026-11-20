"""Compare adjacent-disk centre spacing on two scanlines of the same band."""
import math
from collections import Counter

import cv2
import numpy as np


def validate_gap_delta_threshold(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('MAX_CALIPER_GAP_DELTA_PX must be finite and non-negative')
    return value


def compare_caliper_gaps(mids_1, mids_2, max_delta_px, mask=None):
    """Fail explicitly when corresponding disks cannot be paired reliably.

    Production uses connected disk components in the common band mask, so
    translating/tilting a whole row cannot shift identities by one disk. Only
    intervals adjacent on BOTH scans are compared; missing/merged components
    are reported as skipped, never bridged into a different interval.
    Without a mask, require equal counts and unambiguous translation-corrected
    nearest-X correspondence (useful for explicit point-only diagnostics).
    """
    limit = validate_gap_delta_threshold(max_delta_px)
    first = np.asarray(mids_1, dtype=float).reshape(-1, 2)
    second = np.asarray(mids_2, dtype=float).reshape(-1, 2)
    result = dict(threshold_px=limit, counts=[len(first), len(second)],
                  status='invalid', reason=None, max_delta_px=None, gaps=[],
                  matching_method='mask_components' if mask is not None else 'centered_nearest_x',
                  unmatched_points=[len(first), len(second)], skipped_gaps=[0, 0])
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        result['reason'] = 'nonfinite_midpoints'
        return result
    first = first[np.argsort(first[:, 0])]
    second = second[np.argsort(second[:, 0])]
    if min(len(first), len(second)) < 2:
        result['reason'] = 'fewer_than_two_disks'
        return result
    if np.any(np.diff(first[:, 0]) <= 0) or np.any(np.diff(second[:, 0]) <= 0):
        result['reason'] = 'duplicate_disk_positions'
        return result
    if mask is not None:
        if mask.ndim != 2 or not mask.size:
            raise ValueError('Caliper correspondence requires a non-empty band mask')
        _, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)

        def component_ids(points):
            ids = []
            for x, y in np.rint(points).astype(int):
                ids.append(int(labels[y, x]) if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1] else 0)
            counts = Counter(ids)
            return {identity: i for i, identity in enumerate(ids) if identity > 0 and counts[identity] == 1}

        ids_1, ids_2 = component_ids(first), component_ids(second)
        matches = sorted((i, ids_2[identity]) for identity, i in ids_1.items() if identity in ids_2)
        if any(b[1] <= a[1] for a, b in zip(matches, matches[1:])):
            result['reason'] = 'crossed_disk_correspondence'
            return result
    else:
        if len(first) != len(second):
            result['reason'] = 'different_disk_counts'
            return result
        offset = float(np.median(second[:, 0] - first[:, 0]))
        distances = np.abs(first[:, None, 0] - (second[None, :, 0] - offset))
        indices = np.arange(len(first))
        if (not np.array_equal(np.argmin(distances, axis=1), indices)
                or not np.array_equal(np.argmin(distances, axis=0), indices)
                or np.any(np.sum(distances == distances.min(axis=1, keepdims=True), axis=1) != 1)
                or np.any(np.sum(distances == distances.min(axis=0, keepdims=True), axis=0) != 1)):
            result['reason'] = 'ambiguous_disk_correspondence'
            return result
        matches = list(zip(indices.tolist(), indices.tolist()))

    result['unmatched_points'] = [len(first)-len(matches), len(second)-len(matches)]
    for (i, j), (next_i, next_j) in zip(matches, matches[1:]):
        if next_i != i+1 or next_j != j+1:
            continue
        d1 = np.linalg.norm(first[i+1] - first[i])
        d2 = np.linalg.norm(second[j+1] - second[j])
        delta = float(abs(d1 - d2))
        result['gaps'].append(dict(gap_index=i + 1, other_gap_index=j + 1, distance_1_px=float(d1),
                                   distance_2_px=float(d2), delta_px=delta,
                                   passed=delta <= limit))
    result['skipped_gaps'] = [len(first)-1-len(result['gaps']), len(second)-1-len(result['gaps'])]
    if not result['gaps']:
        result['reason'] = 'no_corresponding_adjacent_disks'
        return result
    result['max_delta_px'] = max(gap['delta_px'] for gap in result['gaps'])
    result['status'] = 'ok' if all(gap['passed'] for gap in result['gaps']) else 'ng'
    return result
