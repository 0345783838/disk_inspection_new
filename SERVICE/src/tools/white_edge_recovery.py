"""Recover locally supported White caliper pairs from weak or reflected masks."""

import math

import cv2
import numpy as np
from scipy.signal import find_peaks

from .white_edge_mask import WhiteEdgeMaskOptions, prepare_white_edge_mask


WEAK_CONFIDENCE = .05
WEAK_MASK_OPTIONS = WhiteEdgeMaskOptions(width=5, bridge=1)


def _pair_center(pair):
    return (pair['e1']['point'][0] + pair['e2']['point'][0]) / 2


def _append_pairs(original, candidates, max_pairs):
    pairs = list(original['pairs'])
    centers = [_pair_center(pair) for pair in pairs]
    added_edges = []
    for pair in sorted(candidates, key=_pair_center):
        if len(pairs) >= max_pairs:
            break
        x = _pair_center(pair)
        if all(abs(x - old) > 8 for old in centers):
            pairs.append(pair)
            centers.append(x)
            added_edges.extend((pair['e1'], pair['e2']))
    if not added_edges:
        return original
    edges = {(edge['point'], edge['sign']): edge for edge in added_edges}
    edges.update({(edge['point'], edge['sign']): edge for edge in original['edges']})
    return {**original, 'pairs': sorted(pairs, key=_pair_center),
            'edges': sorted(edges.values(), key=lambda edge: edge['point'][0])}


def _trace_lengths(mask, scans):
    """Follow mutually unique adjacent runs, with a center shift up to 3 px per row."""
    height = mask.shape[0]
    padded = np.pad(mask > 0, ((0, 0), (1, 1)))
    rows, starts = np.where(~padded[:, :-1] & padded[:, 1:])
    _, ends = np.where(padded[:, :-1] & ~padded[:, 1:])
    offsets = np.searchsorted(rows, np.arange(height + 1))
    up = np.ones(len(starts), np.int32)
    parents = np.full(len(starts), -1, np.int32)
    for y in range(1, height):
        a, b, c = offsets[y-1:y+2]
        left_prev, right_prev = starts[a:b, None], ends[a:b, None]
        left_curr, right_curr = starts[None, b:c], ends[None, b:c]
        adjacent = (left_prev <= right_curr) & (left_curr <= right_prev)
        aligned = np.abs(left_prev + right_prev - left_curr - right_curr) <= 6
        prev, curr = np.where(adjacent & aligned & (adjacent.sum(axis=1, keepdims=True) == 1)
                              & (adjacent.sum(axis=0, keepdims=True) == 1))
        up[b + curr] = up[a + prev] + 1
        parents[b + curr] = a + prev
    down = np.ones(len(starts), np.int32)
    for y in range(height - 1, 0, -1):
        b, c = offsets[y:y+2]
        child = np.arange(b, c)[parents[b:c] >= 0]
        down[parents[child]] = down[child] + 1
    return {y: list(zip(starts[offsets[y]:offsets[y+1]], ends[offsets[y]:offsets[y+1]] - 1,
                        up[offsets[y]:offsets[y+1]], down[offsets[y]:offsets[y+1]])) for y in scans}


def _spans_both_scans(traces, x, y, other, height, max_width=None):
    nearby = sorted((max(a-x, x-b, 0), above, below, b-a+1)
                    for a, b, above, below in traces if a-1 <= x <= b+1)
    if not nearby or (len(nearby) > 1 and (max_width is not None or nearby[0][0] == nearby[1][0])):
        return False
    _, above, below, width = nearby[0]
    margin = math.ceil(.08 * height)
    return ((max_width is None or width <= max_width)
            and above + below - 1 >= math.ceil(.20 * height)
            and above - 1 >= min(margin, y) and below - 1 >= min(margin, height-1-y)
            and y-above+1 <= other <= y+below-1)


def _has_bright_trace(gray, x, y):
    cx = round(x)
    h, w = gray.shape
    if not (14 <= cx < w-14 and 4 <= y < h-4):
        return False
    window = gray[y-4:y+5, cx-14:cx+15].astype(np.int16)
    center = window[:, 10:19].max(axis=1)
    left = window[:, 2:10].min(axis=1)
    right = window[:, 19:27].min(axis=1)
    return np.count_nonzero(center - np.maximum(left, right) >= 6) >= 7


def _suspicious_points(gray, raw_mask, scans, bounds, pairs, num_tiles):
    height, width = gray.shape
    source = gray.astype(np.float32)
    smooth = cv2.GaussianBlur(source, (3, 1), 0)
    wide = cv2.blur(source, (5, 1))
    response = np.maximum.reduce([smooth - np.maximum(np.roll(wide, lag, axis=1), np.roll(wide, -lag, axis=1))
                                  for lag in (5, 8)])
    xs = np.arange(width)
    dy = np.arange(-20, 21)[:, None]
    coordinates = [np.clip(xs + np.rint(dy * slope).astype(int), 0, width-1)
                   for slope in (-.8, -.6, -.4, -.2, 0., .2, .4, .6, .8)]
    seams = np.arange(1, num_tiles) * (width // num_tiles)
    near_seam = np.min(np.abs(xs[:, None] - seams), axis=1) <= 64
    left, right = bounds
    result = {}
    for y in scans:
        if y < 20 or y + 20 >= height:
            result[y] = []
            continue
        # The 35th percentile of 41 samples is exactly the 15th sorted value.
        scores = np.maximum.reduce([np.partition(response[y + dy, columns], 14, axis=0)[14]
                                    for columns in coordinates])
        scores[:left] = 0
        scores[right+1:] = 0
        for pair in pairs[y]['pairs']:
            x = round(_pair_center(pair))
            scores[max(0, x-8):x+9] = 0
        occupied = cv2.dilate(raw_mask[y:y+1], np.ones((1, 9), np.uint8))[0] > 0
        scores[occupied] = 0
        hits = find_peaks(scores, height=18, distance=7)[0]
        result[y] = hits[near_seam[hits]]
    return result


def _reflect_tiles(segmentor, image, hits, confidence, iou):
    height, width = image.shape[:2]
    step = width // segmentor.num_splits
    selected = sorted({min(segmentor.num_splits-1, (width-1-x)//step) for x in hits})
    crops, spans = segmentor.split_image(cv2.flip(image, 1))
    mask = np.zeros((height, width), np.uint8)
    coverage = np.zeros(width, bool)
    for index in selected:
        left, right = spans[index]
        _, _, _, masks = segmentor.segment_objects_debug(crops[index], confidence, iou)
        if len(masks):
            segmentor.paste_tile_mask(mask, segmentor.merge_masks(masks) * 255, left, right, index)
        end = width if index == segmentor.num_splits-1 else (index+1)*step
        coverage[index*step:end] = True
    return cv2.flip(mask, 1), coverage[::-1]


def recover_white_pairs(segmentor, image, raw_mask, weak_mask, scans, measurements,
                        clean, measure, length_rate, max_pairs, confidence, iou):
    """Retain the original measurements and add only confirmed recovery pairs."""
    height, width = raw_mask.shape
    if scans[0] == scans[1]:
        return measurements
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = dict(measurements)
    weak_clean = clean(prepare_white_edge_mask(weak_mask, WEAK_MASK_OPTIONS))
    weak_traces = _trace_lengths(weak_mask, scans)
    for y, other in (scans, scans[::-1]):
        candidates = []
        for pair in measure(weak_clean, (width//2, y))['pairs']:
            x = _pair_center(pair)
            if (_spans_both_scans(weak_traces[y], x, y, other, height, max_width=10)
                    and _has_bright_trace(gray, x, y)):
                candidates.append(pair)
        result[y] = _append_pairs(result[y], candidates, max_pairs)

    if segmentor.num_splits < 2 or width < segmentor.num_splits:
        return result
    half_length = (int(length_rate * width) - 1) / 2
    bounds = (math.ceil(width//2 - half_length), math.floor(width//2 + half_length))
    hits = _suspicious_points(gray, raw_mask, scans, bounds, result, segmentor.num_splits)
    positions = [int(x) for row in hits.values() for x in row]
    if not positions:
        return result
    reflected, known = _reflect_tiles(segmentor, image, positions, confidence, iou)
    reflected_traces = _trace_lengths(reflected, scans)
    reflected_clean = clean(reflected.copy())
    for y, other in (scans, scans[::-1]):
        candidates = []
        for pair in measure(reflected_clean, (width//2, y))['pairs']:
            x = _pair_center(pair)
            if not any(abs(x - position) <= 14 for position in hits[y]):
                continue
            left, right = (pair[key]['point'][0] for key in ('e1', 'e2'))
            if (known[max(0, int(left)-2):math.ceil(right)+3].all()
                    and _spans_both_scans(reflected_traces[y], x, y, other, height)
                    and _has_bright_trace(gray, x, y)):
                candidates.append(pair)
        result[y] = _append_pairs(result[y], candidates, max_pairs)
    return result
