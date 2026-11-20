"""Prepare a binary White segmentation mask for horizontal caliper scans."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class WhiteEdgeMaskOptions:
    width: int = 5
    bridge: int = 1
    min_trace_height: int = 18
    gutter: int = 2


def _runs(mask):
    padded = np.pad(mask, ((0, 0), (1, 1)), constant_values=False)
    rows, starts = np.where(padded[:, 1:] & ~padded[:, :-1])
    _, ends = np.where(padded[:, :-1] & ~padded[:, 1:])
    return rows, starts, ends


def _close_short_gaps(foreground, span):
    if span < 3:
        return foreground.copy()
    offsets = np.arange(span) - span // 2
    filled = foreground.copy()
    source = foreground.astype(np.uint8)

    # Propagate a visible gutter through short breaks too. This prevents a
    # diagonal bridge from using a nearby parallel edge as the other endpoint.
    rows, starts, ends = _runs(foreground)
    gutters = np.zeros(foreground.shape, np.uint8)
    for i in np.flatnonzero((rows[1:] == rows[:-1]) & (starts[1:] - ends[:-1] <= 12)):
        gutters[rows[i], ends[i]:starts[i + 1]] = 1
    blocked = gutters.astype(bool)
    for slope in (0., -.5, .5, -1., 1.):
        kernel = np.zeros((span, span), np.uint8)
        kernel[np.arange(span), span // 2 + np.rint(offsets * slope).astype(int)] = 1
        filled |= cv2.morphologyEx(source, cv2.MORPH_CLOSE, kernel).astype(bool)
        blocked |= cv2.morphologyEx(gutters, cv2.MORPH_CLOSE, kernel).astype(bool)
    return foreground | (filled & ~blocked)


def prepare_white_edge_mask(mask, options=WhiteEdgeMaskOptions()):
    """Close short vertical/slanted breaks and widen supported narrow runs.

    Preserve all source foreground. Isolated low-height components receive no
    additions. Widening leaves a gutter between separate runs on any row.
    The returned uint8 mask uses 0 and 255 for the existing area cleanup/caliper.
    """
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError('White edge mask must be a nonempty 2-D image')
    if options.width < 1 or options.bridge < 1 or options.bridge % 2 == 0:
        raise ValueError('Width must be positive and bridge must be positive and odd')
    if options.min_trace_height < 1 or options.gutter < 1:
        raise ValueError('Trace height and horizontal gutter must be positive')
    original = mask > 0
    if not original.any():
        return np.zeros(mask.shape, np.uint8)
    filled = _close_short_gaps(original, options.bridge)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(filled.astype(np.uint8), connectivity=8)
    tall = stats[:, cv2.CC_STAT_HEIGHT] >= options.min_trace_height
    tall[0] = False
    supported = tall[labels]
    output = original | (filled & supported)

    rows, starts, ends = _runs(output)
    eligible = supported[rows, starts] & (ends - starts < options.width)
    for i in np.flatnonzero(eligible):
        y, start, end = int(rows[i]), int(starts[i]), int(ends[i])
        missing = options.width - (end - start)
        left = max(0, start - missing // 2)
        right = min(mask.shape[1], end + (missing + 1) // 2)
        if i > 0 and rows[i - 1] == y:
            gap = start - int(ends[i - 1])
            left = max(left, start - max(0, (gap - options.gutter) // 2))
        if i + 1 < len(rows) and rows[i + 1] == y:
            gap = int(starts[i + 1]) - end
            right = min(right, end + max(0, (gap - options.gutter + 1) // 2))
        output[y, left:right] = True
    return output.astype(np.uint8) * 255


def _pair_center(pair):
    return (pair['e1']['point'][0] + pair['e2']['point'][0]) / 2


def _supported_pair(source, refined, x, y):
    """Check that this pair follows a narrow, locally aligned raw-mask trace."""
    height, width = source.shape
    x0, x1 = max(0, round(x) - 14), min(width, round(x) + 15)
    y0, y1 = y - 7, y + 8
    if y0 < 0 or y1 > height or x0 >= x1:
        return False
    inside = refined[y0:y1, x0:x1] > 0
    _, components = cv2.connectedComponents(inside.astype(np.uint8))
    at_scan = np.flatnonzero(inside[7])
    if not len(at_scan):
        return False
    closest = at_scan[np.argmin(np.abs((at_scan + x0) - x))]
    if abs(closest + x0 - x) > 1.5:
        return False
    component = components == components[7, closest]
    mask_support = component & (source[y0:y1, x0:x1] > 0)
    row_index = np.flatnonzero(mask_support.any(axis=1))
    above, below = row_index[row_index < 7], row_index[row_index > 7]
    if len(above) < 4 or len(below) < 4 or above[-1] < 4 or below[0] > 10:
        return False
    if np.max(mask_support.sum(axis=1)) > 12:
        return False
    xs = np.array([np.median(np.flatnonzero(mask_support[row])) + x0 for row in row_index])
    ys = row_index - 7
    slope, intercept = np.polyfit(ys, xs, 1)
    residual = xs - (slope * ys + intercept)
    return abs(intercept - x) <= 2.5 and np.max(np.abs(residual)) <= 2.5


def add_supported_white_pairs(standard, widened, raw_mask, widened_mask, scan_y, *, max_pairs=50):
    """Keep standard measurements and add widened-mask pairs with raw trace support."""
    pairs = list(standard['pairs'])
    centers = [_pair_center(pair) for pair in pairs]
    added_edges = []
    for pair in widened['pairs']:
        if len(pairs) >= max_pairs:
            break
        x = _pair_center(pair)
        if all(abs(x - existing) > 8 for existing in centers) and _supported_pair(raw_mask, widened_mask, x, scan_y):
            pairs.append(pair)
            centers.append(x)
            added_edges.extend((pair['e1'], pair['e2']))
    if len(pairs) == len(standard['pairs']):
        return standard
    return {**standard, 'pairs': sorted(pairs, key=_pair_center),
            'edges': sorted([*standard.get('edges', ()), *added_edges], key=lambda edge: edge['point'][0])}


def _trace_endpoints(labels, stats, ident):
    left, top, width, height, _ = map(int, stats[ident])
    if height < 10:
        return None
    component = labels[top:top + height, left:left + width] == ident
    endpoints = []
    for start in (0, height - 8):
        local = component[start:start + 8]
        # Branching or wide mask ends provide poor direction estimates.
        _, starts, ends = _runs(local)
        if len(starts) != len(local) or np.max(ends - starts) > 12:
            return None
        centers = left + (starts + ends - 1) / 2
        slope, offset = np.polyfit(np.arange(8), centers, 1)
        if np.max(np.abs(centers - (slope * np.arange(8) + offset))) > 1.75 or abs(slope) > 1.5:
            return None
        end = 7 if start else 0
        endpoints.append((float(centers[end]), top + start + end, float(slope)))
    return endpoints


def connect_white_edge_endpoints(mask, *, max_gap=12):
    """Join unambiguous short breaks without extending a single unsupported end.

    Fit a direction from both parent fragments. A proposed path needs a clear
    corridor, and each endpoint must have exactly one compatible partner.
    Returns an independent 0/255 mask and the accepted bridge endpoints.
    """
    if mask.ndim != 2 or mask.size == 0 or max_gap < 1:
        raise ValueError('Expected a nonempty 2-D mask and a positive gap limit')
    _, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    endpoints = [(ident, ends) for ident in range(1, len(stats))
                 if (ends := _trace_endpoints(labels, stats, ident)) is not None]
    starters = {}
    for ident, (top, _) in endpoints:
        starters.setdefault(top[1], []).append((ident, top))
    proposals = []
    for upper, (_, (x0, y0, slope0)) in endpoints:
        for y1 in range(y0 + 2, y0 + max_gap + 2):
            for lower, (x1, _, slope1) in starters.get(y1, ()):
                length = y1 - y0
                if upper == lower or abs(slope0 - slope1) > .35:
                    continue
                if max(abs(x0 + slope0 * length - x1), abs(x1 - slope1 * length - x0)) > 3:
                    continue
                # Cubic Hermite interpolation keeps the tangent at both ends.
                ys = np.arange(y0, y1 + 1)
                t = (ys - y0) / length
                xs = ((2*t**3 - 3*t**2 + 1) * x0 + (t**3 - 2*t**2 + t) * length * slope0
                      + (-2*t**3 + 3*t**2) * x1 + (t**3 - t**2) * length * slope1)
                xs = np.rint(xs).astype(int)
                if np.any(xs < 0) or np.any(xs >= mask.shape[1]):
                    continue
                # Check near the path as well as on it. Avoid joining a trace
                # that passes close to a third component, especially a neighbor.
                corridor = labels[ys[:, None], np.clip(xs[:, None] + np.arange(-3, 4), 0, mask.shape[1] - 1)]
                if np.any((corridor != 0) & (corridor != upper) & (corridor != lower)):
                    continue
                proposals.append((upper, lower, xs, ys))

    upper_counts = {}
    lower_counts = {}
    for upper, lower, _, _ in proposals:
        upper_counts[upper] = upper_counts.get(upper, 0) + 1
        lower_counts[lower] = lower_counts.get(lower, 0) + 1
    output = (mask > 0).astype(np.uint8) * 255
    added = np.zeros(mask.shape, np.uint8)
    bridges = []
    for upper, lower, xs, ys in proposals:
        if upper_counts[upper] != 1 or lower_counts[lower] != 1:
            continue
        if added[ys[:, None], np.clip(xs[:, None] + np.arange(-3, 4), 0, mask.shape[1] - 1)].any():
            continue
        points = np.column_stack([xs, ys]).astype(np.int32)
        cv2.polylines(output, [points], False, 255, 1)
        cv2.polylines(added, [points], False, 255, 1)
        added[ys[1:-1], xs[1:-1]] = 255
        bridges.append({'start': points[0].tolist(), 'end': points[-1].tolist()})
    return output, bridges
