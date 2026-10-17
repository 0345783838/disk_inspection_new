"""Experimental UV-only rim tracking. Not connected to the production service.

Blue HSV halos admit adjacent low-saturation bright cores. A fan-shaped warp
aligns the long primary rims; reversed/short reflected arcs do not align.
Count ridge tracks, not the two boundaries of a binary strip. No slot/count prior.
"""
from dataclasses import dataclass, fields
from pathlib import Path
import math
import cv2
import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class RimSettings:
    hue_low: int = 100
    hue_high: int = 135
    saturation_low: int = 12
    value_low: int = 35
    strong_value: int = 85
    weak_contrast: float = 4
    strong_contrast: float = 12
    min_coverage: float = .60
    min_span: float = .80
    min_strong_coverage: float = .35
    min_separation: int = 5
    core_saturation_max: int = 35
    core_value_min: int = 110

    def __post_init__(self):
        limits = {'hue_low': 179, 'hue_high': 179, 'saturation_low': 255,
                  'value_low': 255, 'strong_value': 255,
                  'core_saturation_max': 255, 'core_value_min': 255}
        for name, maximum in limits.items():
            value = getattr(self, name)
            if not isinstance(value, int) or not 0 <= value <= maximum:
                raise ValueError(f'UV_RIM_{name.upper()} must be an integer in 0..{maximum}')
        if self.hue_low > self.hue_high:
            raise ValueError('UV_RIM_HUE_LOW must not exceed UV_RIM_HUE_HIGH')
        for name in ('weak_contrast', 'strong_contrast'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'UV_RIM_{name.upper()} must be finite and nonnegative')
        for name in ('min_coverage', 'min_span', 'min_strong_coverage'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f'UV_RIM_{name.upper()} must be in (0, 1]')
        if not isinstance(self.min_separation, int) or self.min_separation < 1:
            raise ValueError('UV_RIM_MIN_SEPARATION must be an integer >= 1')


def load_rim_settings(config_path=None):
    """Read a fresh config snapshot without modifying global decouple.config.

    Environment overrides follow python-decouple. Missing keys retain original
    defaults; a missing file or malformed value fails explicitly.
    """
    from decouple import Config, RepositoryEnv
    path = Path(config_path) if config_path is not None else Path(__file__).resolve().parents[2] / 'config' / 'config.env'
    config = Config(RepositoryEnv(str(path)))
    defaults = RimSettings()
    values = {}
    for field in fields(defaults):
        default = getattr(defaults, field.name)
        caster = field.type
        key = 'UV_RIM_' + field.name.upper()
        try:
            values[field.name] = config(key, default=default, cast=caster)
        except (ValueError, TypeError) as error:
            raise ValueError(f'Invalid {key} in {path}: expected {caster.__name__}') from error
    return RimSettings(**values)


def uv_masks(image, settings=None):
    settings = load_rim_settings() if settings is None else settings
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    background = cv2.GaussianBlur(v.astype(np.float32), (31, 1), 0)
    contrast = v.astype(np.float32) - background
    blue = (h >= settings.hue_low) & (h <= settings.hue_high) & (s >= settings.saturation_low) & (v >= settings.value_low)
    nearby_blue = cv2.dilate(blue.astype(np.uint8), np.ones((3, 7), np.uint8)) > 0
    # Never accept arbitrary white metal: a bright core must touch a blue halo.
    core = nearby_blue & (s <= settings.core_saturation_max) & (v >= settings.core_value_min) & (contrast >= settings.strong_contrast)
    weak = (blue & (contrast >= settings.weak_contrast)) | core
    strong = weak & (v >= settings.strong_value) & (contrast >= settings.strong_contrast)
    weak = cv2.morphologyEx(weak.astype(np.uint8)*255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return weak, strong.astype(np.uint8)*255, contrast


def estimate_fan(strong, band):
    height, width = strong.shape
    lines = cv2.HoughLinesP(cv2.Canny(strong, 70, 140), 1, np.pi/360,
                            threshold=max(12, int(height*.15)),
                            minLineLength=max(10, int(height*.55)), maxLineGap=max(3, int(height*.06)))
    sign = -1 if band == 1 else 1
    samples = []
    for x1, y1, x2, y2 in ([] if lines is None else lines[:, 0]):
        if abs(y2-y1) < height*.55:
            continue
        slope = (float(x2)-x1)/(y2-y1)
        position = ((x1+x2)/2-width/2)/(width/2)
        if abs(slope) < .8 and abs(position) > .25 and sign*slope*position > 0:
            samples.append((position, slope))
    if not samples:
        return 0., 0., False
    samples = np.asarray(samples)
    if len(samples) < 3:
        # Sparse UV trays must not require a full row of fluorescent disks.
        return float(np.median(samples[:, 1]/samples[:, 0])), 0., True
    a, b = np.polyfit(samples[:, 0], samples[:, 1], 1)
    residual = np.abs(samples[:, 1]-(a*samples[:, 0]+b))
    keep = residual <= max(.025, float(np.median(residual))*2.5)
    if keep.sum() >= 3:
        a, b = np.polyfit(samples[keep, 0], samples[keep, 1], 1)
    if sign*a <= 0 or abs(b) > .2:
        return 0., 0., False
    return float(a), float(b), True


def fan_maps(shape, a, b, curvature):
    height, width = shape
    y, x = np.indices(shape, dtype=np.float32)
    dy = y-(height-1)/2
    scale = 1+a/(width/2)*dy+curvature*(dy/max(1, height))**2
    source_x = width/2+(x-width/2)*scale+b*dy
    return source_x.astype(np.float32), y, scale[:, 0], dy[:, 0]


def select_alignment(strong, band):
    a, b, fitted = estimate_fan(strong, band)
    best = None
    # Tune geometric coherence only: never reward a desired number of peaks.
    for factor in ((.8, 1., 1.2) if fitted else (1.,)):
        for curve in ((-.10, -.08, -.06, -.04, -.02, 0., .02) if fitted else (0.,)):
            mx, my, scale, dy = fan_maps(strong.shape, a*factor, b, curve)
            aligned = cv2.remap(strong, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            profile = (aligned > 0).mean(axis=0)
            score = float(np.sum(profile**3))
            if best is None or score > best[0]:
                best = (score, a*factor, b, curve, mx, my, scale, dy)
    return best, fitted


def count_uv_rims(image, band, settings=None):
    if image is None or image.ndim != 3 or min(image.shape[:2]) < 8 or band not in (1, 2):
        raise ValueError('Expected a BGR UV band image and band 1/2')
    settings = load_rim_settings() if settings is None else settings
    weak, strong, contrast = uv_masks(image, settings)
    alignment, fitted = select_alignment(strong, band)
    _, a, b, curve, mx, my, scale, dy = alignment
    aligned_weak = cv2.remap(weak, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    aligned_strong = cv2.remap(strong, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    aligned_value = cv2.remap(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2], mx, my, cv2.INTER_LINEAR)
    height, width = weak.shape
    wide = cv2.dilate(aligned_weak, np.ones((1, 3), np.uint8)) > 0
    profile = wide.mean(axis=0).astype(np.float32)
    profile = cv2.GaussianBlur(profile.reshape(1, -1), (5, 1), .8).ravel()
    peaks, _ = find_peaks(np.pad(profile, (1, 1)), distance=settings.min_separation,
                          height=.15, prominence=.08)
    peaks = peaks-1
    tracks, rejected, uncertain = [], [], []
    yy = np.arange(height)
    for center in peaks:
        x0, x1 = max(0, center-5), min(width, center+6)
        candidate = aligned_value[:, x0:x1].astype(float)
        candidate[aligned_weak[:, x0:x1] == 0] = -1
        xs = x0+np.argmax(candidate, axis=1)
        supported = candidate.max(axis=1) >= 0
        if supported.sum() < 5:
            continue
        poly = np.polyfit(yy[supported], xs[supported], 2)
        fitted_x = np.polyval(poly, yy)
        # Reject broad background/branches rather than freely jumping to a neighbor.
        supported &= np.abs(xs-fitted_x) <= 3
        if supported.sum() >= 5:
            poly = np.polyfit(yy[supported], xs[supported], 2)
            fitted_x = np.polyval(poly, yy)
        coverage = float(supported.mean())
        rows = np.flatnonzero(supported)
        span = float((rows[-1]-rows[0]+1)/height) if len(rows) else 0.
        strong_rows = np.any(aligned_strong[:, x0:x1] > 0, axis=1) & supported
        strong_coverage = float(strong_rows.mean())
        reasons = []
        if coverage < settings.min_coverage: reasons.append('short/discontinuous')
        if span < settings.min_span: reasons.append('not spanning band')
        if strong_coverage < settings.min_strong_coverage: reasons.append('no sufficient bright UV seed')
        observed_x = width/2+(fitted_x-width/2)*scale+b*dy
        observed_slope = float((observed_x[-1]-observed_x[0])/max(1, height-1))
        expected_slope = float(a*((observed_x[height//2]-width/2)/(width/2))+b)
        if fitted and abs(observed_slope-expected_slope) > max(.08, abs(expected_slope)*.30):
            reasons.append('opposed/off-direction arc')
        points = np.column_stack((observed_x, yy))
        track = {'center_x': int(center), 'coverage': coverage, 'span': span,
                 'strong_coverage': strong_coverage, 'observed_slope': observed_slope,
                 'expected_slope': expected_slope, 'points': points.tolist(), 'reasons': reasons}
        if reasons == ['no sufficient bright UV seed'] and strong_coverage >= .08:
            uncertain.append(track)
        else:
            (rejected if reasons else tracks).append(track)
    # Two profiles tracing the same physical ridge are one instance, not two edges.
    unique = []
    for track in sorted(tracks, key=lambda t: t['coverage']+t['strong_coverage'], reverse=True):
        if any(np.mean(np.abs(np.array(track['points'])[:, 0]-np.array(other['points'])[:, 0])) < settings.min_separation
               for other in unique):
            track['reasons'] = ['duplicate ridge']
            rejected.append(track)
        else:
            unique.append(track)
    tracks = sorted(unique, key=lambda t: t['center_x'])
    kept = np.zeros_like(weak)
    overlay = image.copy()
    for index, track in enumerate(tracks, 1):
        points = np.rint(track['points']).astype(np.int32)
        corridor = np.zeros_like(weak)
        cv2.polylines(corridor, [points], False, 255, 7)
        kept[(corridor > 0) & (weak > 0)] = 255
        cv2.polylines(overlay, [points], False, (0, 230, 0), 1)
        cx, cy = map(int, points[height//2])
        cv2.circle(overlay, (cx, cy), 3, (0, 255, 255), -1)
        cv2.putText(overlay, str(index), (cx-7, max(16, cy-8)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
        track['index'] = index
    for track in rejected:
        cx = int(track['points'][height//2][0])
        cv2.drawMarker(overlay, (cx, height//2), (0, 80, 255), cv2.MARKER_TILTED_CROSS, 7, 1)
    for index, track in enumerate(uncertain, 1):
        points = np.rint(track['points']).astype(np.int32)
        cv2.polylines(overlay, [points], False, (0, 140, 255), 1)
        cx, cy = map(int, points[height//2])
        cv2.putText(overlay, f'?{index}', (cx, max(16, cy-20)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 140, 255), 1)
    result = {'count': len(tracks), 'uncertain_count': len(uncertain), 'uncertain': uncertain,
              'requires_review': bool(uncertain) or (not fitted and bool(tracks)),
              'tracks': tracks, 'rejected': rejected,
              'fan': {'a': a, 'b': b, 'curvature': curve, 'fitted': fitted},
              'profile': profile.tolist(), 'ground_truth_verified': False}
    return result, {'permissive_mask': weak, 'bright_seeds': strong, 'aligned_mask': aligned_weak,
                    'kept_mask': kept, 'overlay': overlay}
