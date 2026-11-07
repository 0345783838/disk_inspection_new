"""Conservative 25-slot completion; never infer disks or missing image pixels.

Require two complete, well-spaced reference rows. At most five detections may
be missing in one row. Ambiguous column assignment, poor fit or an out-of-frame
inferred box causes rejection instead of blindly filling to 25.
"""
import numpy as np


def centers(row):
    return (row[:, :2] + row[:, 2:]) / 2


def validate_complete(row, count):
    if len(row) != count:
        raise ValueError('Reference row is not complete')
    step = np.diff(centers(row)[:, 0])
    pitch = float(np.median(step))
    if pitch <= 0 or np.any(step < .55*pitch) or np.any(step > 1.55*pitch):
        raise ValueError('Reference row has duplicate/irregular slots')
    return pitch


def fit_projective_x(indexes, xs, count, pitch):
    t = (indexes-(count-1)/2)/((count-1)/2)
    grid = (np.arange(count)-(count-1)/2)/((count-1)/2)
    origin, scale = float(np.mean(xs)), float(np.ptp(xs))
    z = (xs-origin)/scale
    design = np.column_stack((t, np.ones(len(t)), -t*z))
    rng = np.random.default_rng(17)
    best = None
    # Deterministic RANSAC for a 1-D projective model, not collinear 2-D homography.
    for subset in [np.arange(len(t))] + [rng.choice(len(t),3,replace=False) for _ in range(100)]:
        if np.linalg.matrix_rank(design[subset]) < 3:
            continue
        a,b,c = np.linalg.lstsq(design[subset],z[subset],rcond=None)[0]
        denominator = 1+c*t
        if np.min(np.abs(denominator)) < .3:
            continue
        residual = np.abs((a*t+b)/denominator*scale+origin-xs)
        inliers = residual <= .12*pitch
        score = (int(inliers.sum()), -float(np.median(residual)))
        if best is None or score > best[0]:
            best = score,inliers
    if best is None or best[1].sum() < max(18,len(xs)-2):
        raise ValueError('Insufficient reliable slot anchors')
    inliers = best[1]
    a,b,c = np.linalg.lstsq(design[inliers],z[inliers],rcond=None)[0]
    if np.min(np.abs(1+c*grid)) < .3:
        raise ValueError('Unstable projective extrapolation')
    predicted = (a*grid+b)/(1+c*grid)*scale+origin
    step = np.diff(predicted)
    if np.any(step < .55*pitch) or np.any(step > 1.55*pitch):
        raise ValueError('Implausible fitted slot pitch')
    return predicted,inliers


def filter_off_grid_extras(rows, count=25):
    """Remove only unambiguous off-grid extras using two complete reference rows.

    A detection close to an occupied slot is ambiguous and is never silently
    discarded. Short rows remain the responsibility of complete_slot_rows.
    """
    rows = [np.asarray(row, dtype=np.float32).copy() for row in rows]
    if len(rows) != 3 or count != 25:
        return tuple(rows), []
    overfull = [i for i, row in enumerate(rows) if len(row) > count]
    if not overfull:
        return tuple(rows), []
    references = [i for i, row in enumerate(rows) if len(row) == count]
    if len(overfull) != 1 or len(references) != 2:
        raise ValueError('Extra points require two complete reference rows')
    rows = [row[np.argsort(centers(row)[:, 0])] for row in rows]
    for i in references:
        validate_complete(rows[i], count)
    target = overfull[0]
    row = rows[target]
    if len(row) > count + 5:
        raise ValueError('Too many extra point detections')
    ref_a, ref_b = [centers(rows[i]) for i in references]
    actual = centers(row)
    ya, yb, y = (float(np.median(points[:, 1])) for points in (ref_a, ref_b, actual))
    if yb <= ya:
        raise ValueError('Invalid reference row order')
    expected_x = ref_a[:, 0] + (ref_b[:, 0] - ref_a[:, 0]) * ((y - ya) / (yb - ya))
    pitch = float(np.median(np.diff(expected_x)))
    if pitch <= 0:
        raise ValueError('Invalid reference pitch')
    indexes = np.argmin(np.abs(actual[:, 0, None] - expected_x[None, :]), axis=1)
    supported = np.abs(actual[:, 0] - expected_x[indexes]) <= .30 * pitch
    if supported.sum() != count or len(np.unique(indexes[supported])) != count:
        raise ValueError('Ambiguous extra point detections')
    selected = row[supported]
    validate_complete(selected, count)
    _, inliers = fit_projective_x(indexes[supported], actual[supported, 0], count, pitch)
    if not inliers.all():
        raise ValueError('Unreliable point grid after removing extras')
    rows[target] = selected
    removed = [{'row': target + 1, 'box': box.tolist(), 'reason': 'outside_reference_slot_grid'}
               for box in row[~supported]]
    return tuple(rows), removed


def complete_slot_rows(rows, image_shape, count=25):
    if count != 25 or len(rows) != 3:
        raise ValueError('Only the calibrated three-row / 25-slot tray is supported')
    rows = [np.asarray(r,dtype=np.float32).copy() for r in rows]
    if any(r.ndim != 2 or r.shape[1] != 4 or not np.isfinite(r).all() or
           np.any(r[:,2:] <= r[:,:2]) for r in rows):
        raise ValueError('Invalid slot boxes')
    rows = [r[np.argsort(centers(r)[:,0])] for r in rows]
    complete = [i for i,r in enumerate(rows) if len(r)==count]
    for i in complete:
        validate_complete(rows[i],count)
    if len(complete)==3:
        return tuple(rows),[]
    if len(complete)!=2:
        raise ValueError('Need two complete reference rows')
    missing_row = next(i for i in range(3) if i not in complete)
    row = rows[missing_row]
    if not count-5 <= len(row) < count:
        raise ValueError('Too many missing slots or extra detections')
    ref_a,ref_b = [centers(rows[i]) for i in complete]
    actual = centers(row)
    ya,yb,y = float(np.median(ref_a[:,1])),float(np.median(ref_b[:,1])),float(np.median(actual[:,1]))
    if yb <= ya:
        raise ValueError('Invalid reference row order')
    expected_x = ref_a[:,0]+(ref_b[:,0]-ref_a[:,0])*((y-ya)/(yb-ya))
    pitch = float(np.median(np.diff(expected_x)))
    if pitch <= 0:
        raise ValueError('Invalid reference pitch')
    indexes = np.argmin(np.abs(actual[:,0,None]-expected_x[None,:]),axis=1)
    if len(np.unique(indexes))!=len(indexes) or np.any(np.abs(actual[:,0]-expected_x[indexes])>.30*pitch):
        raise ValueError('Ambiguous slot indexes / reference alignment')
    if indexes[-1]-indexes[0] < count-7:
        raise ValueError('Slot anchors do not span enough of the row')
    xs,inliers = fit_projective_x(indexes,actual[:,0],count,pitch)
    good_indexes = indexes[inliers]
    t = (good_indexes-12)/12
    y_fit = np.polyfit(t,actual[inliers,1],2)
    y_error = np.abs(np.polyval(y_fit,t)-actual[inliers,1])
    if np.max(y_error)>max(3.0,.12*float(np.median(row[:,3]-row[:,1]))):
        raise ValueError('Row shape is not reliably fitted')
    ys = np.polyval(y_fit,(np.arange(count)-12)/12)
    recovered,metadata = [],[]
    for slot in range(count):
        found = np.flatnonzero((indexes==slot)&inliers)
        if len(found):
            recovered.append(row[found[0]])
            continue
        near = np.argsort(np.abs(good_indexes-slot))[:4]
        sizes = row[inliers,2:]-row[inliers,:2]
        width,height = np.median(sizes[near],axis=0)
        box = np.array([xs[slot]-width/2,ys[slot]-height/2,
                        xs[slot]+width/2,ys[slot]+height/2],np.float32)
        if box[0]<0 or box[1]<0 or box[2]>image_shape[1] or box[3]>image_shape[0]:
            raise ValueError('Inferred slot is outside the available image')
        recovered.append(box)
        metadata.append({'row':missing_row+1,'slot':slot+1,'box':box.tolist(),
                         'source':'geometry_inferred','detector_confidence':None})
    if len(metadata)>5:
        raise ValueError('Recovery exceeds five-slot limit')
    rows[missing_row] = np.asarray(recovered,dtype=np.float32)
    return tuple(rows),metadata
