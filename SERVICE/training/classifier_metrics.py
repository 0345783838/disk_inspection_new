"""Three-class metrics with true rows / predicted columns, including false OK."""
import numpy as np

LABELS = ['ng', 'no_disk', 'ok']


def classification_metrics(targets, predictions):
    targets, predictions = np.asarray(targets, dtype=int), np.asarray(predictions, dtype=int)
    if targets.shape != predictions.shape or targets.ndim != 1 or not len(targets):
        raise ValueError('Expected equally sized nonempty vectors')
    if not (np.isin(targets, range(3)).all() and np.isin(predictions, range(3)).all()):
        raise ValueError('Unknown class index')
    confusion = np.zeros((3,3), dtype=int)
    np.add.at(confusion, (targets, predictions), 1)
    per_class = {}
    for i, label in enumerate(LABELS):
        tp, actual, predicted = int(confusion[i,i]), int(confusion[i].sum()), int(confusion[:,i].sum())
        precision, recall = tp/max(predicted,1), tp/max(actual,1)
        per_class[label] = {'precision':precision, 'recall':recall,
                            'f1':2*precision*recall/max(precision+recall,1e-12), 'support':actual}
    unsafe_total = int(confusion[:2].sum())
    false_ok = int(confusion[:2,2].sum())
    return {'count':len(targets), 'accuracy':float(np.trace(confusion)/len(targets)),
            'macro_f1':float(np.mean([v['f1'] for v in per_class.values()])),
            'confusion_matrix_true_rows':confusion.tolist(), 'class_order':LABELS,
            'per_class':per_class, 'unsafe_false_ok':false_ok,
            'unsafe_false_ok_rate':false_ok/max(unsafe_total,1),
            'ng_as_ok':int(confusion[0,2]), 'ng_as_no_disk':int(confusion[0,1]),
            'ng_not_detected':int(confusion[0,1]+confusion[0,2]),
            'no_disk_as_ok':int(confusion[1,2])}


def safety_fitness(metrics):
    """Exact lexicographic ranking: NG->OK, NG->empty, empty->OK, macro-F1.

    An OK-recall guard prevents a trivial reject-all classifier from winning.
    Supports define integer radix factors so a preceding error always dominates.
    """
    ng_count, empty_count = metrics['per_class']['ng']['support'], metrics['per_class']['no_disk']['support']
    secondary_radix, tertiary_radix = ng_count+1, empty_count+1
    rank = ((ng_count-metrics['ng_as_ok'])*secondary_radix*tertiary_radix +
            (ng_count-metrics['ng_as_no_disk'])*tertiary_radix +
            empty_count-metrics['no_disk_as_ok'] + .001*metrics['macro_f1'])
    maximum = ng_count*secondary_radix*tertiary_radix + ng_count*tertiary_radix + empty_count + 1
    return (2 if metrics['per_class']['ok']['recall']>=.9 else 0)+rank/maximum


def dual_safety_failures(ha, legacy, gates=None):
    gates = gates or {}
    failures = []
    if ha['per_class']['ok']['recall'] < gates.get('ha_ok_recall_min', .90):
        failures.append('HA OK recall below retention gate')
    if legacy['per_class']['ok']['recall'] < gates.get('legacy_ok_recall_min', .95):
        failures.append('Legacy OK recall below retention gate')
    if legacy['accuracy'] < gates.get('legacy_accuracy_min', .97):
        failures.append('Legacy accuracy below retention gate')
    if legacy['ng_as_ok']+legacy['ng_as_no_disk'] > gates.get('legacy_ng_misses_max', 21):
        failures.append('Legacy NG->OK/no_disk above retention gate')
    if legacy['no_disk_as_ok'] > gates.get('legacy_empty_as_ok_max', 2):
        failures.append('Legacy empty->OK above retention gate')
    return failures


def dual_safety_fitness(ha, legacy, gates=None):
    ha_ng_miss=ha['ng_as_ok']+ha['ng_as_no_disk']
    legacy_ng_miss=legacy['ng_as_ok']+legacy['ng_as_no_disk']
    eligible = not dual_safety_failures(ha, legacy, gates)
    # Production WARNING sends OK to PLC: NG->no_disk is also a defect miss.
    ng_risk = .75*ha_ng_miss/ha['per_class']['ng']['support']+.25*legacy_ng_miss/legacy['per_class']['ng']['support']
    pass_ng = .75*ha['ng_as_ok']/ha['per_class']['ng']['support']+.25*legacy['ng_as_ok']/legacy['per_class']['ng']['support']
    empty_risk = .75*ha['no_disk_as_ok']/ha['per_class']['no_disk']['support']+.25*legacy['no_disk_as_ok']/legacy['per_class']['no_disk']['support']
    return ((2 if eligible else 0) + 1-ng_risk + 1e-6*(1-pass_ng) +
            1e-8*(1-empty_risk) + 1e-10*(.8*ha['macro_f1']+.2*legacy['macro_f1']))
