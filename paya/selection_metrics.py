"""Vectorized development metrics with canonical answer alignment and AP."""

from collections import defaultdict

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def calibration_error(confidence, correct, bins=15, equal_mass=False):
    confidence, correct = np.asarray(confidence), np.asarray(correct)
    if equal_mass:
        groups = np.array_split(np.argsort(confidence, kind='stable'), min(bins, len(confidence)))
    else:
        assignments = np.minimum((confidence*bins).astype(int), bins-1)
        groups = [np.flatnonzero(assignments == i) for i in range(bins)]
    reliability = [dict(bin=i, count=len(g), accuracy=float(correct[g].mean()),
                        confidence=float(confidence[g].mean())) for i, g in enumerate(groups) if len(g)]
    return float(sum(v['count']*abs(v['accuracy']-v['confidence']) for v in reliability)/len(confidence)), reliability


def summarize_arrays(logits, targets):
    logits = np.asarray(logits, dtype=np.float64)
    logp = logits-logsumexp(logits, axis=1, keepdims=True)
    p = np.exp(logp)
    # Columns have canonical alphabetic answer order, matching the public tie rule.
    predicted = p.argmax(axis=1); hit = predicted == targets
    confidence = p[np.arange(len(p)), predicted]
    ece, reliability = calibration_error(confidence, hit)
    brier = (p*p).sum(axis=1)-2*p[np.arange(len(p)), targets]+1
    return dict(count=len(p), accuracy=float(hit.mean()),
        nll=float(-logp[np.arange(len(p)), targets].mean()), brier=float(brier.mean()),
        ece_15=ece, reliability=reliability,
        ece_equal_mass=calibration_error(confidence, hit, equal_mass=True)[0]), p, predicted


def score_logits(rows, logits):
    """Score one consistent vocabulary per task; retain unsupported class counts."""
    if len(rows) != len(logits) or not len(rows):
        raise ValueError('Need matching nonempty rows and logits')
    grouped = defaultdict(list)
    for i, row in enumerate(rows): grouped[row['task']].append(i)
    per_task = {}; correct = []; confidence = []; nll = []; brier = []
    for task, positions in sorted(grouped.items()):
        selected = [rows[i] for i in positions]
        labels = sorted(selected[0]['criteria']); lookup = {a:i for i,a in enumerate(labels)}
        if any(sorted(r['criteria']) != labels for r in selected):
            raise ValueError('Each task requires a consistent answer vocabulary')
        order = np.array([[r['criteria'].index(a) for a in labels] for r in selected])
        z = np.asarray(logits)[np.array(positions)[:,None], order]
        if not np.isfinite(z).all(): raise ValueError('Real candidate logits must be finite')
        y = np.array([lookup[r['criteria'][r['target']]] for r in selected])
        value, p, pred = summarize_arrays(z, y)
        matrix = confusion_matrix(y, pred, labels=np.arange(len(labels)))
        classes = {}
        for i, answer in enumerate(labels):
            support = int((y==i).sum()); predicted = int((pred==i).sum()); tp = int(matrix[i,i])
            supported = 0 < support < len(y)
            classes[answer] = dict(support=support, predicted=predicted, correct=tp,
                precision=tp/predicted if predicted else 0., recall=tp/support if support else None,
                f1=2*tp/(support+predicted) if support+predicted else 0.,
                auroc_ovr=float(roc_auc_score(y==i,p[:,i])) if supported else None,
                average_precision=float(average_precision_score(y==i,p[:,i])) if supported else None,
                calibration_error=calibration_error(p[:,i],y==i)[0])
        auc = [v['auroc_ovr'] for v in classes.values() if v['auroc_ovr'] is not None]
        ap = [v['average_precision'] for v in classes.values() if v['average_precision'] is not None]
        value.update(macro_f1=float(np.mean([v['f1'] for v in classes.values()])),
            macro_auroc_ovr=float(np.mean(auc)) if auc else None,
            macro_average_precision=float(np.mean(ap)) if ap else None,
            per_class=classes, confusion={a:{b:int(matrix[i,j]) for j,b in enumerate(labels)}
                                         for i,a in enumerate(labels)})
        per_task[task] = value
        correct.extend((pred==y).tolist()); confidence.extend(p[np.arange(len(p)),pred].tolist())
        nll.extend((logsumexp(z,axis=1)-z[np.arange(len(z)),y]).tolist())
        brier.extend(((p*p).sum(1)-2*p[np.arange(len(p)),y]+1).tolist())
    aucs = [v['macro_auroc_ovr'] for v in per_task.values() if v['macro_auroc_ovr'] is not None]
    pooled = dict(count=len(rows), accuracy=float(np.mean(correct)), nll=float(np.mean(nll)),
        brier=float(np.mean(brier)), ece_15=calibration_error(confidence,correct)[0],
        ece_equal_mass=calibration_error(confidence,correct,equal_mass=True)[0])
    return dict(pooled=pooled, per_task=per_task,
        task_macro_f1=float(np.mean([v['macro_f1'] for v in per_task.values()])),
        task_macro_auroc=float(np.mean(aucs)) if aucs else None,
        task_mean_nll=float(np.mean([v['nll'] for v in per_task.values()])))
