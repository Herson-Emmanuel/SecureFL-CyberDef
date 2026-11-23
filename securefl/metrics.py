"""Detection metrics and validation-based threshold calibration."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)


def metrics_at(y, p, thr):
    y = np.asarray(y).astype(int)
    yhat = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = dict(threshold=float(thr), accuracy=(tp + tn) / len(y), precision=prec, recall=rec,
               f1=f1, fpr=fp / (fp + tn) if fp + tn else 0.0, fnr=fn / (fn + tp) if fn + tp else 0.0,
               tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))
    try:
        out["auc_roc"] = float(roc_auc_score(y, p))
        out["avg_precision"] = float(average_precision_score(y, p))
    except ValueError:
        out["auc_roc"] = out["avg_precision"] = float("nan")
    return out


def best_f1_threshold(y_val, p_val):
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.nanargmax(f1))])


def fpr_target_threshold(y_val, p_val, target=1e-3):
    """Threshold that maximises validation recall subject to FPR <= target."""
    fpr, tpr, thr = roc_curve(y_val, p_val)
    ok = np.where(fpr <= target)[0]
    if len(ok) == 0:
        return float(np.max(p_val) + 1e-12)
    i = ok[np.argmax(tpr[ok])]
    return float(min(thr[i], 1.0))


def evaluate(y_val, p_val, y_test, p_test, fpr_target=1e-3):
    t_f1 = best_f1_threshold(y_val, p_val)
    t_fpr = fpr_target_threshold(y_val, p_val, fpr_target)
    return {"default": metrics_at(y_test, p_test, 0.5),
            "best_f1": metrics_at(y_test, p_test, t_f1),
            "fpr_target": metrics_at(y_test, p_test, t_fpr)}
