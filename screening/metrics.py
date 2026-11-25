"""Screening metrics, validation threshold selection, and bootstrap intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def threshold_for_recall(y_true: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    """Highest score threshold whose recall (score >= threshold) is at least the target.

    Call this on VALIDATION data only. The test set is never used here.
    """
    _, recall, thresholds = precision_recall_curve(y_true, scores)
    # recall[i] is the recall when predicting positive for score >= thresholds[i]
    ok = recall[:-1] >= target_recall
    if not ok.any():
        return float(np.min(scores))
    return float(np.max(thresholds[ok]))


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Confusion counts and rates at a fixed threshold (positive if score >= threshold)."""
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    def ratio(a: int, b: int) -> float:
        return a / b if b else float("nan")

    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    precision = ratio(tp, tp + fp)
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "npv": ratio(tn, tn + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (recall + specificity) / 2,
        "accuracy": ratio(tp + tn, tp + tn + fp + fn),
        "flagged_share": ratio(tp + fp, tp + tn + fp + fn),
    }


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Threshold-free metrics.

    * roc_auc: area under the ROC curve.
    * average_precision: sklearn's step-wise average precision (NOT called PR-AUC).
    * pr_auc_trapezoid: trapezoidal area under the precision-recall curve, reported
      separately and labelled as such because it is a different calculation.
    """
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "pr_auc_trapezoid": float(auc(recall, precision)),
        "prevalence": float(np.mean(y_true)),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    rounds: int,
    seed: int,
) -> dict:
    """Percentile 95% intervals from resampling the TEST rows (model fixed).

    This reflects sampling variation of the test set only, not variation from
    retraining or re-splitting.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = ("roc_auc", "average_precision", "recall", "specificity", "precision")
    vals: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(rounds):
        i = rng.integers(0, n, n)
        yt, s = y_true[i], scores[i]
        if yt.min() == yt.max():
            continue
        vals["roc_auc"].append(roc_auc_score(yt, s))
        vals["average_precision"].append(average_precision_score(yt, s))
        m = threshold_metrics(yt, s, threshold)
        for k in ("recall", "specificity", "precision"):
            vals[k].append(m[k])
    out = {}
    for k, v in vals.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[f"{k}_ci_low"] = float(lo)
        out[f"{k}_ci_high"] = float(hi)
    return out


def curve_points(y_true: np.ndarray, scores: np.ndarray, max_points: int = 400) -> dict:
    """Down-sampled ROC and precision-recall curve points for plotting."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)

    def thin(*arrays: np.ndarray) -> list[np.ndarray]:
        n = len(arrays[0])
        if n <= max_points:
            return list(arrays)
        keep = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
        return [a[keep] for a in arrays]

    fpr, tpr = thin(fpr, tpr)
    prec, rec = thin(prec, rec)
    return {"roc": (fpr, tpr), "pr": (rec, prec)}
