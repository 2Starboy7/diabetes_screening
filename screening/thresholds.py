"""Threshold grid for the screening tool.

A threshold is chosen by saying how many positive people the tool should catch
(the recall target). For every feature set and age group, the matching score
cut-off is found on VALIDATION data only. Test-set results at the same cut-off
are stored next to it so users can see how it held up on unseen data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from screening import config
from screening.metrics import threshold_for_recall, threshold_metrics

BALANCED = "balanced"


def youden_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Cut-off that maximises recall + specificity − 1 (Youden's J)."""
    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    return float(thr[1:][np.argmax(j[1:])])


def age_band_masks(age: np.ndarray) -> dict[str, np.ndarray]:
    masks = {config.ALL_AGES: np.ones(len(age), dtype=bool)}
    for name, (lo, hi) in config.AGE_BANDS.items():
        masks[name] = (age >= lo) & (age <= hi)
    return masks


def build_grid(
    feature_set: str,
    y_valid: np.ndarray, s_valid: np.ndarray, age_valid: np.ndarray,
    y_test: np.ndarray, s_test: np.ndarray, age_test: np.ndarray,
) -> pd.DataFrame:
    rows = []
    vmasks, tmasks = age_band_masks(age_valid), age_band_masks(age_test)
    for band, vm in vmasks.items():
        yv, sv = y_valid[vm], s_valid[vm]
        yt, st_ = y_test[tmasks[band]], s_test[tmasks[band]]
        rules: list[tuple[str, float, float]] = [
            (f"recall_{t:.2f}", t, threshold_for_recall(yv, sv, t)) for t in config.RECALL_TARGET_GRID
        ]
        rules.append((BALANCED, float("nan"), youden_threshold(yv, sv)))
        for rule, target, thr in rules:
            v = threshold_metrics(yv, sv, thr)
            t = threshold_metrics(yt, st_, thr)
            rows.append({
                "feature_set": feature_set, "age_band": band, "rule": rule,
                "recall_target": target, "threshold": thr,
                "valid_n": int(len(yv)), "valid_positives": int(yv.sum()),
                **{f"valid_{k}": v[k] for k in ("recall", "specificity", "precision", "npv", "flagged_share")},
                "test_n": int(len(yt)), "test_positives": int(yt.sum()),
                **{f"test_{k}": t[k] for k in ("recall", "specificity", "precision", "npv", "flagged_share")},
            })
    return pd.DataFrame(rows)


def lookup(grid: pd.DataFrame, feature_set: str, age_band: str, rule: str) -> pd.Series:
    """One row of the grid (threshold plus its validation and test metrics)."""
    row = grid[(grid["feature_set"] == feature_set) & (grid["age_band"] == age_band) & (grid["rule"] == rule)]
    if row.empty:
        raise KeyError(f"No threshold for set={feature_set}, band={age_band}, rule={rule}")
    return row.iloc[0]


def age_band_of(age_code: int) -> str:
    for name, (lo, hi) in config.AGE_BANDS.items():
        if lo <= age_code <= hi:
            return name
    raise ValueError(f"Age code out of range: {age_code}")
