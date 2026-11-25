"""Model definitions, training, and the training-data-only Set C selection."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from screening import config
from screening.metrics import ranking_metrics, threshold_for_recall, threshold_metrics

# Class weighting handles imbalance. No synthetic resampling is used anywhere.


def make_model(name: str, seed: int = config.RANDOM_SEED):
    if name == "logreg":
        # Scaler is fitted inside the pipeline on training data only.
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000),
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=300,
            learning_rate=0.1,
            early_stopping=True,      # internal 10% of TRAINING data
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


def scores(model, X: pd.DataFrame) -> np.ndarray:
    """Positive-class model scores. These are NOT calibrated probabilities."""
    return model.predict_proba(X)[:, 1]


def fit_evaluate(
    name: str,
    features: list[str],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame | None = None,
    y_test: np.ndarray | None = None,
    seed: int = config.RANDOM_SEED,
) -> dict:
    """Fit on train, pick a threshold on validation, then (optionally) score test once."""
    t0 = time.perf_counter()
    model = make_model(name, seed)
    model.fit(X_train[features], y_train)
    s_val = scores(model, X_valid[features])
    thr = threshold_for_recall(y_valid, s_val, config.TARGET_RECALL)
    out = {
        "model": model,
        "threshold": thr,
        "valid": {**ranking_metrics(y_valid, s_val), **threshold_metrics(y_valid, s_val, thr)},
        "valid_scores": s_val,
        "fit_seconds": time.perf_counter() - t0,
    }
    if X_test is not None and y_test is not None:
        s_test = scores(model, X_test[features])
        out["test"] = {**ranking_metrics(y_test, s_test), **threshold_metrics(y_test, s_test, thr)}
        out["test_scores"] = s_test
    return out


def forward_rank_features(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    candidates: list[str],
    seed: int = config.RANDOM_SEED,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Greedy forward ranking of candidate features using TRAINING data only.

    The training partition is split once more (stratified 80/20). At each step
    the feature that most increases inner-holdout ROC-AUC of a gradient-boosting
    model is added. Validation and test data are never touched.
    """
    tr, ho = train_test_split(
        np.arange(len(y_train)),
        test_size=config.INNER_HOLDOUT_FRACTION,
        stratify=y_train,
        random_state=seed,
    )
    Xa, ya, Xh, yh = X_train.iloc[tr], y_train[tr], X_train.iloc[ho], y_train[ho]
    chosen: list[str] = []
    remaining = list(candidates)
    rows = []
    while remaining:
        best_f, best_auc = None, -1.0
        for f in remaining:
            m = make_model("hgb", seed)
            m.fit(Xa[chosen + [f]], ya)
            a = ranking_metrics(yh, scores(m, Xh[chosen + [f]]))["roc_auc"]
            if a > best_auc:
                best_f, best_auc = f, a
        chosen.append(best_f)
        remaining.remove(best_f)
        rows.append({"step": len(chosen), "added_feature": best_f, "inner_holdout_roc_auc": best_auc})
        log(f"  forward step {len(chosen):2d}: +{best_f:<20s} inner AUC={best_auc:.4f}")
    return pd.DataFrame(rows)


def set_c_sweep(
    ranking: list[str],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, int]:
    """Validation sweep over the top-k ranked features (k = 1..len(ranking)).

    Returns the sweep table and the selected k: the smallest k whose validation
    ROC-AUC >= SET_C_MIN_VAL_AUC and validation recall >= SET_C_MIN_VAL_RECALL.
    """
    rows = []
    for k in range(1, len(ranking) + 1):
        feats = ranking[:k]
        r = fit_evaluate("hgb", feats, X_train, y_train, X_valid, y_valid, seed=seed)
        v = r["valid"]
        rows.append({
            "k": k,
            "features": ", ".join(feats),
            "valid_roc_auc": v["roc_auc"],
            "valid_average_precision": v["average_precision"],
            "valid_recall": v["recall"],
            "valid_specificity": v["specificity"],
            "valid_precision": v["precision"],
            "threshold": r["threshold"],
            "meets_rule": bool(v["roc_auc"] >= config.SET_C_MIN_VAL_AUC
                               and v["recall"] >= config.SET_C_MIN_VAL_RECALL),
        })
    sweep = pd.DataFrame(rows)
    meeting = sweep[sweep["meets_rule"]]
    selected_k = int(meeting["k"].min()) if len(meeting) else len(ranking)
    sweep["selected"] = sweep["k"] == selected_k
    return sweep, selected_k
