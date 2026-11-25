"""Data loading, target construction, auditing, and the train/validation/test split."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from screening import config


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, used to record exactly which data produced a result."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the primary BRFSS 2015 CSV without modifying it."""
    path = Path(path or config.DATA_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Primary dataset not found at {path}. Place "
            f"{config.PRIMARY_FILENAME} in data/raw/ or set DIABETES_DATA_PATH."
        )
    df = pd.read_csv(path)
    expected = [config.RAW_TARGET, *config.SET_A]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the study target: 0 = class 0, 1 = classes 1 or 2."""
    out = df.copy()
    out[config.TARGET] = out[config.RAW_TARGET].isin(config.POSITIVE_RAW_CLASSES).astype(int)
    return out


def load_study_data(path: Path | None = None) -> pd.DataFrame:
    """Raw data plus the binary study target. No rows are dropped or altered."""
    return add_binary_target(load_raw(path))


def audit(df: pd.DataFrame) -> dict:
    """Descriptive data checks. Nothing here changes the data."""
    raw_counts = df[config.RAW_TARGET].value_counts().sort_index()
    y = df[config.TARGET]
    bmi = df["BMI"]
    return {
        "n_rows": int(len(df)),
        "n_predictors": len(config.SET_A),
        "missing_cells": int(df[[config.RAW_TARGET, *config.SET_A]].isna().sum().sum()),
        "raw_class_counts": {str(int(k)): int(v) for k, v in raw_counts.items()},
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "positive_rate": float(y.mean()),
        # Repeated answer patterns: counted, NOT removed (no respondent ID exists).
        "repeated_full_rows": int(df[[config.RAW_TARGET, *config.SET_A]].duplicated().sum()),
        "repeated_predictor_patterns": int(df[config.SET_A].duplicated().sum()),
        "bmi": {
            "min": float(bmi.min()),
            "p01": float(bmi.quantile(0.01)),
            "median": float(bmi.median()),
            "p99": float(bmi.quantile(0.99)),
            "max": float(bmi.max()),
            "n_above_60": int((bmi > 60).sum()),
            "n_above_p99": int((bmi > bmi.quantile(0.99)).sum()),
        },
    }


@dataclass(frozen=True)
class Split:
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray


def make_split(y: pd.Series, seed: int = config.RANDOM_SEED) -> Split:
    """Stratified 70/15/15 split on row positions.

    First 70% train vs 30% holdout, then the holdout is halved into validation
    and test, both stratified on the binary target.
    """
    idx = np.arange(len(y))
    holdout_frac = config.VALID_FRACTION + config.TEST_FRACTION
    train_idx, hold_idx = train_test_split(
        idx, test_size=holdout_frac, stratify=y.to_numpy(), random_state=seed
    )
    test_share = config.TEST_FRACTION / holdout_frac
    valid_idx, test_idx = train_test_split(
        hold_idx, test_size=test_share, stratify=y.to_numpy()[hold_idx], random_state=seed
    )
    return Split(np.sort(train_idx), np.sort(valid_idx), np.sort(test_idx))
