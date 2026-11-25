"""Cached loaders and small display helpers shared by the Streamlit pages.

Nothing here trains models: the dashboard only reads what
scripts/run_analysis.py wrote, so the app and the reported results always
share the same target, feature sets, split, and validation thresholds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import pandas as pd
import streamlit as st

from screening import config
from screening.data import audit, file_sha256, load_study_data
from screening.tool import load_settings

SET_COLORS = {"A": "#0072B2", "B": "#E69F00", "C": "#009E73"}  # Okabe–Ito
CLASS_COLORS = {"Negative": "#56B4E9", "Positive": "#D55E00"}

DISCLAIMER = (
    "This is a research prototype for studying questionnaire-based screening. "
    "It is not a diagnostic tool and does not give medical advice. Decisions about "
    "health require a qualified clinician and appropriate testing."
)

ANALYSIS_COMMAND = "py -3.13 scripts/run_analysis.py"


@st.cache_data(show_spinner="Loading the BRFSS 2015 data…")
def load_data() -> pd.DataFrame:
    return load_study_data(config.DATA_PATH)


@st.cache_data(show_spinner=False)
def data_audit() -> dict:
    return audit(load_data())


@st.cache_data(show_spinner=False)
def data_hash() -> str:
    return file_sha256(config.DATA_PATH)


@dataclass
class Results:
    models: pd.DataFrame
    roc: pd.DataFrame
    pr: pd.DataFrame
    sweep: pd.DataFrame
    ranking: pd.DataFrame
    score_hist: pd.DataFrame
    stability: pd.DataFrame | None
    prelim: pd.DataFrame
    run_meta: dict


def _results_stamp() -> float | None:
    """Modification time of the run metadata (written last), or None if outputs are missing."""
    o, c = config.OUTPUTS_DIR, config.CURVES_DIR
    needed = [o / "model_results.csv", o / "run_metadata.json", c / "roc_test.csv",
              c / "pr_test.csv", o / "set_c_validation_sweep.csv"]
    if not all(p.exists() for p in needed):
        return None
    return (o / "run_metadata.json").stat().st_mtime


def load_results() -> Results | None:
    """Saved analysis outputs; re-read automatically after the analysis is re-run."""
    stamp = _results_stamp()
    return None if stamp is None else _load_results(stamp)


@st.cache_data(show_spinner=False, max_entries=2)
def _load_results(stamp: float) -> Results:
    o, c = config.OUTPUTS_DIR, config.CURVES_DIR
    stab_path = o / "stability_hgb.csv"
    return Results(
        models=pd.read_csv(o / "model_results.csv"),
        roc=pd.read_csv(c / "roc_test.csv"),
        pr=pd.read_csv(c / "pr_test.csv"),
        sweep=pd.read_csv(o / "set_c_validation_sweep.csv"),
        ranking=pd.read_csv(o / "set_c_forward_ranking.csv"),
        score_hist=pd.read_csv(c / "set_c_test_score_hist.csv"),
        stability=pd.read_csv(stab_path) if stab_path.exists() else None,
        prelim=pd.read_csv(o / "preliminary_vs_recomputed.csv"),
        run_meta=json.loads((o / "run_metadata.json").read_text()),
    )


def load_set_c_model():
    """The saved Set C model and its metadata, or (None, None) if not trained yet."""
    if not (config.SET_C_MODEL_PATH.exists() and config.SET_C_META_PATH.exists()):
        return None, None
    return _load_set_c_model(config.SET_C_MODEL_PATH.stat().st_mtime)


@st.cache_resource(show_spinner="Loading the Set C model…", max_entries=2)
def _load_set_c_model(stamp: float):
    model = joblib.load(config.SET_C_MODEL_PATH)
    meta = json.loads(config.SET_C_META_PATH.read_text())
    return model, meta


# ---------------------------------------------------------------------------
# Screening tool
# ---------------------------------------------------------------------------
@dataclass
class ToolAssets:
    models: dict
    meta: dict
    grid: pd.DataFrame


def load_tool_assets() -> ToolAssets | None:
    paths = (config.TOOL_MODELS_PATH, config.TOOL_META_PATH, config.THRESHOLD_GRID_PATH)
    if not all(p.exists() for p in paths):
        return None
    return _load_tool_assets(config.TOOL_MODELS_PATH.stat().st_mtime)


@st.cache_resource(show_spinner="Loading screening models…", max_entries=2)
def _load_tool_assets(stamp: float) -> ToolAssets:
    return ToolAssets(
        models=joblib.load(config.TOOL_MODELS_PATH),
        meta=json.loads(config.TOOL_META_PATH.read_text()),
        grid=pd.read_csv(config.THRESHOLD_GRID_PATH),
    )


def require_tool_assets() -> ToolAssets:
    assets = load_tool_assets()
    if assets is None:
        st.warning(
            "The screening-tool models were not found. From the project folder, run:\n\n"
            f"```\n{ANALYSIS_COMMAND}\n```\n\nthen reload this page.",
            title="Run the analysis first",
        )
        st.stop()
    return assets


def tool_settings() -> dict:
    """Current settings for this browser session (loaded from the saved file once)."""
    if "tool_settings" not in st.session_state:
        st.session_state.tool_settings = load_settings()
    return st.session_state.tool_settings


# ---------------------------------------------------------------------------
def require_data() -> None:
    if not config.DATA_PATH.exists():
        st.error(
            f"The primary dataset was not found at `{config.DATA_PATH}`. Place "
            f"`{config.PRIMARY_FILENAME}` in `data/raw/` (or set the "
            "`DIABETES_DATA_PATH` environment variable) and reload.",
            title="Dataset missing",
        )
        st.stop()


def require_results() -> Results:
    res = load_results()
    if res is None:
        st.warning(
            "No analysis outputs were found. From the project folder, run:\n\n"
            f"```\n{ANALYSIS_COMMAND}\n```\n\nthen reload this page.",
            title="Run the analysis first",
        )
        st.stop()
    return res


def check_data_matches(res: Results) -> None:
    """Warn if the data on disk is not the data that produced the saved results."""
    if res.run_meta.get("data_sha256") != data_hash():
        st.warning(
            "The dataset on disk differs from the one used to produce the saved "
            f"results. Re-run `{ANALYSIS_COMMAND}` so the dashboard and results match.",
            title="Results may be out of date",
        )


def fmt_ci(value: float, lo: float | None, hi: float | None, digits: int = 3) -> str:
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ({lo:.{digits}f}–{hi:.{digits}f})"
