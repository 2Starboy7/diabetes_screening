"""Reproducible end-to-end analysis.

Run from the project folder:

    py -3.13 scripts/run_analysis.py

Writes everything the dashboard reads into outputs/ and models/.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from screening import config, figures  # noqa: E402
from screening.data import audit, file_sha256, load_study_data, make_split  # noqa: E402
from screening.metrics import bootstrap_ci, curve_points  # noqa: E402
from screening.modeling import fit_evaluate, forward_rank_features, set_c_sweep  # noqa: E402
from screening.preliminary import PRELIMINARY_RESULTS, PRELIMINARY_SET_C  # noqa: E402
from screening.thresholds import build_grid  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=config.DATA_PATH, help="Primary CSV path")
    ap.add_argument("--bootstrap", type=int, default=config.BOOTSTRAP_ROUNDS,
                    help="Bootstrap rounds for test-set intervals (0 to skip)")
    ap.add_argument("--skip-stability", action="store_true",
                    help="Skip the repeated-split stability check")
    args = ap.parse_args()

    for d in (config.OUTPUTS_DIR, config.FIGURES_DIR, config.CURVES_DIR, config.MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # 1. Load and audit (no rows dropped, no values clipped) -----------------
    log(f"Loading {args.data}")
    df = load_study_data(args.data)
    data_hash = file_sha256(Path(args.data))
    aud = audit(df)
    aud["sha256"] = data_hash
    (config.OUTPUTS_DIR / "data_audit.json").write_text(json.dumps(aud, indent=2))
    log(f"Rows={aud['n_rows']:,}  positives={aud['n_positive']:,} ({aud['positive_rate']:.4f})  "
        f"missing={aud['missing_cells']}  repeated full rows={aud['repeated_full_rows']:,}  "
        f"repeated predictor patterns={aud['repeated_predictor_patterns']:,}")

    # 2. Split -----------------------------------------------------------------
    y = df[config.TARGET]
    split = make_split(y, config.RANDOM_SEED)
    X = df[config.SET_A]
    yv = y.to_numpy()
    X_tr, X_va, X_te = X.iloc[split.train_idx], X.iloc[split.valid_idx], X.iloc[split.test_idx]
    y_tr, y_va, y_te = yv[split.train_idx], yv[split.valid_idx], yv[split.test_idx]
    split_info = {
        name: {"n": int(len(yy)), "positives": int(yy.sum()), "positive_rate": float(yy.mean())}
        for name, yy in (("train", y_tr), ("validation", y_va), ("test", y_te))
    }
    log(f"Split sizes: { {k: v['n'] for k, v in split_info.items()} }")

    # 3. Set C discovery (training data for ranking, validation for k) --------
    log("Ranking Set B features by greedy forward selection on TRAINING data only")
    ranking_df = forward_rank_features(X_tr, y_tr, config.SET_B, log=log)
    ranking_df.to_csv(config.OUTPUTS_DIR / "set_c_forward_ranking.csv", index=False)
    ranking = ranking_df["added_feature"].tolist()

    log("Validation sweep over top-k features (k = 1..16)")
    sweep, k_sel = set_c_sweep(ranking, X_tr, y_tr, X_va, y_va)
    sweep.to_csv(config.OUTPUTS_DIR / "set_c_validation_sweep.csv", index=False)
    set_c = ranking[:k_sel]
    rule_met = bool(sweep.loc[sweep["k"] == k_sel, "meets_rule"].iloc[0])
    log(f"Selected Set C (k={k_sel}, rule met={rule_met}): {set_c}")

    feature_sets = {"A": config.SET_A, "B": config.SET_B, "C": set_c}

    # 4. Main comparison: every model on every set ----------------------------
    rows, roc_rows, pr_rows, fitted = [], [], [], {}
    for fs, feats in feature_sets.items():
        for mdl in ("logreg", "rf", "hgb"):
            log(f"Fitting Set {fs} / {config.MODEL_LABELS[mdl]} ({len(feats)} features)")
            r = fit_evaluate(mdl, feats, X_tr, y_tr, X_va, y_va, X_te, y_te)
            fitted[(fs, mdl)] = r
            row = {
                "feature_set": fs, "model": mdl, "model_label": config.MODEL_LABELS[mdl],
                "n_features": len(feats), "features": ", ".join(feats),
                "threshold": r["threshold"], "fit_seconds": round(r["fit_seconds"], 1),
            }
            row.update({f"valid_{k}": v for k, v in r["valid"].items() if k != "threshold"})
            row.update({f"test_{k}": v for k, v in r["test"].items() if k != "threshold"})
            if args.bootstrap > 0:
                ci = bootstrap_ci(y_te, r["test_scores"], r["threshold"], args.bootstrap, config.RANDOM_SEED)
                row.update({f"test_{k}": v for k, v in ci.items()})
            rows.append(row)
            cp = curve_points(y_te, r["test_scores"])
            roc_rows.append(pd.DataFrame({"feature_set": fs, "model": mdl,
                                          "fpr": cp["roc"][0], "tpr": cp["roc"][1]}))
            pr_rows.append(pd.DataFrame({"feature_set": fs, "model": mdl,
                                         "recall": cp["pr"][0], "precision": cp["pr"][1]}))
            log(f"   test AUC={r['test']['roc_auc']:.4f} AP={r['test']['average_precision']:.4f} "
                f"recall={r['test']['recall']:.4f} spec={r['test']['specificity']:.4f} "
                f"prec={r['test']['precision']:.4f} thr={r['threshold']:.4f}")
    results = pd.DataFrame(rows)
    results.to_csv(config.OUTPUTS_DIR / "model_results.csv", index=False)
    roc = pd.concat(roc_rows, ignore_index=True)
    pr = pd.concat(pr_rows, ignore_index=True)
    roc.to_csv(config.CURVES_DIR / "roc_test.csv", index=False)
    pr.to_csv(config.CURVES_DIR / "pr_test.csv", index=False)

    # Score distribution of the Set C model on test, by class (for the demo page)
    sc = fitted[("C", "hgb")]
    bins = np.linspace(0, 1, 41)
    hist = []
    for cls in (0, 1):
        counts, _ = np.histogram(sc["test_scores"][y_te == cls], bins=bins)
        hist.append(pd.DataFrame({"bin_start": bins[:-1], "bin_end": bins[1:],
                                  "target": cls, "count": counts}))
    pd.concat(hist).to_csv(config.CURVES_DIR / "set_c_test_score_hist.csv", index=False)

    # 5. Save the Set C model used by the dashboard ---------------------------
    joblib.dump(sc["model"], config.SET_C_MODEL_PATH)
    meta = {
        "description": "Histogram gradient boosting on Set C, trained on the 70% training partition.",
        "features": set_c,
        "threshold": sc["threshold"],
        "threshold_rule": f"Highest validation threshold with recall >= {config.TARGET_RECALL}",
        "valid": sc["valid"],
        "test": sc["test"],
        "random_seed": config.RANDOM_SEED,
        "data_sha256": data_hash,
        "sklearn_version": sklearn.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    config.SET_C_META_PATH.write_text(json.dumps(meta, indent=2))
    log(f"Saved Set C model to {config.SET_C_MODEL_PATH.relative_to(PROJECT_ROOT)}")

    # 5b. Screening-tool models (gradient boosting for A, B and C) and the
    #     validation-based threshold grid by set and age group -----------------
    age_va, age_te = X_va["Age"].to_numpy(), X_te["Age"].to_numpy()
    grids = []
    for fs in feature_sets:
        r = fitted[(fs, "hgb")]
        grids.append(build_grid(fs, y_va, r["valid_scores"], age_va, y_te, r["test_scores"], age_te))
    pd.concat(grids, ignore_index=True).to_csv(config.THRESHOLD_GRID_PATH, index=False)
    joblib.dump({fs: fitted[(fs, "hgb")]["model"] for fs in feature_sets}, config.TOOL_MODELS_PATH)
    tool_meta = {
        "description": "Histogram gradient boosting per feature set, trained on the 70% training partition. "
                       "Thresholds in outputs/threshold_grid.csv were chosen on validation data.",
        "features": feature_sets,
        # Median training answer per question: the 'typical answer' used to
        # explain which answers raised a person's score.
        "typical_answers": {f: float(X_tr[f].median()) for f in config.SET_A},
        "test_roc_auc": {fs: fitted[(fs, "hgb")]["test"]["roc_auc"] for fs in feature_sets},
        "data_sha256": data_hash,
        "sklearn_version": sklearn.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    config.TOOL_META_PATH.write_text(json.dumps(tool_meta, indent=2))
    log("Saved screening-tool models and threshold grid")

    # 6. Repeated-split stability check (HGB only, Set C features held fixed) --
    if not args.skip_stability:
        stab = []
        for seed in config.STABILITY_SEEDS:
            sp = make_split(y, seed)
            for fs, feats in feature_sets.items():
                r = fit_evaluate("hgb", feats,
                                 X.iloc[sp.train_idx], yv[sp.train_idx],
                                 X.iloc[sp.valid_idx], yv[sp.valid_idx],
                                 X.iloc[sp.test_idx], yv[sp.test_idx], seed=seed)
                stab.append({"seed": seed, "feature_set": fs, "n_features": len(feats),
                             "threshold": r["threshold"],
                             **{f"test_{k}": r["test"][k] for k in
                                ("roc_auc", "average_precision", "recall", "specificity", "precision")}})
            log(f"Stability seed {seed} done")
        pd.DataFrame(stab).to_csv(config.OUTPUTS_DIR / "stability_hgb.csv", index=False)

    # 7. Compare with the preliminary single-split numbers --------------------
    comp = []
    for p in PRELIMINARY_RESULTS:
        m = results.query("feature_set == @p['feature_set'] and model == @p['model']")
        if m.empty:
            continue
        m = m.iloc[0]
        for metric in ("roc_auc", "average_precision", "recall", "specificity", "precision"):
            new = float(m[f"test_{metric}"])
            comp.append({"feature_set": p["feature_set"], "model": p["model"], "metric": metric,
                         "preliminary": p[metric], "recomputed": round(new, 4),
                         "difference": round(new - p[metric], 4)})
    pd.DataFrame(comp).to_csv(config.OUTPUTS_DIR / "preliminary_vs_recomputed.csv", index=False)
    setc_check = {
        "preliminary_features": PRELIMINARY_SET_C["features"],
        "recomputed_features": set_c,
        "same_features": sorted(PRELIMINARY_SET_C["features"]) == sorted(set_c),
        "preliminary_valid_roc_auc": PRELIMINARY_SET_C["valid_roc_auc"],
        "recomputed_valid_roc_auc": sc["valid"]["roc_auc"],
        "preliminary_valid_recall": PRELIMINARY_SET_C["valid_recall"],
        "recomputed_valid_recall": sc["valid"]["recall"],
        "preliminary_threshold": PRELIMINARY_SET_C["threshold"],
        "recomputed_threshold": sc["threshold"],
    }

    # 8. Figures ---------------------------------------------------------------
    figures.roc_figure(roc, results, config.FIGURES_DIR / "roc_curves_test.png")
    figures.pr_figure(pr, results, float(y_te.mean()), config.FIGURES_DIR / "pr_curves_test.png")
    figures.burden_figure(sweep, config.FIGURES_DIR / "set_c_burden_curve.png")
    figures.bmi_figure(df, config.FIGURES_DIR / "bmi_by_target.png")

    # 9. Run metadata -----------------------------------------------------------
    run_meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
        "data_file": Path(args.data).name,
        "data_sha256": data_hash,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "random_seed": config.RANDOM_SEED,
        "split_fractions": [config.TRAIN_FRACTION, config.VALID_FRACTION, config.TEST_FRACTION],
        "split": split_info,
        "target_recall": config.TARGET_RECALL,
        "set_c_rule": {"min_valid_roc_auc": config.SET_C_MIN_VAL_AUC,
                       "min_valid_recall": config.SET_C_MIN_VAL_RECALL},
        "set_c_selection_method": (
            "Greedy forward selection over Set B with histogram gradient boosting, scored by "
            f"ROC-AUC on an inner {int(config.INNER_HOLDOUT_FRACTION * 100)}% holdout of the "
            "TRAINING partition; then top-k validation sweep and the smallest k meeting the rule."
        ),
        "set_c": set_c,
        "set_c_rule_met": rule_met,
        "set_c_vs_preliminary": setc_check,
        "bootstrap_rounds": args.bootstrap,
        "stability_seeds": [] if args.skip_stability else list(config.STABILITY_SEEDS),
        "resampling": "none (class weighting only)",
        "rows_removed": 0,
        "bmi_clipped": False,
    }
    (config.OUTPUTS_DIR / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
    log(f"Done in {run_meta['runtime_seconds']} s")


if __name__ == "__main__":
    main()
