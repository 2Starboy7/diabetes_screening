import altair as alt
import pandas as pd
import streamlit as st

from screening import config
from screening.dashboard import SET_COLORS, check_data_matches, fmt_ci, require_results

res = require_results()
check_data_matches(res)
m = res.models.copy()
meta = res.run_meta

SET_SCALE = alt.Scale(domain=list(SET_COLORS), range=list(SET_COLORS.values()))
m["Set"] = m["feature_set"]
m["Model"] = m["model_label"]

split = meta["split"]
st.markdown(
    f"Every model was trained on the **training** partition ({split['train']['n']:,} records), "
    f"its decision threshold was chosen on the **validation** partition ({split['validation']['n']:,}) "
    f"as the highest score that still gives at least {config.TARGET_RECALL:.0%} recall, and it was "
    f"scored once on the **held-out test** partition ({split['test']['n']:,}). "
    "Class weighting was used; no synthetic resampling. All numbers below are recomputed by "
    "`scripts/run_analysis.py`."
)

with st.expander("What the metrics mean", icon=":material/menu_book:"):
    st.markdown(
        "- **ROC-AUC** — chance that a randomly chosen positive record gets a higher score than a "
        "randomly chosen negative one. 0.5 = no better than chance, 1.0 = perfect ranking. "
        "Does not depend on the threshold.\n"
        "- **Average precision (AP)** — summarises the precision–recall curve as a weighted mean of "
        "precision at each recall level. Its no-skill baseline equals the positive share "
        f"(≈{split['test']['positive_rate']:.3f} here), not 0.5. Reported separately from the "
        "trapezoidal *PR-AUC*, which is a different calculation (shown in the full table).\n"
        "- **Recall (sensitivity)** — share of positive records that the model flags.\n"
        "- **Specificity** — share of negative records that the model does not flag.\n"
        "- **Precision (positive predictive value)** — share of flagged records that are positive. "
        "It depends heavily on how common positives are.\n"
        f"- **Accuracy** is shown only in the full table: with {1 - split['test']['positive_rate']:.0%} "
        "negatives, flagging nobody would already be that accurate, so it is a poor guide for screening."
    )

# ---------------------------------------------------------------------------
st.subheader("Held-out test results", anchor=False)
show = pd.DataFrame({
    "Set": m["Set"],
    "Model": m["Model"],
    "Questions": m["n_features"],
    "ROC-AUC (95% CI)": [fmt_ci(r["test_roc_auc"], r.get("test_roc_auc_ci_low"), r.get("test_roc_auc_ci_high"))
                         for _, r in m.iterrows()],
    "Average precision (95% CI)": [fmt_ci(r["test_average_precision"], r.get("test_average_precision_ci_low"),
                                          r.get("test_average_precision_ci_high")) for _, r in m.iterrows()],
    "Recall": m["test_recall"],
    "Specificity": m["test_specificity"],
    "Precision": m["test_precision"],
    "Validation threshold": m["threshold"],
})
st.dataframe(
    show,
    hide_index=True,
    column_config={
        "Recall": st.column_config.NumberColumn(format="%.3f"),
        "Specificity": st.column_config.NumberColumn(format="%.3f"),
        "Precision": st.column_config.NumberColumn(format="%.3f"),
        "Validation threshold": st.column_config.NumberColumn(format="%.4f"),
    },
    alt="Test-set ROC-AUC, average precision, recall, specificity and precision for each model and set",
)
st.caption(
    f"Intervals are 95% percentile bootstrap intervals from {meta['bootstrap_rounds']} resamples of the "
    "test rows with the model held fixed; they reflect test-sample variation only, not retraining. "
    "Set C was selected with gradient boosting; logistic regression and random forest on Set C are "
    "shown for comparison only."
)

# ---------------------------------------------------------------------------
st.subheader("Compare a metric across sets", anchor=False)
metric_opts = {
    "ROC-AUC": "roc_auc",
    "Average precision": "average_precision",
    "Specificity at the recall target": "specificity",
    "Precision at the recall target": "precision",
}
choice = st.segmented_control("Metric", list(metric_opts), default="ROC-AUC", key="cmp_metric")
key = metric_opts[choice or "ROC-AUC"]
plot = m[["Set", "Model", "n_features", f"test_{key}"]].rename(columns={f"test_{key}": "Value"})
lo_col, hi_col = f"test_{key}_ci_low", f"test_{key}_ci_high"
has_ci = lo_col in m.columns
if has_ci:
    plot["Low"], plot["High"] = m[lo_col], m[hi_col]
base = alt.Chart(plot).encode(
    x=alt.X("Set:N", title=None, axis=alt.Axis(labelExpr="'Set ' + datum.value")),
    xOffset="Set:N",
)
bars = base.mark_bar().encode(
    y=alt.Y("Value:Q", title=f"{choice or 'ROC-AUC'} (test set)", scale=alt.Scale(zero=False)),
    color=alt.Color("Set:N", scale=SET_SCALE, legend=None),
    tooltip=["Set", "Model", alt.Tooltip("n_features:Q", title="Questions"), alt.Tooltip("Value:Q", format=".4f")],
)
layers = [bars]
if has_ci:
    layers.append(base.mark_rule(color="#333333").encode(y="Low:Q", y2="High:Q"))
chart = (
    alt.layer(*layers)
    .properties(height=280, width=140)
    .facet(column=alt.Column("Model:N", title=None, sort=list(config.MODEL_LABELS.values())))
    .resolve_scale(y="shared")
)
st.altair_chart(chart, alt=f"{choice} on the test set for each model and feature set")
st.caption("Bars start above zero to make differences visible; black lines are 95% bootstrap intervals where available.")

# ---------------------------------------------------------------------------
st.subheader("ROC and precision–recall curves (test set)", anchor=False)
model_pick = st.pills(
    "Models to show", list(config.MODEL_LABELS.values()), selection_mode="multi",
    default=["Histogram gradient boosting"], key="curve_models",
)
picked = [k for k, v in config.MODEL_LABELS.items() if v in (model_pick or [])]
roc = res.roc[res.roc["model"].isin(picked)].assign(Model=lambda d: d["model"].map(config.MODEL_LABELS))
pr = res.pr[res.pr["model"].isin(picked)].assign(Model=lambda d: d["model"].map(config.MODEL_LABELS))
if roc.empty:
    st.info("Pick at least one model to draw curves.")
else:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="grey", strokeDash=[2, 2]).encode(x="x", y="y")
        ch = alt.Chart(roc).mark_line().encode(
            x=alt.X("fpr:Q", title="False-positive rate (1 − specificity)"),
            y=alt.Y("tpr:Q", title="True-positive rate (recall)"),
            color=alt.Color("feature_set:N", scale=SET_SCALE, title="Set"),
            strokeDash=alt.StrokeDash("Model:N", legend=alt.Legend(orient="bottom")),
            tooltip=["feature_set", "Model", alt.Tooltip("fpr:Q", format=".3f"), alt.Tooltip("tpr:Q", format=".3f")],
        )
        st.altair_chart((diag + ch).properties(title="ROC curves", height=360),
                        alt="Test-set ROC curves for the selected models on Sets A, B and C")
        st.caption("Dashed grey diagonal = no discrimination.")
    with c2:
        prev = split["test"]["positive_rate"]
        base_line = alt.Chart(pd.DataFrame({"y": [prev]})).mark_rule(color="grey", strokeDash=[2, 2]).encode(y="y")
        ch = alt.Chart(pr).mark_line().encode(
            x=alt.X("recall:Q", title="Recall"),
            y=alt.Y("precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("feature_set:N", scale=SET_SCALE, title="Set"),
            strokeDash=alt.StrokeDash("Model:N", legend=alt.Legend(orient="bottom")),
            tooltip=["feature_set", "Model", alt.Tooltip("recall:Q", format=".3f"), alt.Tooltip("precision:Q", format=".3f")],
        )
        st.altair_chart((base_line + ch).properties(title="Precision–recall curves", height=360),
                        alt="Precision-recall curves: precision falls as recall rises for all sets")
        st.caption(f"Dashed grey line = positive share in the test set ({prev:.3f}), the no-skill level.")

# ---------------------------------------------------------------------------
st.subheader("Confusion counts at the validation threshold", anchor=False)
c1, c2 = st.columns([1, 2], gap="large")
with c1:
    fs_pick = st.segmented_control("Feature set", ["A", "B", "C"], default="C", key="cm_set",
                                   format_func=lambda s: f"Set {s}")
    mdl_pick = st.selectbox("Model", list(config.MODEL_LABELS), index=2, key="cm_model",
                            format_func=config.MODEL_LABELS.get)
row = m[(m["feature_set"] == (fs_pick or "C")) & (m["model"] == mdl_pick)].iloc[0]
with c2:
    cm = pd.DataFrame(
        {"Flagged by model": [int(row.test_tp), int(row.test_fp)],
         "Not flagged": [int(row.test_fn), int(row.test_tn)]},
        index=["Actually positive", "Actually negative"],
    )
    st.table(cm, alt="Confusion counts on the test set")
    n = row.test_tp + row.test_fp + row.test_tn + row.test_fn
    per = 1000 / n
    st.markdown(
        f"Threshold **{row.threshold:.4f}** (chosen on validation). Out of every 1,000 test records, "
        f"the model flags about **{(row.test_tp + row.test_fp) * per:.0f}**: "
        f"**{row.test_tp * per:.0f}** positive and **{row.test_fp * per:.0f}** negative. It misses "
        f"about **{row.test_fn * per:.0f}** positives. Test recall {row.test_recall:.1%}, "
        f"specificity {row.test_specificity:.1%}, precision {row.test_precision:.1%}."
    )

thr_tbl = m[["Set", "Model", "threshold", "valid_recall", "valid_specificity", "test_recall", "test_specificity"]]
with st.expander("Validation thresholds and how they carried over to test"):
    st.dataframe(
        thr_tbl.rename(columns={"threshold": "Threshold", "valid_recall": "Validation recall",
                                "valid_specificity": "Validation specificity", "test_recall": "Test recall",
                                "test_specificity": "Test specificity"}),
        hide_index=True,
        column_config={c: st.column_config.NumberColumn(format="%.4f") for c in
                       ["Threshold", "Validation recall", "Validation specificity", "Test recall", "Test specificity"]},
        alt="Validation thresholds with validation and test recall and specificity",
    )
    st.caption("Thresholds are not re-tuned on test, so test recall can land slightly above or below 75%.")

# ---------------------------------------------------------------------------
st.subheader("How Set C was selected", anchor=False)
sw = res.sweep
k_sel = int(sw.loc[sw["selected"], "k"].iloc[0])
st.markdown(
    f"1. **Ranking (training data only).** {meta['set_c_selection_method']}\n"
    f"2. **Sweep (validation data).** For k = 1 … {len(sw)}, a gradient-boosting model on the top-k "
    "questions was trained on the training partition and its threshold set on validation.\n"
    f"3. **Rule.** Choose the smallest k with validation ROC-AUC ≥ {config.SET_C_MIN_VAL_AUC} and "
    f"validation recall ≥ {config.SET_C_MIN_VAL_RECALL:.0%}. Result: **k = {k_sel}** "
    f"({', '.join(config.FEATURE_LABELS[f] for f in meta['set_c'])})."
)
sw_long = sw.melt(id_vars=["k", "features", "selected"],
                  value_vars=["valid_roc_auc", "valid_specificity", "valid_average_precision"],
                  var_name="Metric", value_name="Value")
sw_long["Metric"] = sw_long["Metric"].map({
    "valid_roc_auc": "ROC-AUC", "valid_specificity": "Specificity at ≥75% recall",
    "valid_average_precision": "Average precision"})
lines = alt.Chart(sw_long).mark_line(point=True).encode(
    x=alt.X("k:O", title="Number of Set B questions kept (forward-ranked on training data)"),
    y=alt.Y("Value:Q", title="Validation metric", scale=alt.Scale(zero=False)),
    color=alt.Color("Metric:N", scale=alt.Scale(range=["#009E73", "#E69F00", "#CC79A7"]),
                    legend=alt.Legend(orient="bottom")),
    tooltip=["k", "features", "Metric", alt.Tooltip("Value:Q", format=".4f")],
)
rule = alt.Chart(pd.DataFrame({"y": [config.SET_C_MIN_VAL_AUC]})).mark_rule(color="grey", strokeDash=[4, 3]).encode(y="y")
sel = alt.Chart(pd.DataFrame({"k": [k_sel]})).mark_rule(color="black", strokeDash=[6, 3]).encode(x="k:O")
st.altair_chart((lines + rule + sel).properties(title="Question burden vs. validation performance", height=340),
                alt=f"Validation ROC-AUC rises quickly over the first few questions; k = {k_sel} selected")
st.caption(f"Grey dashed line: ROC-AUC rule ({config.SET_C_MIN_VAL_AUC}). Black dashed line: selected k.")
with st.expander("Sweep table and forward ranking"):
    st.dataframe(sw, hide_index=True, alt="Validation sweep over the number of questions kept")
    st.dataframe(res.ranking, hide_index=True, alt="Order in which Set B questions were added during forward ranking")

# ---------------------------------------------------------------------------
if res.stability is not None:
    st.subheader("Does the ranking hold on other splits?", anchor=False)
    s = res.stability
    agg = s.groupby("feature_set").agg(
        Questions=("n_features", "first"),
        Splits=("seed", "nunique"),
        **{f"{lbl} mean": (f"test_{k}", "mean") for k, lbl in
           [("roc_auc", "ROC-AUC"), ("average_precision", "AP"), ("recall", "Recall"), ("specificity", "Specificity")]},
        **{f"{lbl} SD": (f"test_{k}", "std") for k, lbl in [("roc_auc", "ROC-AUC"), ("specificity", "Specificity")]},
    ).reset_index().rename(columns={"feature_set": "Set"})
    st.dataframe(agg, hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.4f") for c in agg.columns
                                if c not in ("Set", "Questions", "Splits")},
                 alt="Mean and standard deviation of test metrics across repeated splits")
    st.caption(
        f"Gradient boosting re-trained on {s['seed'].nunique()} different stratified 70/15/15 splits "
        f"(seeds {', '.join(map(str, sorted(s['seed'].unique())))}), with thresholds re-chosen on each "
        "validation partition. Set C's questions were held fixed (not re-selected per split)."
    )

# ---------------------------------------------------------------------------
st.subheader("Preliminary vs. recomputed numbers", anchor=False)
p = res.prelim.copy()
p["Set"] = p["feature_set"]
p["Model"] = p["model"].map(config.MODEL_LABELS)
p["Metric"] = p["metric"].map({"roc_auc": "ROC-AUC", "average_precision": "Average precision",
                               "recall": "Recall", "specificity": "Specificity", "precision": "Precision"})
max_diff = p["difference"].abs().max()
st.markdown(
    "The project brief listed preliminary single-split results. This run recomputed them from the "
    f"data and code in this project. The largest absolute difference is **{max_diff:.4f}**."
)
setc_chk = meta["set_c_vs_preliminary"]
st.markdown(
    f"Set C: preliminary {setc_chk['preliminary_features']} vs. recomputed {setc_chk['recomputed_features']} "
    f"({'same questions' if setc_chk['same_features'] else 'different questions'}); validation threshold "
    f"{setc_chk['preliminary_threshold']:.4f} vs. {setc_chk['recomputed_threshold']:.4f}."
)
with st.expander("Full comparison table"):
    st.dataframe(p[["Set", "Model", "Metric", "preliminary", "recomputed", "difference"]], hide_index=True,
                 alt="Preliminary values next to recomputed values and their difference")
    st.caption(
        "Differences can come from model settings that were not recorded with the preliminary run "
        "(for example random-forest depth or boosting iterations), library versions, or how the "
        "split was drawn. The dashboard always shows the recomputed values."
    )

with st.expander("Full results table (all columns, CSV)"):
    st.dataframe(m.drop(columns=["Set", "Model"]), hide_index=True, alt="All computed metrics for every model")
    st.download_button("Download model_results.csv", m.drop(columns=["Set", "Model"]).to_csv(index=False),
                       file_name="model_results.csv", mime="text/csv", icon=":material/download:")
