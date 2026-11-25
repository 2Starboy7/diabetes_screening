import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from screening import config
from screening.dashboard import (
    CLASS_COLORS,
    DISCLAIMER,
    check_data_matches,
    load_set_c_model,
    require_results,
)

res = require_results()
check_data_matches(res)
model, meta = load_set_c_model()
if model is None:
    st.error("The Set C model file is missing from `models/`. Run `py -3.13 scripts/run_analysis.py`.",
             title="Model not found")
    st.stop()

features: list[str] = meta["features"]
threshold: float = float(meta["threshold"])

st.warning(
    DISCLAIMER + " The number shown is an uncalibrated model score, not your probability of "
    "having diabetes, and it gives no treatment or care recommendation.",
    title="Not a diagnosis or medical advice",
    icon=":material/info:",
)
st.markdown(
    f"This page runs the **Set C** gradient-boosting model ({len(features)} questions) exactly as it "
    "was evaluated on *Compare models*: same training data, same questions, and the same "
    f"research threshold (**{threshold:.4f}**), which was chosen on validation data as the highest "
    f"score that still flagged at least {config.TARGET_RECALL:.0%} of positive validation records."
)


def bmi_input(key: str) -> int:
    val = st.number_input(
        "Body mass index (kg/m²)", min_value=12, max_value=98, value=27, step=1, key=key,
        help="The survey stores BMI as a whole number. Valid range in the training data: 12–98.",
    )
    with st.expander("Calculate BMI from height and weight"):
        h = st.number_input("Height (cm)", 100.0, 230.0, 170.0, 0.5, key=f"{key}_h")
        w = st.number_input("Weight (kg)", 30.0, 300.0, 78.0, 0.5, key=f"{key}_w")
        bmi = w / (h / 100) ** 2
        st.caption(f"BMI = weight ÷ height² = **{bmi:.1f} kg/m²** (enter {round(bmi)} above).")
    return int(val)


def feature_input(f: str):
    """One input widget per Set C question, using the survey's own categories."""
    if f == "Age":
        return st.selectbox("Age group (years)", list(config.AGE_LABELS), index=7,
                            format_func=config.AGE_LABELS.get, key="in_Age")
    if f == "GenHlth":
        v = st.segmented_control(
            "In general, how would you rate your health?", list(config.GENHLTH_LABELS),
            default=3, format_func=config.GENHLTH_LABELS.get, key="in_GenHlth",
        )
        return 3 if v is None else v
    if f == "BMI":
        return bmi_input("in_BMI")
    if f in ("MentHlth", "PhysHlth"):
        return st.slider(config.FEATURE_LABELS[f], 0, 30, 0, key=f"in_{f}")
    if f in ("Education", "Income"):
        lo, hi = (1, 6) if f == "Education" else (1, 8)
        return st.selectbox(config.FEATURE_LABELS[f], list(range(lo, hi + 1)), index=hi - lo, key=f"in_{f}")
    if f == "Sex":
        v = st.segmented_control("Sex", [0, 1], default=0, format_func={0: "Female", 1: "Male"}.get, key="in_Sex")
        return 0 if v is None else v
    v = st.segmented_control(config.FEATURE_LABELS[f], [0, 1], default=0,
                             format_func={0: "No", 1: "Yes"}.get, key=f"in_{f}")
    return 0 if v is None else v


left, right = st.columns([2, 3], gap="large")
with left, st.container(border=True):
    st.markdown("**Answers**")
    answers = {f: feature_input(f) for f in features}

x = pd.DataFrame([answers])[features].astype(float)
score = float(model.predict_proba(x)[:, 1][0])
above = score >= threshold
test = meta["test"]

with right:
    with st.container(horizontal=True):
        st.metric("Uncalibrated model score", f"{score:.3f}", border=True,
                  help="Output of the model on a 0–1 scale. Class weighting during training shifts these "
                       "scores upward, so they are NOT probabilities of having diabetes.")
        st.metric("Research threshold", f"{threshold:.3f}", border=True,
                  help="Chosen on validation data to reach at least 75% recall.")
    if above:
        st.info(
            "This answer pattern scores **at or above** the research threshold. In the study, records "
            "like this are the ones the model would flag for follow-up testing.",
            icon=":material/flag:",
        )
    else:
        st.info(
            "This answer pattern scores **below** the research threshold. In the study, records like "
            "this would not be flagged. Many positive records also score below the threshold.",
            icon=":material/remove:",
        )
    st.markdown(
        f"**What the threshold means at group level (test set, {test['tp'] + test['fp'] + test['tn'] + test['fn']:,} records).** "
        f"Of all records the model flagged, **{test['precision']:.1%}** were screen-positive. Of records it "
        f"did not flag, **{test['npv']:.1%}** were negative. It flagged **{test['recall']:.1%}** of positive "
        f"records and correctly left **{test['specificity']:.1%}** of negative records unflagged. These are "
        "properties of the group, not a statement about one person."
    )

    hist = res.score_hist.copy()
    hist["Group"] = hist["target"].map({0: "Negative", 1: "Positive"})
    totals = hist.groupby("Group")["count"].transform("sum")
    hist["Share of group"] = hist["count"] / totals
    base_hist = alt.Chart(hist).mark_bar(opacity=0.6).encode(
        x=alt.X("bin_start:Q", bin="binned", title="Uncalibrated model score", scale=alt.Scale(domain=[0, 1])),
        x2="bin_end:Q",
        y=alt.Y("Share of group:Q", stack=None, axis=alt.Axis(format="%"), title="Share of test records in group"),
        color=alt.Color("Group:N", scale=alt.Scale(domain=list(CLASS_COLORS), range=list(CLASS_COLORS.values())),
                        legend=alt.Legend(orient="bottom", title="Study label")),
        tooltip=["Group", alt.Tooltip("bin_start:Q", title="Score from", format=".3f"),
                 alt.Tooltip("bin_end:Q", title="Score to", format=".3f"),
                 alt.Tooltip("Share of group:Q", format=".1%")],
    )
    marks = pd.DataFrame({"x": [threshold, score], "Line": ["Research threshold", "This answer pattern"]})
    rules = alt.Chart(marks).mark_rule(strokeWidth=2).encode(
        x="x:Q",
        strokeDash=alt.StrokeDash("Line:N", scale=alt.Scale(domain=["Research threshold", "This answer pattern"],
                                                            range=[[6, 3], [1, 0]]),
                                  legend=alt.Legend(orient="bottom", title=None)),
        color=alt.value("#222222"),
        tooltip=["Line", alt.Tooltip("x:Q", title="Score", format=".3f")],
    )
    st.altair_chart((base_hist + rules).properties(title="Where this score sits among test-set scores", height=280),
                    alt="Distribution of Set C test scores by study label with the threshold and the current score")

if "BMI" in features:
    st.subheader("How the score changes with BMI (other answers fixed)", anchor=False)
    bmi_values = np.arange(15, 61)
    grid = pd.DataFrame({f: [answers[f]] * len(bmi_values) for f in features})
    grid["BMI"] = bmi_values
    grid["score"] = model.predict_proba(grid[features].astype(float))[:, 1]
    line = alt.Chart(grid).mark_line(color="#009E73", strokeWidth=2.5).encode(
        x=alt.X("BMI:Q", title="BMI (kg/m²)"),
        y=alt.Y("score:Q", title="Uncalibrated model score", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("BMI:Q"), alt.Tooltip("score:Q", format=".3f", title="Score")],
    )
    thr_rule = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(color="#222222", strokeDash=[6, 3]).encode(y="y:Q")
    point = alt.Chart(pd.DataFrame({"BMI": [answers["BMI"]], "score": [score]})).mark_point(
        size=120, filled=True, color="#222222").encode(x="BMI:Q", y="score:Q")
    st.altair_chart((line + thr_rule + point).properties(height=300),
                    alt="Model score across BMI values for the selected other answers, with the threshold line")
    st.caption(
        "Shows how the model responds, not how risk works biologically. Steps in the line come from "
        "the tree-based model. Dashed line = research threshold; dot = current answers."
    )

with st.expander("About this model", icon=":material/description:"):
    st.markdown(
        f"- **Questions:** {', '.join(config.FEATURE_LABELS[f] for f in features)}\n"
        f"- **Model:** histogram gradient boosting with class weighting, trained on the 70% training partition "
        f"(seed {meta['random_seed']}).\n"
        f"- **Threshold:** {meta['threshold_rule']} → {threshold:.4f}.\n"
        f"- **Validation:** ROC-AUC {meta['valid']['roc_auc']:.4f}, recall {meta['valid']['recall']:.4f}.\n"
        f"- **Test:** ROC-AUC {test['roc_auc']:.4f}, recall {test['recall']:.4f}, specificity {test['specificity']:.4f}.\n"
        f"- **Built:** {meta['created_utc']} with scikit-learn {meta['sklearn_version']}; "
        f"data SHA-256 `{meta['data_sha256'][:16]}…`.\n"
        "- **Not established:** calibration, performance in other populations or years, and clinical usefulness."
    )
