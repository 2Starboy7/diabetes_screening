import altair as alt
import pandas as pd
import streamlit as st

from screening import config
from screening.dashboard import (
    SET_COLORS,
    check_data_matches,
    data_audit,
    load_results,
    require_data,
)

require_data()
aud = data_audit()
res = load_results()
if res is not None:
    check_data_matches(res)

st.markdown(
    "**Research question.** How does screening performance change as the questionnaire "
    "is shortened from all predictors (**Set A**), to a non-invasive subset (**Set B**), "
    "to a minimal subset selected from Set B (**Set C**)?"
)
st.markdown(
    "The data are self-reported answers from the CDC Behavioral Risk Factor Surveillance "
    "System (BRFSS) 2015 telephone survey, as distributed in "
    f"`{config.PRIMARY_FILENAME}`. Every number on this page is computed from that file."
)

with st.container(horizontal=True):
    st.metric("Survey records", f"{aud['n_rows']:,}", border=True)
    st.metric("Predictor questions", aud["n_predictors"], border=True)
    st.metric("Screen-positive records", f"{aud['n_positive']:,}", border=True)
    st.metric("Positive share", f"{aud['positive_rate']:.2%}", border=True)
    st.metric("Missing cells", f"{aud['missing_cells']:,}", border=True)

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Target definition", anchor=False)
    st.markdown(
        "The survey variable `Diabetes_012` has three classes. The study combines "
        "prediabetes and diabetes into one **screen-positive** group, because the aim "
        "is to find people who may need follow-up testing."
    )
    counts = aud["raw_class_counts"]
    target_tbl = pd.DataFrame(
        {
            "Diabetes_012": [0, 1, 2],
            "Survey meaning": ["No diabetes", "Prediabetes", "Diabetes"],
            "Records": [counts["0"], counts["1"], counts["2"]],
            "Study label": ["Negative", "Positive", "Positive"],
        }
    )
    target_tbl["Share of all records"] = target_tbl["Records"] / aud["n_rows"]
    st.dataframe(
        target_tbl,
        hide_index=True,
        column_config={
            "Records": st.column_config.NumberColumn(format="localized"),
            "Share of all records": st.column_config.NumberColumn(format="percent"),
        },
        alt="Records per Diabetes_012 class and the study label each class maps to",
    )

with right:
    st.subheader("Class balance", anchor=False)
    chart_df = target_tbl.assign(
        Class=target_tbl["Diabetes_012"].map({0: "0 · No diabetes", 1: "1 · Prediabetes", 2: "2 · Diabetes"})
    )
    bars = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            y=alt.Y("Class:N", title=None, sort=None, axis=alt.Axis(labelOverlap=False, labelLimit=200)),
            x=alt.X("Records:Q", title="Number of records"),
            color=alt.Color(
                "Study label:N",
                scale=alt.Scale(domain=["Negative", "Positive"], range=["#56B4E9", "#D55E00"]),
                legend=alt.Legend(orient="bottom", title="Study label"),
            ),
            tooltip=["Class", alt.Tooltip("Records:Q", format=","), alt.Tooltip("Share of all records:Q", format=".2%")],
        )
        .properties(height=alt.Step(44))
    )
    labels = bars.mark_text(align="left", dx=4).encode(
        text=alt.Text("Records:Q", format=","), color=alt.value("#888888")
    )
    st.altair_chart(bars + labels, alt="Most records are class 0; about 16% are prediabetes or diabetes")
    st.caption(
        f"Natural class balance is kept throughout: {aud['positive_rate']:.2%} of records are "
        "screen-positive. No rows are removed and no synthetic records are added."
    )

st.subheader("Three questionnaire lengths", anchor=False)
set_c = res.run_meta["set_c"] if res is not None else None
cards = [
    ("A", "Full benchmark", config.SET_A,
     "All 21 predictors, including five that depend on prior medical contact."),
    ("B", "Non-invasive", config.SET_B,
     "Set A without high blood pressure, high cholesterol, cholesterol check, stroke, and "
     "heart disease or attack. Each remaining question can be answered without a prior diagnosis."),
    ("C", "Minimal", set_c or [],
     "The smallest subset of Set B that met the study rule on validation data. "
     "Selected using training data only (see *Compare models*)."),
]
cols = st.columns(3, gap="medium")
for col, (key, name, feats, desc) in zip(cols, cards):
    with col, st.container(border=True, height="stretch"):
        st.markdown(f"#### Set {key} · {name}")
        st.markdown(f"**{len(feats) if feats else '—'} questions.** {desc}")
        if feats:
            st.caption(", ".join(config.FEATURE_LABELS[f] for f in feats))
        else:
            st.caption("Run the analysis to select Set C.")

if res is not None:
    st.subheader("Headline result (held-out test set)", anchor=False)
    hgb = res.models[res.models["model"] == "hgb"].set_index("feature_set")
    base_auc = hgb.loc["A", "test_roc_auc"]
    with st.container(horizontal=True):
        for fs in ("A", "B", "C"):
            r = hgb.loc[fs]
            delta = 0 if fs == "A" else round(float(r["test_roc_auc"] - base_auc), 4)
            st.metric(
                f"Set {fs} · {int(r['n_features'])} questions · ROC-AUC",
                f"{r['test_roc_auc']:.3f}",
                delta=delta,
                delta_color="off",
                delta_description="reference" if fs == "A" else "vs. Set A",
                border=True,
                help="Histogram gradient boosting model. Delta compares with Set A.",
            )
        for fs in ("A", "B", "C"):
            r = hgb.loc[fs]
            st.metric(
                f"Set {fs} · specificity at the recall target",
                f"{r['test_specificity']:.1%}",
                border=True,
                help=f"Share of negatives correctly not flagged, at the validation threshold "
                     f"chosen for ≥{config.TARGET_RECALL:.0%} recall. Test recall: {r['test_recall']:.1%}.",
            )
    st.caption(
        "Gradient-boosting results shown; all models are on *Compare models*. These come "
        "from one stratified 70/15/15 split (seed 42). Repeated-split checks are reported there too."
    )

st.subheader("What this prototype does and does not establish", anchor=False)
does, doesnt = st.columns(2, gap="medium")
with does, st.container(border=True, height="stretch"):
    st.markdown("**It does**")
    st.markdown(
        "- Compare discrimination of models built from shorter question sets on the same split.\n"
        "- Report recall, specificity, and precision at a threshold fixed on validation data.\n"
        "- Show the trade-off between number of questions and validation performance."
    )
with doesnt, st.container(border=True, height="stretch"):
    st.markdown("**It does not**")
    st.markdown(
        "- Diagnose diabetes or estimate an individual's probability of having it.\n"
        "- Validate the models on another population, survey year, or clinical setting.\n"
        "- Replace laboratory tests (e.g., HbA1c or fasting glucose) or clinical judgement."
    )

with st.expander("About the other CSV files in the original download"):
    st.markdown(
        "- `diabetes_binary_health_indicators_BRFSS2015.csv` uses a **different target**: "
        "prediabetes is grouped with *no diabetes*, giving 35,346 positives instead of 39,977.\n"
        "- `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` is an **artificially "
        "balanced** subsample (35,346 per class).\n\n"
        "Neither matches this study's target or natural class balance, so neither is used here."
    )
