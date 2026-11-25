import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from screening import config
from screening.dashboard import CLASS_COLORS, data_audit, load_data, require_data

require_data()
df = load_data()
aud = data_audit()

CLASS_SCALE = alt.Scale(domain=list(CLASS_COLORS), range=list(CLASS_COLORS.values()))


@st.cache_data(show_spinner=False)
def missing_table() -> pd.DataFrame:
    d = load_data()
    cols = [config.RAW_TARGET, *config.SET_A]
    return pd.DataFrame({
        "Column": cols,
        "Missing cells": [int(d[c].isna().sum()) for c in cols],
        "Distinct values": [int(d[c].nunique()) for c in cols],
    })


@st.cache_data(show_spinner=False)
def rate_by(col: str) -> pd.DataFrame:
    d = load_data()
    g = d.groupby(col)[config.TARGET].agg(["mean", "size"]).reset_index()
    g.columns = [col, "Positive share", "Records"]
    return g


@st.cache_data(show_spinner=False)
def bmi_hist(width: int) -> pd.DataFrame:
    d = load_data()
    edges = np.arange(10, 100 + width, width)
    rows = []
    for val, lbl in ((0, "Negative"), (1, "Positive")):
        counts, _ = np.histogram(d.loc[d[config.TARGET] == val, "BMI"], bins=edges)
        rows.append(pd.DataFrame({
            "BMI from": edges[:-1], "BMI to": edges[1:], "Group": lbl,
            "Records": counts, "Share of group": counts / counts.sum(),
        }))
    return pd.concat(rows, ignore_index=True)


@st.cache_data(show_spinner=False)
def predictor_summary() -> pd.DataFrame:
    d = load_data()
    pos = d[d[config.TARGET] == 1]
    neg = d[d[config.TARGET] == 0]
    group_of = {f: g for g, fs in config.FEATURE_GROUPS.items() for f in fs}
    rows = []
    for f in config.SET_A:
        n_unique = d[f].nunique()
        kind = "Yes/No (0/1)" if n_unique == 2 else ("Count, 0–30 days" if f in ("MentHlth", "PhysHlth")
                                                   else "Numeric" if f == "BMI" else "Ordered category")
        rows.append({
            "Question": config.FEATURE_LABELS[f],
            "Column": f,
            "Group": group_of[f],
            "In Set B": f in config.SET_B,
            "Type": kind,
            "Min": d[f].min(), "Max": d[f].max(),
            "Mean (all)": d[f].mean(),
            "Mean (negative)": neg[f].mean(),
            "Mean (positive)": pos[f].mean(),
        })
    return pd.DataFrame(rows)


tab_missing, tab_class, tab_bmi, tab_pred, tab_repeat = st.tabs(
    ["Missingness", "Class distribution", "BMI", "Predictor summaries", "Repeated patterns"]
)

with tab_missing:
    mt = missing_table()
    if mt["Missing cells"].sum() == 0:
        st.success(f"No missing cells in any of the {len(mt)} columns ({aud['n_rows']:,} rows).",
                   icon=":material/check_circle:")
    st.dataframe(mt, hide_index=True, alt="Missing cell count and number of distinct values per column")
    st.caption(
        "\"Missing\" means an empty cell in the CSV. The distributed file already encodes survey "
        "answers as numbers, so refusals or \"don't know\" answers from the original survey cannot "
        "be identified from this file."
    )

with tab_class:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        age = rate_by("Age").assign(**{"Age group": lambda t: t["Age"].astype(int).map(config.AGE_LABELS)})
        ch = (
            alt.Chart(age).mark_bar(color="#0072B2")
            .encode(
                x=alt.X("Age group:N", sort=list(config.AGE_LABELS.values()), title="Age group (years)"),
                y=alt.Y("Positive share:Q", axis=alt.Axis(format="%"), title="Screen-positive share"),
                tooltip=["Age group", alt.Tooltip("Positive share:Q", format=".1%"),
                         alt.Tooltip("Records:Q", format=",")],
            )
            .properties(title="Screen-positive share by age group", height=300)
        )
        st.altair_chart(ch, alt="Screen-positive share rises steadily with age group")
    with c2:
        gh = rate_by("GenHlth").assign(**{"General health": lambda t: t["GenHlth"].astype(int).map(config.GENHLTH_LABELS)})
        ch = (
            alt.Chart(gh).mark_bar(color="#009E73")
            .encode(
                x=alt.X("General health:N", sort=list(config.GENHLTH_LABELS.values()),
                        title="Self-rated general health"),
                y=alt.Y("Positive share:Q", axis=alt.Axis(format="%"), title="Screen-positive share"),
                tooltip=["General health", alt.Tooltip("Positive share:Q", format=".1%"),
                         alt.Tooltip("Records:Q", format=",")],
            )
            .properties(title="Screen-positive share by self-rated health", height=300)
        )
        st.altair_chart(ch, alt="Screen-positive share is higher for poorer self-rated health")
    st.caption(
        "Descriptive shares from all records. They describe associations in this survey sample "
        "and do not show cause and effect."
    )

with tab_bmi:
    b = aud["bmi"]
    with st.container(horizontal=True):
        st.metric("Minimum BMI", f"{b['min']:.0f}", border=True)
        st.metric("Median BMI", f"{b['median']:.0f}", border=True)
        st.metric("99th percentile", f"{b['p99']:.0f}", border=True)
        st.metric("Maximum BMI", f"{b['max']:.0f}", border=True)
        st.metric("Records with BMI > 60", f"{b['n_above_60']:,}", border=True)
    width = st.segmented_control("Bin width (kg/m²)", [1, 2, 5], default=2, key="bmi_bin")
    hist = bmi_hist(width or 2)
    ch = (
        alt.Chart(hist).mark_bar(opacity=0.65)
        .encode(
            x=alt.X("BMI from:Q", bin="binned", title="BMI (kg/m², as recorded)", scale=alt.Scale(domain=[10, 100])),
            x2="BMI to:Q",
            y=alt.Y("Share of group:Q", stack=None, axis=alt.Axis(format="%"), title="Share of records in group"),
            color=alt.Color("Group:N", scale=CLASS_SCALE, legend=alt.Legend(orient="bottom", title="Study label")),
            tooltip=["Group", "BMI from", "BMI to", alt.Tooltip("Records:Q", format=","),
                     alt.Tooltip("Share of group:Q", format=".2%")],
        )
        .properties(title="BMI distribution, negative vs. positive records", height=320)
    )
    st.altair_chart(ch, alt="Positive records are shifted toward higher BMI than negative records")
    st.markdown(
        f"BMI values are whole numbers between {b['min']:.0f} and {b['max']:.0f}. A long right tail "
        f"exists ({b['n_above_60']:,} records above 60). **BMI is not clipped or winsorized**: "
        "without the original measurements there is no evidence to tell data-entry errors from "
        "genuine extreme values. Tree-based models split on value order, so extreme values have "
        "limited influence on them; logistic regression may be more sensitive. "
        "Any future clipping should be justified, documented, and fitted on training data only."
    )

with tab_pred:
    st.dataframe(
        predictor_summary(),
        hide_index=True,
        column_config={
            "In Set B": st.column_config.CheckboxColumn(help="Unchecked = prior-diagnosis question, Set A only"),
            "Mean (all)": st.column_config.NumberColumn(format="%.3f"),
            "Mean (negative)": st.column_config.NumberColumn(format="%.3f"),
            "Mean (positive)": st.column_config.NumberColumn(format="%.3f"),
        },
        alt="Summary of all 21 predictors: type, range and mean by study label",
    )
    st.caption(
        "For yes/no questions the mean is the share answering yes. Ordered categories (age, "
        "general health, education, income) are codes, so their means are only rough summaries."
    )

with tab_repeat:
    with st.container(horizontal=True):
        st.metric("Repeated full rows", f"{aud['repeated_full_rows']:,}", border=True,
                  help="Rows identical to an earlier row in all 22 columns (target included).")
        st.metric("Repeated predictor patterns", f"{aud['repeated_predictor_patterns']:,}", border=True,
                  help="Rows whose 21 predictor answers match an earlier row.")
        st.metric("Rows removed", "0", border=True)
    st.info(
        "These are **repeated answer patterns, not confirmed duplicate people.** Most questions "
        "are yes/no or short ordinal scales, so different respondents can give identical answers. "
        "The file has no respondent identifier, so there is no way to confirm a true duplicate. "
        "All rows are kept. Because identical patterns can fall into both training and test "
        "partitions, results could be slightly optimistic if some repeats are real duplicates; "
        "this is a stated limitation.",
        title="How to read these counts",
    )
