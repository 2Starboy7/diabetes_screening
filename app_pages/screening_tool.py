from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from screening import config
from screening.dashboard import SET_COLORS, require_tool_assets, tool_settings
from screening.thresholds import age_band_of, lookup
from screening.tool import (
    ALERT,
    LIFESTYLE_QUESTIONS,
    MEDICAL_QUESTIONS,
    NO_ALERT,
    PRIORITY,
    SHORT_NAMES,
    Question,
    alert_level,
    alert_rule_for,
    answered,
    bmi_from,
    choose_set,
    clip_bmi_for_model,
    explain,
    rule_label,
    score_one,
)

assets = require_tool_assets()
settings = tool_settings()
set_features: dict[str, list[str]] = assets.meta["features"]
core = set_features["C"]

if "screen_log" not in st.session_state:
    st.session_state.screen_log = []


def clear_answers() -> None:
    for k in [k for k in st.session_state if str(k).startswith("q_")]:
        del st.session_state[k]


st.caption(
    "Enter answers as you talk with the person. The tool picks the most complete model the "
    "answers allow (Set A, B or C) and raises an alert when the score reaches the threshold "
    "chosen on *Tool settings*. It supports screening only; it does not diagnose diabetes."
)
with st.container(horizontal=True, vertical_alignment="center"):
    if settings["by_age"]:
        st.badge("Alert threshold: set by age group", icon=":material/tune:", color="blue")
    else:
        st.badge(f"Alert threshold: {rule_label(settings['alert_rule'])}", icon=":material/tune:", color="blue")
    if settings["priority_rule"] != "off":
        st.badge(f"Priority level: {rule_label(settings['priority_rule'])}", icon=":material/priority_high:",
                 color="orange")
    if settings.get("campaign_name"):
        st.badge(settings["campaign_name"], icon=":material/campaign:", color="gray")
    st.page_link("app_pages/tool_settings.py", label="Change settings", icon=":material/settings:")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def ask(q: Question):
    """One question. Every question starts blank (None = not asked)."""
    key = f"q_{q.feature}"
    if q.options is None:
        v = st.number_input(q.text, min_value=0, max_value=30, value=None, step=1, key=key,
                            help=q.help, placeholder="0–30 days")
        return None if v is None else float(v)
    if len(q.options) <= 2:
        v = st.segmented_control(q.text, list(q.options), format_func=q.options.get, key=key, help=q.help)
    else:
        v = st.selectbox(q.text, list(q.options), index=None, format_func=q.options.get, key=key,
                         help=q.help, placeholder="Not asked")
    return None if v is None else float(v)


answers: dict[str, float | None] = {}
left, right = st.columns([3, 2], gap="large")

with left:
    with st.container(border=True):
        st.markdown("**1 · Core questions** — required (these three make up Set C)")
        age = st.selectbox("Age group (years)", list(config.AGE_LABELS), index=None,
                           format_func=config.AGE_LABELS.get, key="q_Age", placeholder="Choose age group")
        answers["Age"] = None if age is None else float(age)
        gh = st.segmented_control("In general, how would they rate their health?", list(config.GENHLTH_LABELS),
                                  format_func=config.GENHLTH_LABELS.get, key="q_GenHlth")
        answers["GenHlth"] = None if gh is None else float(gh)
        how = st.segmented_control("BMI entry", ["Height and weight", "BMI known"], default="Height and weight",
                                   key="q_bmi_mode")
        if how == "BMI known":
            bmi = st.number_input("Body mass index (kg/m²)", min_value=12.0, max_value=98.0, value=None,
                                  step=0.5, key="q_BMI_direct", placeholder="e.g. 27")
        else:
            c1, c2 = st.columns(2)
            h = c1.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=None, step=0.5,
                                format="%.1f", key="q_height", placeholder="e.g. 165")
            w = c2.number_input("Weight (kg)", min_value=25.0, max_value=300.0, value=None, step=0.5,
                                format="%.1f", key="q_weight", placeholder="e.g. 70")
            bmi = None if (h is None or w is None) else bmi_from(h, w)
            if bmi is not None:
                st.caption(f"BMI = {bmi:.1f} kg/m² (the model uses whole numbers: {clip_bmi_for_model(bmi)})")
        answers["BMI"] = None if bmi is None else float(clip_bmi_for_model(bmi))

    life_feats = [q.feature for q in LIFESTYLE_QUESTIONS]
    with st.container(border=True):
        st.markdown(f"**2 · Lifestyle and wellbeing** — optional, all {len(life_feats)} needed for Set B")
        for q in LIFESTYLE_QUESTIONS:
            answers[q.feature] = ask(q)
        st.caption(f"{answered(answers, life_feats)} of {len(life_feats)} answered.")

    med_feats = [q.feature for q in MEDICAL_QUESTIONS]
    with st.container(border=True):
        st.markdown(f"**3 · Medical history** — optional, all {len(med_feats)} needed for Set A")
        st.caption("Only if the person already knows these from earlier contact with a health professional.")
        for q in MEDICAL_QUESTIONS:
            answers[q.feature] = ask(q)
        st.caption(f"{answered(answers, med_feats)} of {len(med_feats)} answered.")

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
fs = choose_set(answers, set_features)


def render_result() -> None:
    if fs is None:
        missing = [SHORT_NAMES[f] for f in core if answers.get(f) is None]
        st.info("Answer the core questions to see a result.\n\nStill needed: " + ", ".join(missing),
                icon=":material/edit_note:")
        return

    feats = set_features[fs]
    model = assets.models[fs]
    full = {f: answers[f] for f in feats}
    score = score_one(model, feats, full)
    rule, band = alert_rule_for(settings, age_band_of(int(answers["Age"])))
    alert_row = lookup(assets.grid, fs, band, rule)
    thr = float(alert_row["threshold"])
    prio_thr = None
    if settings["priority_rule"] != "off":
        prio_thr = float(lookup(assets.grid, fs, band, settings["priority_rule"])["threshold"])
    group = "all ages" if band == config.ALL_AGES else f"age group {band}"
    level = alert_level(score, thr, prio_thr)

    with st.container(border=True):
        st.markdown(f"**Model used: Set {fs}** · {len(feats)} questions")
        if fs != "A":
            nxt = "A" if fs == "B" else "B"
            n_missing = len([f for f in set_features[nxt] if answers.get(f) is None])
            st.caption(f"Answer {n_missing} more question(s) to use the Set {nxt} model. "
                       f"Test ROC-AUC: A {assets.meta['test_roc_auc']['A']:.3f}, "
                       f"B {assets.meta['test_roc_auc']['B']:.3f}, C {assets.meta['test_roc_auc']['C']:.3f}.")

        if level == PRIORITY:
            st.error("**Priority alert.** Score is in the highest group. Refer for a confirmatory blood "
                     "test with a health professional, following your campaign's protocol.",
                     icon=":material/priority_high:")
        elif level == ALERT:
            st.warning("**Alert.** Score reached the alert threshold. Refer for a confirmatory blood test "
                       "with a health professional, following your campaign's protocol.",
                       icon=":material/notifications_active:")
        else:
            st.success("**No alert.** Score is below the alert threshold. This does not rule out "
                       "diabetes; anyone with symptoms or concerns should still see a health professional.",
                       icon=":material/check_circle:")

        with st.container(horizontal=True):
            st.metric("Model score", f"{score:.3f}", border=True,
                      help="Uncalibrated 0–1 score from the model. Not a probability of having diabetes.")
            st.metric("Alert threshold", f"{thr:.3f}", border=True,
                      help=f"{rule_label(rule)}, chosen on validation data ({group}).")
            if prio_thr is not None and prio_thr > thr:
                st.metric("Priority threshold", f"{prio_thr:.3f}", border=True)

        bar_df = pd.DataFrame({"a": [0.0], "x": [score]})
        marks = pd.DataFrame({"x": [thr] + ([prio_thr] if prio_thr and prio_thr > thr else []),
                              "Line": ["Alert"] + (["Priority"] if prio_thr and prio_thr > thr else [])})
        base = alt.Chart(pd.DataFrame({"a": [0], "b": [1]})).mark_bar(color="#DDDDDD", height=18).encode(
            x=alt.X("a:Q", scale=alt.Scale(domain=[0, 1]), title="Model score (0–1)"), x2="b:Q")
        fill = alt.Chart(bar_df).mark_bar(color=SET_COLORS[fs], height=18).encode(x="a:Q", x2="x:Q")
        rules = alt.Chart(marks).mark_tick(thickness=3, size=34).encode(
            x="x:Q",
            color=alt.Color("Line:N", scale=alt.Scale(domain=["Alert", "Priority"], range=["#E69F00", "#D55E00"]),
                            legend=alt.Legend(orient="bottom", title="Threshold")),
            tooltip=["Line", alt.Tooltip("x:Q", format=".3f", title="Threshold")])
        st.altair_chart((base + fill + rules).properties(height=90),
                        alt=f"Score {score:.2f} shown against alert threshold {thr:.2f}")

        st.markdown(
            f"**What this setting means ({group})**, on {int(alert_row['valid_n']):,} validation survey "
            f"records: it caught **{alert_row['valid_recall']:.0%}** of people with prediabetes or diabetes, "
            f"and **{alert_row['valid_precision']:.0%}** of the people it alerted actually had prediabetes "
            f"or diabetes. It alerted **{alert_row['valid_flagged_share']:.0%}** of everyone screened."
        )

    with st.container(border=True):
        st.markdown("**Which answers moved the score**")
        ex = explain(model, feats, full, assets.meta["typical_answers"])
        ex = ex[ex["effect"].abs() >= 0.005].head(8)
        if ex.empty:
            st.caption("All answers are close to the typical survey answer.")
        else:
            ex["Direction"] = ex["effect"].map(lambda e: "Raised the score" if e > 0 else "Lowered the score")
            ch = alt.Chart(ex).mark_bar().encode(
                y=alt.Y("answer:N", sort=None, title=None, axis=alt.Axis(labelLimit=170)),
                x=alt.X("effect:Q", title="Score change"),
                color=alt.Color("Direction:N", scale=alt.Scale(domain=["Raised the score", "Lowered the score"],
                                                               range=["#D55E00", "#0072B2"]),
                                legend=alt.Legend(orient="bottom", title=None, direction="vertical")),
                tooltip=["answer", alt.Tooltip("typical:N", title="Typical answer"),
                         alt.Tooltip("effect:Q", format="+.3f", title="Change")],
            ).properties(height=alt.Step(28))
            st.altair_chart(ch, alt="Answers that raised or lowered this person's score the most")
            raisers = ex[ex["effect"] > 0]["answer"].tolist()[:3]
            if raisers:
                st.markdown("Answers linked to a higher score: " + "; ".join(f"**{r}**" for r in raisers) + ".")
        st.caption("Each bar compares the answer given with the typical survey answer. These are patterns "
                   "the model learned from the survey, not proven causes of diabetes.")

    with st.container(border=True):
        st.markdown("**Save and continue**")
        ref = st.text_input("Reference code (optional)", key="q_ref", max_chars=30,
                            placeholder="e.g. camp-12 · do not enter names",
                            help="Stored only in this browser session. Avoid names or phone numbers.")
        with st.container(horizontal=True):
            if st.button("Save to session log", icon=":material/save:", type="primary", key="save_log"):
                st.session_state.screen_log.append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "reference": ref,
                    "age_group": config.AGE_LABELS[int(answers["Age"])],
                    "model_set": fs,
                    "questions_answered": answered(answers, config.SET_A),
                    "score": round(score, 4),
                    "alert_threshold": round(thr, 4),
                    "threshold_rule": rule_label(rule),
                    "result": level,
                })
                st.toast("Saved to the session log.", icon=":material/check:")
            st.button("Clear for next person", icon=":material/restart_alt:", on_click=clear_answers, key="clear_next")


with right:
    render_result()

log =st.session_state.screen_log
if log:
    st.subheader("Session log", anchor=False)
    log_df = pd.DataFrame(log)
    with st.container(horizontal=True):
        st.metric("Screened", len(log_df), border=True)
        st.metric("Alerts", int((log_df["result"] != NO_ALERT).sum()), border=True)
        st.metric("Priority alerts", int((log_df["result"] == PRIORITY).sum()), border=True)
    st.dataframe(log_df, hide_index=True, alt="People screened in this session and their results")
    with st.container(horizontal=True):
        st.download_button("Download log (CSV)", log_df.to_csv(index=False),
                           file_name=f"screening_log_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv",
                           icon=":material/download:")
        if st.button("Clear log", icon=":material/delete:"):
            st.session_state.screen_log = []
            st.rerun()
    st.caption("The log lives only in this browser tab. Download it before closing the page.")
