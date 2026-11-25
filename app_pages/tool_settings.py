import json

import altair as alt
import pandas as pd
import streamlit as st

from screening import config
from screening.dashboard import require_tool_assets, tool_settings
from screening.thresholds import lookup
from screening.tool import DEFAULT_SETTINGS, PRESETS, alert_rule_for, recall_rule, rule_label, save_settings

assets = require_tool_assets()
grid = assets.grid
settings = tool_settings()

st.markdown(
    "A **threshold** decides who gets an alert. Instead of a raw score, you choose **how many people "
    "with prediabetes or diabetes the tool should catch**. The matching score cut-off was worked out "
    "on validation survey data, separately for each model (Set A, B, C) and each age group. Catching "
    "more people always means more false alarms."
)

# ---------------------------------------------------------------------------
st.subheader("What the options mean", anchor=False)
view_set = st.segmented_control("Show numbers for", ["C", "B", "A"], default="C", key="ts_view_set",
                                format_func=lambda s: f"Set {s} model")
view_set = view_set or "C"
rows = []
for p in PRESETS:
    r = lookup(grid, view_set, config.ALL_AGES, p.rule)
    rows.append({
        "Option": p.name,
        "What it usually means": p.meaning,
        "Catches": f"{r['valid_recall']:.0%}",
        "Alerted who are positive": f"{r['valid_precision']:.0%}",
        "Negatives not alerted": f"{r['valid_specificity']:.0%}",
        "Alerts per 100 screened": f"{r['valid_flagged_share'] * 100:.0f}",
        "Score cut-off": f"{r['threshold']:.3f}",
    })
st.table(pd.DataFrame(rows).set_index("Option"),
         alt="Threshold options with their recall, precision, specificity and alert rate")
st.caption(
    f"Validation data, all ages, Set {view_set} model. **Catches** = share of people with prediabetes or "
    "diabetes who get an alert (recall). **Alerted who are positive** = precision. **Negatives not "
    "alerted** = specificity. About 16 in every 100 survey respondents had prediabetes or diabetes, so "
    "precision cannot be very high for any option."
)

trade = grid[(grid["feature_set"] == view_set) & (grid["age_band"] == config.ALL_AGES)
             & grid["rule"].str.startswith("recall_")].copy()
trade_long = trade.melt(id_vars=["recall_target"], value_vars=["valid_flagged_share", "valid_precision"],
                        var_name="Measure", value_name="Value")
trade_long["Measure"] = trade_long["Measure"].map({
    "valid_flagged_share": "Share of everyone alerted", "valid_precision": "Alerted who are positive"})
ch = alt.Chart(trade_long).mark_line(point=True).encode(
    x=alt.X("recall_target:Q", title="Share of positive people caught (recall target)", axis=alt.Axis(format="%")),
    y=alt.Y("Value:Q", title="Share", axis=alt.Axis(format="%")),
    color=alt.Color("Measure:N", scale=alt.Scale(range=["#D55E00", "#0072B2"]), legend=alt.Legend(orient="bottom", title=None)),
    tooltip=[alt.Tooltip("recall_target:Q", format=".0%", title="Catch"), "Measure", alt.Tooltip("Value:Q", format=".1%")],
).properties(title="Catching more people means alerting many more people", height=300)
st.altair_chart(ch, alt="As the recall target rises, the share alerted rises steeply and precision falls")

# ---------------------------------------------------------------------------
st.subheader("Choose the settings", anchor=False)


def rule_picker(label: str, current: str, key: str, allow_off: bool = False) -> str:
    options = (["off"] if allow_off else []) + [p.rule for p in PRESETS] + ["custom"]
    names = {"off": "Off", "custom": "Custom…", **{p.rule: p.name for p in PRESETS}}
    start = current if current in options else "custom"
    choice = st.selectbox(label, options, index=options.index(start), format_func=names.get, key=key)
    if choice != "custom":
        return choice
    default = float(current.split("_")[1]) if current.startswith("recall_") else 0.75
    pct = st.select_slider(f"{label}: share of positive people to catch",
                           options=list(config.RECALL_TARGET_GRID), value=default,
                           format_func=lambda v: f"{v:.0%}", key=f"{key}_custom")
    return recall_rule(pct)


new = json.loads(json.dumps(settings))
with st.container(border=True):
    new["campaign_name"] = st.text_input("Campaign or site name (optional)", value=settings["campaign_name"],
                                         key="ts_campaign", placeholder="e.g. Ward 5 health camp, March")
    mode = st.segmented_control("Alert threshold", ["Same for all ages", "Different by age group"],
                                default="Different by age group" if settings["by_age"] else "Same for all ages",
                                key="ts_mode")
    new["by_age"] = mode == "Different by age group"
    if new["by_age"]:
        cols = st.columns(len(config.AGE_BANDS))
        for col, band in zip(cols, config.AGE_BANDS):
            with col:
                new["age_rules"][band] = rule_picker(f"Age {band}", settings["age_rules"][band], f"ts_band_{band}")
        st.caption("Each age group gets its own cut-off, chosen on validation records from that age group.")
    else:
        new["alert_rule"] = rule_picker("Alert level", settings["alert_rule"], "ts_alert")
    new["priority_rule"] = rule_picker("Priority alert level (stricter, optional)", settings["priority_rule"],
                                       "ts_priority", allow_off=True)
    st.caption("A priority alert is shown only when its cut-off is higher than the alert cut-off.")

st.session_state.tool_settings = new  # applies straight away in this browser session

# Preview of resulting cut-offs ------------------------------------------------
prev = []
for band in (config.AGE_BANDS if new["by_age"] else [config.ALL_AGES]):
    rule, grid_band = alert_rule_for(new, band) if new["by_age"] else (new["alert_rule"], config.ALL_AGES)
    for fs in ("A", "B", "C"):
        r = lookup(grid, fs, grid_band, rule)
        prev.append({
            "Age group": band, "Model": f"Set {fs}", "Rule": rule_label(rule),
            "Score cut-off": r["threshold"],
            "Catches (validation)": r["valid_recall"] * 100, "Catches (test)": r["test_recall"] * 100,
            "Alerted who are positive (test)": r["test_precision"] * 100,
            "Alerts per 100 (test)": r["test_flagged_share"] * 100,
            "Positives in validation group": int(r["valid_positives"]),
        })
with st.expander("Resulting cut-offs", expanded=True):
    if not new["by_age"]:
        st.caption("One cut-off for everyone. Because younger people score lower, a single cut-off "
                   "catches far fewer positives under 45 than over 65 (Set C, 75% setting, test data: "
                   "about 37% vs. 83%). Choose *Different by age group* to give each age group its own "
                   "catch rate.")
    pct = st.column_config.NumberColumn(format="%.0f%%")
    st.dataframe(
        pd.DataFrame(prev), hide_index=True,
        column_config={
            "Score cut-off": st.column_config.NumberColumn(format="%.3f"),
            "Catches (validation)": pct,
            "Catches (test)": pct,
            "Alerted who are positive (test)": pct,
            "Alerts per 100 (test)": st.column_config.NumberColumn(format="%.0f"),
        },
        alt="Score cut-offs that the current settings produce, with validation and test performance",
    )
    st.caption("Cut-offs come from validation data; test columns show how they held up on unseen survey "
               "records. Younger groups have fewer positive records, so their cut-offs are less certain.")


def _reset() -> None:
    st.session_state.tool_settings = json.loads(json.dumps(DEFAULT_SETTINGS))
    for k in [k for k in st.session_state if str(k).startswith("ts_") and k != "ts_view_set"]:
        del st.session_state[k]


with st.container(horizontal=True):
    if st.button("Save as default for this computer", type="primary", icon=":material/save:"):
        save_settings(new)
        st.toast(f"Saved to {config.TOOL_SETTINGS_PATH.relative_to(config.PROJECT_ROOT)}", icon=":material/check:")
    st.button("Reset to study defaults", icon=":material/restart_alt:", on_click=_reset)
st.caption("Changes apply immediately in this browser tab. Saving makes them the starting settings "
           "next time the app opens on this computer.")
