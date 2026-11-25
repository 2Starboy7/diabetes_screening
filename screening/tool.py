"""Screening-tool logic (no Streamlit code): questions, automatic choice of
feature set, alert levels, score explanations, and saved settings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from screening import config

# ---------------------------------------------------------------------------
# Questions, worded as in the BRFSS 2015 indicator definitions
# ---------------------------------------------------------------------------
YES_NO = {0: "No", 1: "Yes"}


@dataclass(frozen=True)
class Question:
    feature: str
    text: str
    options: dict[int, str] | None = None  # None = numeric entry
    help: str | None = None


LIFESTYLE_QUESTIONS: list[Question] = [
    Question("Sex", "Sex", {0: "Female", 1: "Male"}),
    Question("PhysActivity", "Any physical activity or exercise in the past 30 days, not counting their job?", YES_NO),
    Question("Fruits", "Eats fruit at least once a day?", YES_NO),
    Question("Veggies", "Eats vegetables at least once a day?", YES_NO),
    Question("Smoker", "Has smoked at least 100 cigarettes in their whole life?", YES_NO,
             help="About 5 packs. Counts former smokers too."),
    Question("HvyAlcoholConsump", "Heavy drinker?", YES_NO,
             help="Survey definition: more than 14 drinks a week for men, more than 7 for women."),
    Question("DiffWalk", "Serious difficulty walking or climbing stairs?", YES_NO),
    Question("MentHlth", "In the past 30 days, on how many days was their mental health not good?", None,
             help="0–30 days. Includes stress, depression and problems with emotions."),
    Question("PhysHlth", "In the past 30 days, on how many days was their physical health not good?", None,
             help="0–30 days. Includes physical illness and injury."),
    Question("AnyHealthcare", "Has any kind of health insurance or health-care coverage?", YES_NO),
    Question("NoDocbcCost", "In the past 12 months, needed a doctor but could not go because of cost?", YES_NO),
    Question("Education", "Highest level of school completed", {
        1: "Never attended / kindergarten only", 2: "Grades 1–8", 3: "Grades 9–11",
        4: "Grade 12 or equivalent", 5: "College 1–3 years", 6: "College 4+ years"}),
    Question("Income", "Yearly household income (US dollars, survey brackets)", {
        1: "Under $10k", 2: "$10k–15k", 3: "$15k–20k", 4: "$20k–25k",
        5: "$25k–35k", 6: "$35k–50k", 7: "$50k–75k", 8: "$75k or more"},
        help="The survey used US-dollar brackets in 2015. Convert local income before choosing."),
]

MEDICAL_QUESTIONS: list[Question] = [
    Question("HighBP", "Ever told by a health professional that they have high blood pressure?", YES_NO),
    Question("HighChol", "Ever told by a health professional that they have high cholesterol?", YES_NO),
    Question("CholCheck", "Had a cholesterol check in the past 5 years?", YES_NO),
    Question("Stroke", "Ever told they had a stroke?", YES_NO),
    Question("HeartDiseaseorAttack", "Ever had coronary heart disease or a heart attack?", YES_NO),
]

SHORT_NAMES: dict[str, str] = {
    "Age": "Age group", "BMI": "BMI", "GenHlth": "General health", "Sex": "Sex",
    "PhysActivity": "Physical activity", "Fruits": "Daily fruit", "Veggies": "Daily vegetables",
    "Smoker": "Smoking history", "HvyAlcoholConsump": "Heavy drinking", "DiffWalk": "Difficulty walking",
    "MentHlth": "Poor mental-health days", "PhysHlth": "Poor physical-health days",
    "AnyHealthcare": "Health coverage", "NoDocbcCost": "Skipped doctor due to cost",
    "Education": "Education", "Income": "Income", "HighBP": "High blood pressure",
    "HighChol": "High cholesterol", "CholCheck": "Cholesterol check", "Stroke": "Stroke",
    "HeartDiseaseorAttack": "Heart disease / attack",
}


def describe_answer(feature: str, value: float) -> str:
    """Human-readable answer, e.g. 'Age group: 55–59'."""
    v = int(round(value))
    if feature == "Age":
        text = config.AGE_LABELS.get(v, str(v))
    elif feature == "GenHlth":
        text = config.GENHLTH_LABELS.get(v, str(v))
    elif feature == "BMI":
        text = f"{v}"
    elif feature in ("MentHlth", "PhysHlth"):
        text = f"{v} days"
    else:
        q = {q.feature: q for q in LIFESTYLE_QUESTIONS + MEDICAL_QUESTIONS}.get(feature)
        text = q.options.get(v, str(v)) if q and q.options else str(v)
    return f"{SHORT_NAMES.get(feature, feature)}: {text}"


# ---------------------------------------------------------------------------
# Automatic feature-set choice
# ---------------------------------------------------------------------------
def answered(answers: dict[str, float | None], features: list[str]) -> int:
    return sum(answers.get(f) is not None for f in features)


def choose_set(answers: dict[str, float | None], set_features: dict[str, list[str]]) -> str | None:
    """Use the most complete model whose questions are ALL answered.

    A (21 questions) if everything is answered, otherwise B (16 non-invasive
    questions), otherwise C (the minimal set). None if even Set C is incomplete.
    """
    for fs in ("A", "B", "C"):
        if all(answers.get(f) is not None for f in set_features[fs]):
            return fs
    return None


# ---------------------------------------------------------------------------
# Threshold rules ("how many positive people should the tool catch?")
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Preset:
    rule: str
    name: str
    meaning: str


PRESETS: list[Preset] = [
    Preset("recall_0.90", "Catch nearly all (90%)",
           "Very few people with prediabetes or diabetes are missed, but many people are flagged "
           "who turn out not to have it. Suits campaigns where a confirmatory test is cheap and on site."),
    Preset("recall_0.75", "Study setting (75%)",
           "The threshold used for the research results: catches about 3 in 4 people with "
           "prediabetes or diabetes, with a moderate number of false alarms."),
    Preset("balanced", "Balanced",
           "Gives equal weight to catching positives and not flagging negatives "
           "(the highest recall + specificity on validation data)."),
    Preset("recall_0.60", "Fewer false alarms (60%)",
           "Flags fewer people, so a larger share of those flagged are positive, but more "
           "positives are missed. Suits campaigns with limited follow-up capacity."),
    Preset("recall_0.40", "Highest scores only (40%)",
           "Flags only the people with the highest scores. Useful as a second, 'priority' "
           "level on top of a more sensitive alert."),
]
PRESET_BY_RULE = {p.rule: p for p in PRESETS}


def rule_label(rule: str) -> str:
    if rule in PRESET_BY_RULE:
        return PRESET_BY_RULE[rule].name
    if rule.startswith("recall_"):
        return f"Custom: catch {float(rule.split('_')[1]):.0%}"
    return rule


def recall_rule(target: float) -> str:
    return f"recall_{target:.2f}"


# ---------------------------------------------------------------------------
# Settings (saved per campaign as JSON)
# ---------------------------------------------------------------------------
DEFAULT_SETTINGS: dict = {
    "campaign_name": "",
    "alert_rule": "recall_0.75",
    "by_age": False,
    "age_rules": {band: "recall_0.75" for band in config.AGE_BANDS},
    "priority_rule": "recall_0.40",   # or "off"
}


def load_settings(path: Path = config.TOOL_SETTINGS_PATH) -> dict:
    settings = json.loads(json.dumps(DEFAULT_SETTINGS))
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            settings.update({k: v for k, v in saved.items() if k in DEFAULT_SETTINGS})
            settings["age_rules"] = {**DEFAULT_SETTINGS["age_rules"], **saved.get("age_rules", {})}
        except (json.JSONDecodeError, OSError):
            pass
    return settings


def save_settings(settings: dict, path: Path = config.TOOL_SETTINGS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")


def alert_rule_for(settings: dict, age_band: str) -> tuple[str, str]:
    """(rule, grid age band) to use for a person in `age_band`.

    Same-for-all-ages mode uses one cut-off computed on all validation records;
    by-age mode uses the cut-off computed within the person's age group.
    """
    if settings["by_age"]:
        return settings["age_rules"][age_band], age_band
    return settings["alert_rule"], config.ALL_AGES


# ---------------------------------------------------------------------------
# Scoring and explanation
# ---------------------------------------------------------------------------
NO_ALERT, ALERT, PRIORITY = "No alert", "Alert", "Priority alert"


def alert_level(score: float, alert_thr: float, priority_thr: float | None) -> str:
    if priority_thr is not None and priority_thr > alert_thr and score >= priority_thr:
        return PRIORITY
    return ALERT if score >= alert_thr else NO_ALERT


def score_one(model, features: list[str], answers: dict[str, float]) -> float:
    x = pd.DataFrame([[float(answers[f]) for f in features]], columns=features)
    return float(model.predict_proba(x)[:, 1][0])


def explain(model, features: list[str], answers: dict[str, float], typical: dict[str, float]) -> pd.DataFrame:
    """How much each answer moved the score compared with the typical (median) answer.

    For each question, the answer is swapped for the typical training answer and
    the change in score is recorded. Positive = this answer raised the score.
    This describes the model, not causes of diabetes.
    """
    base = score_one(model, features, answers)
    rows, swapped = [], []
    for f in features:
        if float(answers[f]) == float(typical[f]):
            continue
        alt = dict(answers)
        alt[f] = typical[f]
        swapped.append((f, alt))
    if swapped:
        X = pd.DataFrame([[float(a[f]) for f in features] for _, a in swapped], columns=features)
        alt_scores = model.predict_proba(X)[:, 1]
        for (f, _), s in zip(swapped, alt_scores):
            rows.append({
                "feature": f,
                "answer": describe_answer(f, answers[f]),
                "typical": describe_answer(f, typical[f]).split(": ", 1)[1],
                "effect": base - float(s),
            })
    out = pd.DataFrame(rows, columns=["feature", "answer", "typical", "effect"])
    return out.reindex(out["effect"].abs().sort_values(ascending=False).index) if len(out) else out


def bmi_from(height_cm: float, weight_kg: float) -> float:
    return weight_kg / (height_cm / 100) ** 2


def clip_bmi_for_model(bmi: float) -> int:
    """Round to a whole number and keep inside the range seen in training (12–98)."""
    return int(np.clip(round(bmi), 12, 98))
