"""Project-wide settings: paths, target definition, feature sets, and study protocol.

Every script and dashboard page imports from here so that the analysis and the
app always use the same definitions.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the project folder; override the data file with an env var)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PRIMARY_FILENAME = "diabetes_012_health_indicators_BRFSS2015.csv"

#: Set DIABETES_DATA_PATH to point at the primary CSV somewhere else.
DATA_PATH = Path(os.environ.get("DIABETES_DATA_PATH", RAW_DIR / PRIMARY_FILENAME))

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
CURVES_DIR = OUTPUTS_DIR / "curves"
MODELS_DIR = PROJECT_ROOT / "models"

SET_C_MODEL_PATH = MODELS_DIR / "set_c_hgb.joblib"
SET_C_META_PATH = MODELS_DIR / "set_c_hgb.json"

#: Screening-tool models: one gradient-boosting model per feature set.
TOOL_MODELS_PATH = MODELS_DIR / "screening_tool_models.joblib"
TOOL_META_PATH = MODELS_DIR / "screening_tool_models.json"
THRESHOLD_GRID_PATH = OUTPUTS_DIR / "threshold_grid.csv"

#: Campaign settings saved from the "Tool settings" page (created on first save).
SETTINGS_DIR = PROJECT_ROOT / "settings"
TOOL_SETTINGS_PATH = SETTINGS_DIR / "tool_settings.json"

# ---------------------------------------------------------------------------
# Target definition
# ---------------------------------------------------------------------------
RAW_TARGET = "Diabetes_012"
TARGET = "screen_positive"
#: Class 0 -> negative; classes 1 (prediabetes) and 2 (diabetes) -> positive.
POSITIVE_RAW_CLASSES = (1, 2)

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
SET_A: list[str] = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]
PRIOR_DIAGNOSIS_FEATURES: list[str] = [
    "HighBP", "HighChol", "CholCheck", "Stroke", "HeartDiseaseorAttack",
]
SET_B: list[str] = [f for f in SET_A if f not in PRIOR_DIAGNOSIS_FEATURES]

assert len(SET_A) == 21 and len(SET_B) == 16

# ---------------------------------------------------------------------------
# Study protocol
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15
TEST_FRACTION = 0.15
TARGET_RECALL = 0.75          # thresholds chosen on validation to reach this
SET_C_MIN_VAL_AUC = 0.78      # Set C rule: smallest k with val ROC-AUC >= this ...
SET_C_MIN_VAL_RECALL = 0.75   # ... and val recall >= this
INNER_HOLDOUT_FRACTION = 0.20 # part of TRAINING data used to rank Set B features
BOOTSTRAP_ROUNDS = 300

# Screening tool: thresholds are expressed as "share of positive people to catch"
# (recall target) and converted to a score cut-off on VALIDATION data, per set and age group.
RECALL_TARGET_GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
                      0.75, 0.80, 0.85, 0.90, 0.95)
#: Age groups for age-specific thresholds (survey age codes, inclusive).
AGE_BANDS: dict[str, tuple[int, int]] = {
    "18–44": (1, 5),
    "45–64": (6, 9),
    "65 and older": (10, 13),
}
ALL_AGES = "All ages"
STABILITY_SEEDS = (42, 43, 44, 45, 46)

# ---------------------------------------------------------------------------
# Human-readable labels
# ---------------------------------------------------------------------------
FEATURE_LABELS: dict[str, str] = {
    "HighBP": "Told they have high blood pressure",
    "HighChol": "Told they have high cholesterol",
    "CholCheck": "Cholesterol check in past 5 years",
    "BMI": "Body mass index (kg/m²)",
    "Smoker": "Smoked at least 100 cigarettes in life",
    "Stroke": "Ever told they had a stroke",
    "HeartDiseaseorAttack": "Coronary heart disease or heart attack",
    "PhysActivity": "Physical activity in past 30 days",
    "Fruits": "Eats fruit 1+ times per day",
    "Veggies": "Eats vegetables 1+ times per day",
    "HvyAlcoholConsump": "Heavy alcohol consumption",
    "AnyHealthcare": "Has any health care coverage",
    "NoDocbcCost": "Could not see doctor because of cost",
    "GenHlth": "Self-rated general health (1–5)",
    "MentHlth": "Days of poor mental health, past 30",
    "PhysHlth": "Days of poor physical health, past 30",
    "DiffWalk": "Serious difficulty walking or climbing stairs",
    "Sex": "Sex (0 = female, 1 = male)",
    "Age": "Age group (13 categories)",
    "Education": "Education level (1–6)",
    "Income": "Household income level (1–8)",
}

FEATURE_GROUPS: dict[str, list[str]] = {
    "Demographics": ["Age", "Sex", "BMI"],
    "Lifestyle": ["Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"],
    "Self-rated health": ["GenHlth", "MentHlth", "PhysHlth", "DiffWalk"],
    "Socioeconomic": ["Education", "Income"],
    "Health care access": ["AnyHealthcare", "NoDocbcCost"],
    "Prior diagnoses (Set A only)": PRIOR_DIAGNOSIS_FEATURES,
}

AGE_LABELS: dict[int, str] = {
    1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44", 6: "45–49",
    7: "50–54", 8: "55–59", 9: "60–64", 10: "65–69", 11: "70–74",
    12: "75–79", 13: "80 or older",
}
GENHLTH_LABELS: dict[int, str] = {
    1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor",
}

MODEL_LABELS: dict[str, str] = {
    "logreg": "Logistic regression",
    "rf": "Random forest",
    "hgb": "Histogram gradient boosting",
}
SET_DESCRIPTIONS: dict[str, str] = {
    "A": "All 21 predictors, including prior diagnoses",
    "B": "16 non-invasive, self-reportable predictors",
    "C": "Minimal subset selected from Set B",
}
