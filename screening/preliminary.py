"""Preliminary single-split results supplied with the project brief.

Kept ONLY so the analysis can report how far the recomputed numbers differ.
The dashboard always displays the recomputed results from outputs/.
"""

PRELIMINARY_RESULTS = [
    {"feature_set": "A", "model": "logreg", "roc_auc": 0.8194, "average_precision": 0.4376, "recall": 0.7542, "specificity": 0.7318, "precision": 0.3447},
    {"feature_set": "A", "model": "rf", "roc_auc": 0.8200, "average_precision": 0.4555, "recall": 0.7557, "specificity": 0.7319, "precision": 0.3453},
    {"feature_set": "A", "model": "hgb", "roc_auc": 0.8262, "average_precision": 0.4661, "recall": 0.7534, "specificity": 0.7390, "precision": 0.3507},
    {"feature_set": "B", "model": "logreg", "roc_auc": 0.7991, "average_precision": 0.3981, "recall": 0.7465, "specificity": 0.7072, "precision": 0.3230},
    {"feature_set": "B", "model": "rf", "roc_auc": 0.7975, "average_precision": 0.4104, "recall": 0.7562, "specificity": 0.6944, "precision": 0.3164},
    {"feature_set": "B", "model": "hgb", "roc_auc": 0.8072, "average_precision": 0.4297, "recall": 0.7552, "specificity": 0.7097, "precision": 0.3273},
    {"feature_set": "C", "model": "hgb", "roc_auc": 0.7999, "average_precision": 0.4200, "recall": 0.7509, "specificity": 0.7000, "precision": 0.3189},
]

PRELIMINARY_SET_C = {
    "features": ["GenHlth", "Age", "BMI"],
    "valid_roc_auc": 0.8021,
    "valid_recall": 0.7518,
    "threshold": 0.5157,
}
