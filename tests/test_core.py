"""Fast checks of definitions and helpers. Run from the project folder:

    py -3.13 -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from screening import config  # noqa: E402
from screening.data import add_binary_target, make_split  # noqa: E402
from screening.metrics import threshold_for_recall, threshold_metrics  # noqa: E402


class FeatureSets(unittest.TestCase):
    def test_sizes_and_exclusions(self):
        self.assertEqual(len(config.SET_A), 21)
        self.assertEqual(len(config.SET_B), 16)
        for f in ("HighBP", "HighChol", "CholCheck", "Stroke", "HeartDiseaseorAttack"):
            self.assertIn(f, config.SET_A)
            self.assertNotIn(f, config.SET_B)
        self.assertTrue(set(config.SET_B) < set(config.SET_A))

    def test_labels_cover_all_features(self):
        self.assertEqual(set(config.FEATURE_LABELS), set(config.SET_A))
        grouped = [f for fs in config.FEATURE_GROUPS.values() for f in fs]
        self.assertEqual(sorted(grouped), sorted(config.SET_A))


class Target(unittest.TestCase):
    def test_binarisation(self):
        df = pd.DataFrame({config.RAW_TARGET: [0.0, 1.0, 2.0, 0.0]})
        out = add_binary_target(df)
        self.assertEqual(out[config.TARGET].tolist(), [0, 1, 1, 0])


class Split(unittest.TestCase):
    def test_disjoint_stratified(self):
        y = pd.Series(np.r_[np.ones(160), np.zeros(840)].astype(int))
        s = make_split(y, 42)
        allidx = np.concatenate([s.train_idx, s.valid_idx, s.test_idx])
        self.assertEqual(len(allidx), len(np.unique(allidx)))
        self.assertEqual(len(allidx), 1000)
        self.assertEqual(len(s.train_idx), 700)
        for part in (s.train_idx, s.valid_idx, s.test_idx):
            self.assertAlmostEqual(y.iloc[part].mean(), 0.16, delta=0.01)
        s2 = make_split(y, 42)
        self.assertTrue(np.array_equal(s.test_idx, s2.test_idx))


class Thresholds(unittest.TestCase):
    def test_recall_target_met(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 2000)
        s = np.clip(y * 0.3 + rng.normal(0.4, 0.2, 2000), 0, 1)
        t = threshold_for_recall(y, s, 0.75)
        m = threshold_metrics(y, s, t)
        self.assertGreaterEqual(m["recall"], 0.75)
        # Slightly higher threshold must drop below target (highest qualifying threshold).
        higher = np.min(s[s > t])
        self.assertLess(threshold_metrics(y, s, higher)["recall"], 0.75)

    def test_confusion_counts(self):
        y = np.array([1, 1, 0, 0])
        s = np.array([0.9, 0.2, 0.8, 0.1])
        m = threshold_metrics(y, s, 0.5)
        self.assertEqual((m["tp"], m["fn"], m["fp"], m["tn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["specificity"], 0.5)


class ScreeningTool(unittest.TestCase):
    SETS = {"A": config.SET_A, "B": config.SET_B, "C": ["GenHlth", "Age", "BMI"]}

    def test_question_coverage(self):
        from screening.tool import LIFESTYLE_QUESTIONS, MEDICAL_QUESTIONS
        asked = {q.feature for q in LIFESTYLE_QUESTIONS + MEDICAL_QUESTIONS} | {"Age", "GenHlth", "BMI"}
        self.assertEqual(asked, set(config.SET_A))
        self.assertEqual({q.feature for q in MEDICAL_QUESTIONS}, set(config.PRIOR_DIAGNOSIS_FEATURES))

    def test_choose_set(self):
        from screening.tool import choose_set
        core = {"GenHlth": 3.0, "Age": 8.0, "BMI": 27.0}
        self.assertIsNone(choose_set({"Age": 8.0}, self.SETS))
        self.assertEqual(choose_set(core, self.SETS), "C")
        b = {f: 0.0 for f in config.SET_B} | core
        self.assertEqual(choose_set(b, self.SETS), "B")
        a = {f: 0.0 for f in config.SET_A} | core
        self.assertEqual(choose_set(a, self.SETS), "A")
        b_partial = dict(b)
        b_partial["Income"] = None
        self.assertEqual(choose_set(b_partial | {f: 1.0 for f in config.PRIOR_DIAGNOSIS_FEATURES}, self.SETS), "C")

    def test_alert_levels(self):
        from screening.tool import ALERT, NO_ALERT, PRIORITY, alert_level
        self.assertEqual(alert_level(0.3, 0.5, 0.8), NO_ALERT)
        self.assertEqual(alert_level(0.6, 0.5, 0.8), ALERT)
        self.assertEqual(alert_level(0.9, 0.5, 0.8), PRIORITY)
        self.assertEqual(alert_level(0.9, 0.5, None), ALERT)
        self.assertEqual(alert_level(0.9, 0.5, 0.4), ALERT)  # priority must be stricter

    def test_age_bands_cover_all_codes(self):
        from screening.thresholds import age_band_of
        self.assertEqual({age_band_of(a) for a in range(1, 14)}, set(config.AGE_BANDS))

    def test_settings_roundtrip(self):
        import tempfile
        from screening.tool import load_settings, save_settings
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.json"
            self.assertEqual(load_settings(p)["alert_rule"], "recall_0.75")
            s = load_settings(p)
            s["by_age"] = True
            s["age_rules"]["65 and older"] = "recall_0.90"
            save_settings(s, p)
            self.assertEqual(load_settings(p)["age_rules"]["65 and older"], "recall_0.90")


if __name__ == "__main__":
    unittest.main()
