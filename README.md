# Minimal-feature diabetes screening: research prototype

A reproducible analysis and Streamlit dashboard for one question:

> How does screening performance change as we reduce the questionnaire from all predictors
> (Set A), to a non-invasive subset (Set B), to a minimal subset selected from Set B (Set C)?

**This is a research prototype for screening research. It is not a diagnostic system or a
clinical tool, has not been clinically validated, and gives no medical advice.**

---

## Host on Streamlit Community Cloud

1. Push this repo to GitHub (already set as `origin`).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and **Create app**.
3. Select this repository, branch `main`, and set **Main file path** to `app.py`.
4. Deploy. The app uses the committed models and outputs; no re-training is required.

`runtime.txt` pins Python 3.13 for the cloud runtime.

## Quick start (local, Python 3.13)

```bash
python3.13 -m pip install -r requirements.txt
python3.13 -m streamlit run app.py          # opens http://localhost:8501
```

Optional — re-run the analysis (~5 minutes; rewrites `outputs/` and `models/`):

```bash
python3.13 scripts/run_analysis.py
```

In the app, **Screening tool → Screen a person** is the field tool, and **Screening tool → Tool
settings** is where you set the alert thresholds. The **Research** menu holds the study pages.

Precomputed outputs and models are in the repo, so you can launch the dashboard straight away.
Re-run the analysis whenever the data or code changes.

Run the unit tests:

```bash
python3.13 -m unittest discover -s tests -v
```

### Data placement

`data/raw/diabetes_012_health_indicators_BRFSS2015.csv` is committed so the hosted app can load
the Explore page. To point at a copy elsewhere, set `DIABETES_DATA_PATH` to the full file path
before running anything. The analysis never modifies the file. It records the file's SHA-256
hash, and the dashboard warns you if the file on disk no longer matches the saved results.

---

## Study design (as implemented)

| Item | Setting |
|---|---|
| Data | `diabetes_012_health_indicators_BRFSS2015.csv`, 253,680 rows, 21 predictors |
| Target | `Diabetes_012` = 0 is negative; 1 (prediabetes) or 2 (diabetes) is positive. 39,977 positives (15.76%) |
| Cleaning | None needed: 0 missing cells. **No rows removed. BMI not clipped.** |
| Split | Stratified 70 / 15 / 15 train / validation / test, seed 42 |
| Imbalance | Class weighting only. No SMOTE or other resampling |
| Models | Logistic regression (with standard scaling), random forest, histogram gradient boosting (scikit-learn) |
| Threshold | Highest score with validation recall ≥ 75%. The test set is never used to choose it |
| Set A | All 21 predictors |
| Set B | Set A without `HighBP`, `HighChol`, `CholCheck`, `Stroke`, `HeartDiseaseorAttack` (16) |
| Set C | Chosen from Set B using training data only (see below) |

**How Set C is selected**

1. Rank the 16 Set B questions by greedy forward selection. At each step, add the question that
   most improves ROC-AUC of a gradient-boosting model. Fit on 80% of the **training** partition
   and score on the other 20%.
2. For k = 1…16, train gradient boosting on the top-k questions using the full training
   partition, then set its threshold on **validation**.
3. Pick the smallest k with validation ROC-AUC ≥ 0.78 and validation recall ≥ 75%.

Result: **k = 3: `GenHlth`, `Age`, `BMI`**. This matches the preliminary selection.

**Repeated patterns.** 23,899 rows repeat an earlier row across all 22 columns, and 25,772 repeat
an earlier row's 21 predictor answers. These are repeated answer patterns, not confirmed duplicate
people: the file has no respondent ID. They are counted and reported but **not deleted**.

---

## Results (recomputed, held-out test set, n = 38,052)

| Set | Model | Questions | ROC-AUC (95% CI) | Average precision | Recall | Specificity | Precision | Validation threshold |
|---|---|---:|---|---:|---:|---:|---:|---:|
| A | Logistic regression | 21 | 0.8175 (0.8125–0.8226) | 0.4290 | 0.7519 | 0.7336 | 0.3455 | 0.5099 |
| A | Random forest | 21 | 0.8205 (0.8153–0.8261) | 0.4524 | 0.7564 | 0.7338 | 0.3471 | 0.4155 |
| A | Histogram gradient boosting | 21 | 0.8261 (0.8213–0.8315) | 0.4588 | 0.7547 | 0.7420 | 0.3537 | 0.5369 |
| B | Logistic regression | 16 | 0.7967 (0.7914–0.8025) | 0.3894 | 0.7397 | 0.7126 | 0.3250 | 0.4918 |
| B | Random forest | 16 | 0.7972 (0.7914–0.8036) | 0.4071 | 0.7497 | 0.6995 | 0.3182 | 0.3933 |
| B | Histogram gradient boosting | 16 | 0.8068 (0.8017–0.8125) | 0.4237 | 0.7472 | 0.7162 | 0.3300 | 0.5270 |
| C | Logistic regression | 3 | 0.7916 (0.7861–0.7980) | 0.3812 | 0.7432 | 0.6998 | 0.3165 | 0.4858 |
| C | Random forest | 3 | 0.7923 (0.7871–0.7984) | 0.3970 | 0.7445 | 0.6965 | 0.3145 | 0.5048 |
| C | Histogram gradient boosting | 3 | 0.7993 (0.7943–0.8053) | 0.4115 | 0.7445 | 0.7042 | 0.3201 | 0.5256 |

Set C validation results (gradient boosting): ROC-AUC 0.7992, recall 0.7507, threshold 0.5256.

The 95% intervals are bootstrap intervals over the test rows with the model held fixed (300
resamples). They show test-sample variation only. Set C was selected with gradient boosting,
so the logistic-regression and random-forest rows for Set C are for comparison only.

Average precision is scikit-learn's step-wise `average_precision_score`. The trapezoidal PR-AUC
is a different calculation. It appears separately as `test_pr_auc_trapezoid` in
`outputs/model_results.csv`.

**Repeated splits.** Gradient boosting was re-run on five different splits (seeds 42–46) with
thresholds re-chosen each time. Set C questions were held fixed. Mean ± SD test ROC-AUC:
A 0.8254 ± 0.0019, B 0.8065 ± 0.0023, C 0.8000 ± 0.0017.

**Compared with the preliminary numbers.** Every metric is within ±0.009 of the preliminary
single-split value. Average precision is 0.003–0.009 lower in this run. Recall is up to 0.008
lower and specificity 0.002–0.007 higher, because thresholds are not re-tuned on test. The
preliminary run's exact model settings (for example forest depth or boosting iterations) were
not recorded, so exact agreement is not expected. Per-metric differences are in
`outputs/preliminary_vs_recomputed.csv`. **Report the recomputed values.**

### What these results do and do not establish

- They show that on one BRFSS 2015 split, dropping the five prior-diagnosis questions lowers
  test ROC-AUC by about 0.02. Shrinking further to three questions (general health, age, BMI)
  lowers it by about 0.01 more. At a fixed ≥75% recall target, specificity drops from about 74%
  to about 70%.
- They do **not** show performance in other populations or survey years, calibration (scores
  are not probabilities), clinical usefulness, or cost-effectiveness. Repeated answer patterns
  may appear in both training and test data, which may make results slightly optimistic.

---

## Screening tool (field use)

Two pages under **Screening tool** in the top menu turn the models into a reusable tool for
health camps and outreach. A volunteer can fill it in while talking with someone.

**Screen a person**

1. **Core questions (required).** Age group, self-rated general health, and BMI. BMI can be
   entered directly or worked out from height and weight.
2. **Lifestyle and wellbeing (optional).** Physical activity outside work, daily fruit and
   vegetables, smoking, heavy drinking, difficulty walking, poor mental and physical health days,
   health coverage, cost barriers, education, and income.
3. **Medical history (optional).** Known high blood pressure, high cholesterol, cholesterol check,
   stroke, and heart disease.

Every question starts blank. The tool **automatically uses the most complete model the answers
allow**:

- **Set A** when all 21 questions are answered.
- **Set B** when all 16 non-invasive questions are answered.
- **Set C** otherwise.

It then shows one of three results:

- **No alert:** the score is below the alert threshold.
- **Alert:** the score reached the alert threshold.
- **Priority alert:** the score also reached a stricter priority threshold (optional).

The result card also shows:

- what the threshold means for the person's age group;
- a chart of **which answers raised or lowered the score** compared with the typical survey answer.

Results can be saved to a session log with an optional reference code (no names) and downloaded as CSV.

**Tool settings**

- Thresholds are chosen as **"how many people with prediabetes or diabetes should the tool
  catch?"** rather than as raw scores. Options:

  | Option | Catches | What it means |
  |---|---|---|
  | Catch nearly all | 90% | Few cases missed, many false alarms |
  | Study setting | 75% | The setting used for the research results (default) |
  | Balanced | varies | Best combined recall and specificity (Youden's J) |
  | Fewer false alarms | 60% | For campaigns with limited follow-up capacity |
  | Highest scores only | 40% | Default priority level |
  | Custom | 30–95% | Any value, in steps of 5% |

- Each option is turned into a score cut-off **on validation data**, separately for each model and
  each age group (18–44, 45–64, 65+). Test-set results at the same cut-offs are shown next to them.
- Thresholds can be the **same for all ages** or **different for each age group**. This choice
  matters. With one shared cut-off (Set C, 75% setting), the test set showed about 37% of positive
  people aged 18–44 were alerted, against 83% of those aged 65+. Age-specific cut-offs give each
  group its own catch rate, at the cost of more alerts among younger people.
- **Save as default for this computer** writes `settings/tool_settings.json`.

The cut-offs come from `outputs/threshold_grid.csv`, which `scripts/run_analysis.py` writes.

**Limits to state when using it.** These limits apply to the tool:

- The models learned from US survey answers (BRFSS 2015). They have not been validated for other
  countries or for clinical use.
- Income brackets are in 2015 US dollars.
- "Work nature" (job activity) is not a question in this dataset, so it cannot be used. The
  physical-activity question excludes activity at work.
- An alert means "refer for a confirmatory blood test, per your campaign protocol". It is not a
  diagnosis, and "no alert" does not rule diabetes out.

## Dashboard (research pages)

`py -3.13 -m streamlit run app.py` also opens four research pages:

1. **Overview**: research question, target definition, dataset size, class balance, the three sets, and the headline result.
2. **Explore the data**: missingness, positive share by age and general health, BMI distribution, predictor summaries, and the repeated-pattern caveat.
3. **Compare models**: results table with confidence intervals, metric charts, ROC and precision–recall curves, confusion counts, validation thresholds, the Set C burden curve, repeated-split stability, and the comparison with preliminary numbers.
4. **Try the three-question model**: inputs for Age, BMI and General Health. Shows the **uncalibrated model score** next to the validation **research threshold**, with a visible not-a-diagnosis note.

The dashboard never trains models. It reads `outputs/` and `models/`, so it always uses the
same target, feature sets, split and threshold as the reported results. Data and models are
cached, and outputs reload automatically after you re-run the analysis.

---

## Folder layout

```
diabetes-screening-prototype/
├── app.py                    Streamlit entry point (navigation + shared header)
├── app_pages/                One file per dashboard page
├── screening/                Reusable package
│   ├── config.py             Paths, target, feature sets, protocol constants, labels
│   ├── data.py               Loading, target construction, audit, 70/15/15 split
│   ├── metrics.py            Threshold selection, screening metrics, bootstrap intervals
│   ├── modeling.py           Model definitions, training, Set C ranking and sweep
│   ├── figures.py            Static PNG figures (matplotlib)
│   ├── thresholds.py         Validation-based threshold grid by set and age group
│   ├── tool.py               Screening-tool questions, set choice, alerts, explanations, settings
│   ├── dashboard.py          Cached loaders and helpers for the app
│   └── preliminary.py        Preliminary numbers from the brief (for comparison only)
├── scripts/run_analysis.py   End-to-end reproducible analysis
├── tests/test_core.py        Unit tests (unittest)
├── data/raw/                 Input CSV, unmodified (see data/README.md)
├── outputs/                  Generated results
│   ├── data_audit.json            Counts, missingness, repeated patterns, BMI summary
│   ├── model_results.csv          All validation + test metrics, CIs, thresholds
│   ├── set_c_forward_ranking.csv  Forward-selection order (training data)
│   ├── set_c_validation_sweep.csv k = 1..16 validation metrics and the rule
│   ├── stability_hgb.csv          Repeated-split results
│   ├── preliminary_vs_recomputed.csv
│   ├── threshold_grid.csv         Cut-offs per set × age group × option (validation) + test results
│   ├── run_metadata.json          Versions, seed, split sizes, data hash, method notes
│   ├── analysis_log.txt           Console log of the run
│   ├── curves/                    ROC / PR points and Set C score histogram (test)
│   └── figures/                   PNG figures for reports
├── models/
│   ├── set_c_hgb.joblib      Set C gradient-boosting model used by the demo
│   ├── set_c_hgb.json        Its features, threshold, metrics, data hash, versions
│   ├── screening_tool_models.joblib  Gradient-boosting models for Sets A, B, C (screening tool)
│   └── screening_tool_models.json    Their features, typical answers, test ROC-AUC, versions
├── settings/tool_settings.json  Saved campaign thresholds (created when you click Save)
├── .streamlit/config.toml    Theme (colour-blind-safe chart palette)
├── requirements.txt
└── .gitignore
```

## Differences from the CIA-1 draft methodology

The draft paper (CIA-1) proposed some steps that this implementation deliberately does differently,
following the project brief. If the paper describes this work, update it to match:

| Draft proposed | Implemented | Why |
|---|---|---|
| Remove duplicate records before splitting | Repeated patterns counted, **not removed** | No respondent ID; identical answers can come from different people |
| Winsorize BMI to 1st–99th percentile | **No clipping** | No evidence that extreme values are errors |
| 80/20 split; threshold tuned on training set | 70/15/15 split; threshold tuned on **validation** | Keeps threshold choice independent of both training fit and test |
| XGBoost; SHAP + RFE for Set C | scikit-learn histogram gradient boosting; greedy forward selection on training data | No extra dependencies; selection uses training data only |
| PR-AUC | Average precision (plus trapezoidal PR-AUC, separately labelled) | Avoids mixing up two different calculations |
| Decision curve analysis | Not implemented | Out of scope for this prototype |
| "≈17% positive rate" | 15.76% measured | Computed from the file |

## Secondary files

`diabetes_binary_health_indicators_BRFSS2015.csv` (prediabetes counted as negative) and
`diabetes_binary_5050split_health_indicators_BRFSS2015.csv` (artificially balanced) have a different
target definition and class distribution. They are **not used** anywhere in this project.

## Reproducibility notes

Tested on Windows 11 with Python 3.13.17 and the pinned versions in `requirements.txt`. The
saved model is a scikit-learn pickle, so load it with the same scikit-learn version. With a
different version, re-run the analysis. Results may differ slightly on other library versions or CPUs.
