# data/

`raw/` holds the input data exactly as downloaded. Nothing in this project writes to it.
The primary CSV is tracked in git so Streamlit Cloud (and clones) can run Explore without
a separate download.

| File | Used? | Notes |
|---|---|---|
| `raw/diabetes_012_health_indicators_BRFSS2015.csv` | **Yes — primary** | 253,680 rows; target `Diabetes_012` (0 / 1 prediabetes / 2 diabetes). |
| `diabetes_binary_health_indicators_BRFSS2015.csv` | No | Different target (prediabetes counted as negative; 35,346 positives). |
| `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` | No | Artificially balanced 50/50 subsample. |

The two secondary files are not interchangeable with the primary file and are not needed.

To use a copy stored elsewhere, set `DIABETES_DATA_PATH` to the full file path before running.

The analysis records the file's SHA-256 in `outputs/run_metadata.json`; the dashboard warns if
the file on disk no longer matches.
