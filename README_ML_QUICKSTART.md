ML Pipeline Quickstart

This document explains how to run the ML pipeline and where to find outputs.

Requirements
- Python 3.8+
- pandas, numpy (these are used by the scripts). If you use a virtualenv/conda env, activate it first.

Run the pipeline
From the repository root run:

```bash
python scripts/ml_pipeline.py
```

What the script does
- Builds and sanitizes the training features (One Big Pot outputs expected under `1_DatasetCharacteristics/processed_data/`).
- Runs a ridge alpha grid search to pick the best regularization value by time-series CV.
- Computes baseline and cleaned models (feature reduction + grouping of rare weather codes), preserves Top-30 baseline features by importance, and evaluates them.

Where to find outputs
- `1_DatasetCharacteristics/processed_data/ml_metrics_summary.json` — final CV metrics for baseline and cleaned models.
- `1_DatasetCharacteristics/processed_data/ml_report.txt` — human-readable report and lists of dropped/grouped features.
- `1_DatasetCharacteristics/processed_data/feature_importances_baseline.csv` — baseline feature importances.
- `1_DatasetCharacteristics/processed_data/feature_importances_cleaned.csv` — cleaned feature importances.
- `1_DatasetCharacteristics/processed_data/ridge_alpha_grid.json` — per-alpha CV metrics from the grid search.
- `1_DatasetCharacteristics/processed_data/ml_fold_diagnostics.csv` — per-fold diagnostics (condition numbers, coef NaN flags, y_val/y_pred stats).

Notes
- The script filters out rows with missing target values before cross-validation.
- If you prefer tree-based models (LightGBM/XGBoost), set up a compatible Python environment and modify the pipeline accordingly.

Contact
- If you need me to tune hyperparameters further, try different `TOP_KEEP` values, or add LightGBM support, tell me and I'll implement those changes.
