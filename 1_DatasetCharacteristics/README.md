# Dataset Characteristics — Merge Pipeline

This folder contains the data and notebooks for merging sales and weather data used in the team project.

Summary of the pipeline

- Input raw CSVs (located in `1_DatasetCharacteristics/raw_data/`):
  - `umsatzdaten_gekuerzt.csv` — primary sales (Umsatz) data (contains `Datum`, `Warengruppe`, `Umsatz`, ...)
  - `test.csv` — additional sales rows to extend coverage
  - `wetter.csv` — daily weather measurements (contains `Datum`, `Temperatur`, `Wettercode`, `Windgeschwindigkeit`, ...)
  - `Niederschlag.csv` — precipitation data (if used separately)
  - `kiwo.csv` — auxiliary data

- Merge logic (implemented in `scripts/merge_datasets_fixed.py` and mirrored in `notebooks/data_merge.ipynb`):
  1. Load `umsatzdaten_gekuerzt.csv` and `test.csv` with `parse_dates=['Datum']`.
  2. Concatenate the two sales files vertically into `all_data` so `test.csv` rows are included in the base sales table.
  3. Load weather and auxiliary files with parsed dates.
  4. Outer-join weather and auxiliary data onto `all_data` using `Datum` as the key.
  5. Add a holiday indicator (`is_holiday`) and perform basic cleaning.
  6. Interpolate continuous weather columns (`Temperatur`, `Niederschlag`, `Windgeschwindigkeit`) and forward-fill `Wettercode`.
  7. Filter to keep only rows where `Umsatz` is present and save the final cleaned dataset.

Key scripts / notebooks

- `scripts/merge_datasets_fixed.py` — runnable script that performs the full merge and saves the result to `1_DatasetCharacteristics/processed_data/combined_data_outer.csv` (and filled/cleaned variants during processing).
- `1_DatasetCharacteristics/notebooks/data_merge.ipynb` — notebook which calls the script and displays summaries. Useful for reproducible steps and for inspection.

Outputs saved in `1_DatasetCharacteristics/processed_data/` (committed):
- `combined_data_outer.csv` — merged dataset (pre-filled)
- `combined_data_outer_filled_global.csv` — merged dataset after global interpolation/ffill of weather columns
- `combined_data_outer_final.csv` — final cleaned dataset with only rows that have `Umsatz` (this file was just committed)

How to reproduce locally

1. Create and activate a Python virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas nbconvert ipykernel
```

2. Run the merge script (recommended):

```bash
python3 scripts/merge_datasets_fixed.py
```

3. Or execute the notebook (the notebook runs the script):

```bash
python3 -m nbconvert --to notebook --execute 1_DatasetCharacteristics/notebooks/data_merge.ipynb --inplace
```

Notes / recommendations

- All date parsing uses `parse_dates=['Datum']` to ensure merges are performed on pandas datetimes.
- The script uses an outer join to preserve all dates from all sources. Use `_merge` (indicator) to audit unmatched rows.
- For large datasets, adapt the script to process by chunks or aggregate per-date before joining to reduce memory usage.

If you want, I can add a `requirements.txt`, a short top-level README entry, or a small CI check that ensures the merge runs in the project environment.
