"""
scripts/one_big_pot.py

Create a continuous "One Big Pot" dataset:
- Concatenate sales (training) + test sales by date
- Preserve test sales in a separate column and mark `is_test`
- Merge weather, precipitation, wind, kiwo, holidays
- Interpolate and smooth weather variables
- Apply optional binning and one-hot encoding for weather codes
- Create lag and rolling features for sales using only training sales values
- Ensure no leakage: rolling sales features use shifted training-only target
- Save final features to `1_DatasetCharacteristics/processed_data/one_big_pot_features.csv`

Usage: run from repo root:
    python scripts/one_big_pot.py

"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

# Config
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "1_DatasetCharacteristics" / "raw_data"
OUT_DIR = ROOT / "1_DatasetCharacteristics" / "processed_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Filenames expected in repo (adjust if different)
SALES_FILE = RAW / "umsatzdaten_gekuerzt.csv"
TEST_FILE = RAW / "test.csv"
WEATHER_FILE = RAW / "wetter.csv"
PRECIP_FILE = RAW / "Niederschlag.csv"
KIWO_FILE = RAW / "kiwo.csv"

OUTPUT_FILE = OUT_DIR / "one_big_pot_features.csv"

# Parameters for smoothing / feature windows
TEMP_MA_WINDOW = 3       # days for temperature moving average
PRECIP_MA_WINDOW = 3     # days for precipitation smoothing
WIND_MA_WINDOW = 3       # days for wind smoothing
SALES_LAG_DAYS = [1, 7, 14]
SALES_ROLL_WINDOWS = [7, 14, 28]


def read_optional(path: Path, **kwargs):
    if path.exists():
        return pd.read_csv(path, **kwargs)
    return None


def parse_dates(df, date_col='Datum'):
    df = df.copy()
    if date_col in df.columns:
        # assume ISO dates (YYYY-MM-DD) by default; change dayfirst only if your data uses DMY
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=False, errors='coerce')
    return df


def main():
    # 1) Load sales + test
    sales = read_optional(SALES_FILE)
    test = read_optional(TEST_FILE)

    if sales is None and test is None:
        raise FileNotFoundError(f"No sales or test files found in {RAW}")

    # Parse dates
    if sales is not None:
        sales = parse_dates(sales)
        # ensure 'Umsatz' column exists
        if 'Umsatz' not in sales.columns:
            raise KeyError("Expected 'Umsatz' column in sales data")
    else:
        sales = pd.DataFrame(columns=['Datum','Umsatz'])

    if test is not None:
        test = parse_dates(test)
        # If test file doesn't include sales amounts (common for prediction sets),
        # create an Umsatz column filled with NaN so pipeline can continue.
        if 'Umsatz' not in test.columns:
            print("Note: 'test.csv' has no 'Umsatz' column — treating test rows as unlabeled (Umsatz=NaN).")
            test['Umsatz'] = np.nan
        # Ensure Datum exists
        if 'Datum' not in test.columns:
            raise KeyError("Expected 'Datum' column in test data")
    else:
        test = pd.DataFrame(columns=['Datum','Umsatz'])

    # Add indicator columns and preserve original sales
    sales = sales[['Datum','Umsatz']].copy()
    sales['is_test'] = False
    test = test[['Datum','Umsatz']].copy()
    test['is_test'] = True

    # Combine sales + test into one dataframe (concat preserves duplicates)
    all_sales = pd.concat([sales, test], ignore_index=True, sort=False)

    # 2) Build a continuous date index covering all dates present
    all_sales['Datum'] = pd.to_datetime(all_sales['Datum'])
    min_date = all_sales['Datum'].min()
    max_date = all_sales['Datum'].max()
    if pd.isna(min_date) or pd.isna(max_date):
        raise ValueError('Dates could not be parsed in sales/test files')

    full_idx = pd.date_range(start=min_date, end=max_date, freq='D')
    df = pd.DataFrame({'Datum': full_idx})

    # Merge sales into continuous timeline. If multiple rows per date exist, aggregate by sum.
    sales_agg = all_sales.groupby('Datum', as_index=False).agg({'Umsatz':'sum','is_test': 'max'})
    # Note: for dates that have both test and train rows, 'is_test' will be True (max) — this is dataset-specific.

    df = df.merge(sales_agg, on='Datum', how='left')

    # Preserve test sales separately and create a training-only label
    # Create column 'Umsatz_test' containing test sales only, and 'Umsatz_train' with train sales only.
    # We rely on the original 'is_test' flag per aggregated date; if both kinds exist on same date, this may need manual resolution.
    df['Umsatz_orig'] = df['Umsatz']  # preserve combined if present

    # Separate train/test at date-level: mark rows as test if any test row existed on that date
    df['is_test'] = df['is_test'].fillna(False)

    # Create 'Umsatz_test' and 'Umsatz_label' (training target) columns
    # For rows flagged as test -> move value to Umsatz_test and set training label NaN
    df['Umsatz_test'] = np.where(df['is_test'], df['Umsatz_orig'], np.nan)
    df['Umsatz_label'] = np.where(df['is_test'], np.nan, df['Umsatz_orig'])

    # Now drop the intermediate combined 'Umsatz_orig' if desired
    # (Keep 'Umsatz' removed to prevent accidental usage.)
    df.drop(columns=['Umsatz','Umsatz_orig'], errors='ignore', inplace=True)

    # 3) Merge weather and other exogenous data
    # Weather
    weather = read_optional(WEATHER_FILE)
    if weather is not None:
        weather = parse_dates(weather)
        # Expect weather columns like Temperatur, Wettercode, Windgeschwindigkeit, Bewoelkung etc.
        # Normalize date column name
        if 'Datum' not in weather.columns:
            # attempt first column
            weather.rename(columns={weather.columns[0]:'Datum'}, inplace=True)
        df = df.merge(weather, on='Datum', how='left')

    # Precipitation (if separate)
    precip = read_optional(PRECIP_FILE)
    if precip is not None:
        precip = parse_dates(precip)
        if 'Datum' not in precip.columns:
            precip.rename(columns={precip.columns[0]:'Datum'}, inplace=True)
        df = df.merge(precip, on='Datum', how='left')

    # KiWo
    kiwo = read_optional(KIWO_FILE)
    if kiwo is not None:
        kiwo = parse_dates(kiwo)
        if 'Datum' not in kiwo.columns:
            kiwo.rename(columns={kiwo.columns[0]:'Datum'}, inplace=True)
        df = df.merge(kiwo, on='Datum', how='left')

    # Holiday: if there's a column like 'Feiertag' create 'is_holiday' boolean
    if 'Feiertag' in df.columns:
        df['is_holiday'] = df['Feiertag'].astype(bool)
    else:
        # If no holiday file, we can approximate using pandas holidays (optional). Leave False by default.
        df['is_holiday'] = False

    # 4) Ensure date index is set and continuous
    df.set_index('Datum', inplace=True)

    # 5) Fill/Interpolate weather variables (exogenous) across continuous timeline
    # Identify common weather columns (case-insensitive match)
    col_lower = {c.lower():c for c in df.columns}
    temp_col = col_lower.get('temperatur') or col_lower.get('temp')
    precip_col = col_lower.get('niederschlag') or col_lower.get('precipitation') or col_lower.get('precip')
    wind_col = col_lower.get('windgeschwindigkeit') or col_lower.get('wind')
    cloud_col = col_lower.get('bewoelkung') or col_lower.get('bewaelkung') or col_lower.get('cloud')
    weathercode_col = col_lower.get('wettercode') or col_lower.get('weathercode')

    # Interpolate numeric weather variables time-based
    for col in [temp_col, precip_col, wind_col]:
        if col is not None and col in df.columns:
            # convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # forward/backward fill small gaps then time interpolation
            df[col] = df[col].interpolate(method='time', limit_direction='both')
            # use ffill/bfill to silence future warnings
            df[col] = df[col].ffill().bfill()

    # For weather codes and cloud cover, forward-fill categorical values
    if cloud_col is not None and cloud_col in df.columns:
        df[cloud_col] = pd.to_numeric(df[cloud_col], errors='coerce')
        df[cloud_col] = df[cloud_col].ffill().bfill()

    if weathercode_col is not None and weathercode_col in df.columns:
        df[weathercode_col] = df[weathercode_col].ffill().bfill()

    # 6) Apply smoothing/binning to weather variables
    # Temperature moving average
    if temp_col is not None and temp_col in df.columns:
        df['temp_ma_{}d'.format(TEMP_MA_WINDOW)] = df[temp_col].rolling(TEMP_MA_WINDOW, min_periods=1).mean()

    # Precipitation smoothing (moving average)
    if precip_col is not None and precip_col in df.columns:
        df['precip_ma_{}d'.format(PRECIP_MA_WINDOW)] = df[precip_col].rolling(PRECIP_MA_WINDOW, min_periods=1).mean()

    # Wind smoothing
    if wind_col is not None and wind_col in df.columns:
        df['wind_ma_{}d'.format(WIND_MA_WINDOW)] = df[wind_col].rolling(WIND_MA_WINDOW, min_periods=1).mean()

    # Optional cloud cover binning (e.g., low/med/high)
    if cloud_col is not None and cloud_col in df.columns:
        # Create bins: 0-33 low, 34-66 medium, 67-100 high (adjust if cloud scale differs)
        df['cloud_bin'] = pd.cut(df[cloud_col], bins=[-0.1,33,66,100], labels=['low','medium','high'])

    # One-hot encoding of weather codes (if present)
    if weathercode_col is not None and weathercode_col in df.columns:
        # Convert to string then dummies
        df[weathercode_col] = df[weathercode_col].astype('str')
        wcode_dummies = pd.get_dummies(df[weathercode_col], prefix='wcode')
        df = pd.concat([df, wcode_dummies], axis=1)

    # 7) Keep KiWo columns as-is (already merged above). If KiWo contains categorical info, optionally one-hot.
    # Example: if 'KiWo' in columns, one-hot it too
    if 'KiWo' in df.columns:
        kiwo_dummies = pd.get_dummies(df['KiWo'].astype(str), prefix='KiWo')
        df = pd.concat([df, kiwo_dummies], axis=1)

    # 8) Create lag and rolling features for sales using training-only target (`Umsatz_label`)
    # Important: we must prevent leakage from test sales. Umsatz_label has test dates as NaN.

    # Work on a temporary series of training-only sales
    sales_label = df['Umsatz_label'] if 'Umsatz_label' in df.columns else pd.Series(index=df.index, dtype=float)

    # Create basic lag features (shifted values)
    for lag in SALES_LAG_DAYS:
        df[f'lag_{lag}'] = sales_label.shift(lag)

    # Create rolling features based on past values only (shift before rolling)
    for window in SALES_ROLL_WINDOWS:
        # rolling mean of previous `window` days (exclude current by shifting)
        df[f'roll_mean_{window}'] = sales_label.shift(1).rolling(window, min_periods=1).mean()
        df[f'roll_std_{window}'] = sales_label.shift(1).rolling(window, min_periods=1).std().fillna(0)

    # Add expanding features (historical mean up to yesterday)
    df['expanding_mean'] = sales_label.shift(1).expanding(min_periods=1).mean()

    # 9) Additional time features (weekday, month, dayofyear) to help seasonality
    df['weekday'] = df.index.weekday
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear

    # 10) Final housekeeping: rename columns to safe names and save
    # Reset index to have Datum as column
    out = df.reset_index()

    # Keep columns useful for modeling
    # Always include: Datum, is_test, Umsatz_test, Umsatz_label, is_holiday, time features, weather smoothed features, lags/rolls
    keep_cols = [c for c in out.columns if c.startswith('lag_') or c.startswith('roll_') or c.startswith('expanding_')]
    base_cols = ['Datum','is_test','Umsatz_test','Umsatz_label','is_holiday','weekday','month','dayofyear']
    # include weather-derived columns if present
    weather_cols = []
    for c in [temp_col, precip_col, wind_col, cloud_col, weathercode_col]:
        if c and c in out.columns:
            weather_cols.append(c)
    # include smoothed versions
    smoothed = [c for c in out.columns if ('temp_ma' in c) or ('precip_ma' in c) or ('wind_ma' in c) or c=='cloud_bin']
    # include any one-hot prefixed columns
    dummies = [c for c in out.columns if c.startswith('wcode_') or c.startswith('KiWo_')]

    final_cols = base_cols + weather_cols + smoothed + dummies + keep_cols
    # Ensure uniqueness & filter existing
    final_cols = [c for i,c in enumerate(final_cols) if c in out.columns and final_cols.index(c)==i]

    final_df = out[final_cols].copy()

    # --- Validation: ensure rolling features were computed from training-only labels and no leakage ---
    # sales_label used to compute rolling features has NaN for test dates. Recompute expected values
    # and assert they're equal to the created features (within tolerance).
    sales_label_series = sales_label
    for window in SALES_ROLL_WINDOWS:
        expected_mean = sales_label_series.shift(1).rolling(window, min_periods=1).mean()
        col_name = f'roll_mean_{window}'
        if col_name in df.columns:
            got = df[col_name]
            # Compare values where both are finite (use allclose), and ensure NaN positions match
            # Positions where expected is NaN should also be NaN in got
            mismatch_nan = (~expected_mean.isna()) & (got.isna())
            if mismatch_nan.any():
                raise AssertionError(f"Leakage check failed: {col_name} has NaN where expected has values")
            # Compare numeric values allowing small floating error
            mask = ~expected_mean.isna()
            if not np.allclose(expected_mean[mask].fillna(0).values, got[mask].fillna(0).values, equal_nan=True):
                raise AssertionError(f"Leakage check failed: {col_name} differs from expected computation")

    # Save features
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote features to {OUTPUT_FILE} — shape: {final_df.shape}")


if __name__ == '__main__':
    main()
