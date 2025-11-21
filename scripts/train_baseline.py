"""
Train a simple Linear Regression baseline using the One Big Pot features.

- Loads `train_features.csv` and `test_features.csv` (but uses only train for fitting/validation).
- Features: `Temperatur`, `Niederschlag`, `Windgeschwindigkeit`, `Wettercode`, `is_holiday`.
- Target: `Umsatz` (falls back to `Umsatz_label`).
- Splits train into 80/20 train/validation.
- Trains `LinearRegression` and prints RMSE on validation set.

Run: python scripts/train_baseline.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from math import sqrt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / '1_DatasetCharacteristics' / 'processed_data'
TRAIN_FILE = DATA_DIR / 'train_features.csv'
TEST_FILE = DATA_DIR / 'test_features.csv'

FEATURE_COLUMNS = ['Temperatur','Niederschlag','Windgeschwindigkeit','Wettercode','is_holiday']

def load_data(path):
    df = pd.read_csv(path, parse_dates=['Datum'], dayfirst=False)
    return df


def prepare_xy(df):
    # Determine target column
    if 'Umsatz' in df.columns:
        y = df['Umsatz']
    elif 'Umsatz_label' in df.columns:
        y = df['Umsatz_label']
    else:
        raise KeyError('No Umsatz or Umsatz_label target in dataframe')

    X = df.copy()
    # Keep only requested features; if missing, create with NaN
    for c in FEATURE_COLUMNS:
        if c not in X.columns:
            X[c] = np.nan

    X = X[FEATURE_COLUMNS]

    # Coerce Wettercode to numeric if present
    if 'Wettercode' in X.columns:
        X['Wettercode'] = pd.to_numeric(X['Wettercode'], errors='coerce')

    # Fill numeric NaNs with median
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        med = X[c].median()
        if pd.isna(med):
            med = 0.0
        X[c] = X[c].fillna(med)

    # Fill is_holiday as 0/1
    if 'is_holiday' in X.columns:
        X['is_holiday'] = X['is_holiday'].fillna(False).astype(int)

    # Align y and X, drop rows where y is NaN
    mask = ~y.isna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    return X, y


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Train file not found: {TRAIN_FILE}")

    df_train = load_data(TRAIN_FILE)
    X, y = prepare_xy(df_train)

    # Split 80/20 using a reproducible random permutation
    n = len(X)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    cut = int(n * 0.8)
    train_idx = idx[:cut]
    val_idx = idx[cut:]

    X_train = X.iloc[train_idx].to_numpy()
    y_train = y.iloc[train_idx].to_numpy()
    X_val = X.iloc[val_idx].to_numpy()
    y_val = y.iloc[val_idx].to_numpy()

    # Fit linear regression via least squares with intercept
    # Add column of ones for intercept
    X_train_aug = np.hstack([np.ones((X_train.shape[0],1)), X_train])
    coef, *_ = np.linalg.lstsq(X_train_aug, y_train, rcond=None)

    # Predict on validation set
    X_val_aug = np.hstack([np.ones((X_val.shape[0],1)), X_val])
    y_pred = X_val_aug.dot(coef)

    # RMSE
    rmse = sqrt(((y_val - y_pred)**2).mean())
    print(f"Validation RMSE: {rmse:.4f}")

    # Print coefficients (intercept + feature coefs)
    cols = ['intercept'] + X.columns.tolist()
    print('\nModel coefficients:')
    for c, v in zip(cols, coef):
        print(f"  {c}: {v:.6f}")

if __name__ == '__main__':
    main()
