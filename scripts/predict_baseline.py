"""
Fit linear least-squares on full training set and predict baseline on test_features.csv
- Uses same features as training script.
- Saves predictions to `1_DatasetCharacteristics/processed_data/test_predictions_baseline.csv` with columns: Datum, Umsatz_pred

Run: python scripts/predict_baseline.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from math import sqrt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / '1_DatasetCharacteristics' / 'processed_data'
TRAIN_FILE = DATA_DIR / 'train_features.csv'
TEST_FILE = DATA_DIR / 'test_features.csv'
OUT_FILE = DATA_DIR / 'test_predictions_baseline.csv'
FEATURE_COLUMNS = ['Temperatur','Niederschlag','Windgeschwindigkeit','Wettercode','is_holiday']


def load_df(path):
    return pd.read_csv(path, parse_dates=['Datum'], dayfirst=False)


def prepare_X(df):
    X = df.copy()
    for c in FEATURE_COLUMNS:
        if c not in X.columns:
            X[c] = np.nan
    X = X[FEATURE_COLUMNS]
    if 'Wettercode' in X.columns:
        X['Wettercode'] = pd.to_numeric(X['Wettercode'], errors='coerce')
    # Fill numerics with median
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        med = X[c].median()
        if pd.isna(med):
            med = 0.0
        X[c] = X[c].fillna(med)
    # is_holiday to 0/1
    if 'is_holiday' in X.columns:
        X['is_holiday'] = X['is_holiday'].fillna(False).astype(int)
    return X


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(TRAIN_FILE)
    if not TEST_FILE.exists():
        raise FileNotFoundError(TEST_FILE)

    df_train = load_df(TRAIN_FILE)
    # target
    if 'Umsatz' in df_train.columns:
        y = df_train['Umsatz']
    else:
        y = df_train['Umsatz_label']
    X_train = prepare_X(df_train)
    # drop rows with NaN target
    mask = ~y.isna()
    X_train = X_train.loc[mask].to_numpy()
    y_train = y.loc[mask].to_numpy()

    # fit least squares with intercept
    X_aug = np.hstack([np.ones((X_train.shape[0],1)), X_train])
    coef, *_ = np.linalg.lstsq(X_aug, y_train, rcond=None)

    # load test and prepare
    df_test = load_df(TEST_FILE)
    X_test_df = prepare_X(df_test)
    X_test = X_test_df.to_numpy()
    X_test_aug = np.hstack([np.ones((X_test.shape[0],1)), X_test])
    preds = X_test_aug.dot(coef)

    out = pd.DataFrame({'Datum': df_test['Datum'], 'Umsatz_pred': preds})
    out.to_csv(OUT_FILE, index=False)
    print(f"Wrote predictions to {OUT_FILE} — shape: {out.shape}")

if __name__ == '__main__':
    main()
