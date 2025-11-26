import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import cross_val_score
import numpy as np

# --- CONFIGURATION ---
DATA_DIR = Path('1_DatasetCharacteristics/processed_data')
TRAIN_FILE = DATA_DIR / 'train_features.csv'

def main():
    print("🚀 Starting XGBoost Training Script...")

    # 1. SAFETY CHECK
    if not TRAIN_FILE.exists():
        print("\n❌ ERROR: File not found!")
        print(f"   Looking for: {TRAIN_FILE}")
        print("   👉 FIX: Please run 'python scripts/ml_pipeline.py' first.\n")
        return

    # 2. DATA LOADING
    print(f"📂 Loading data from {TRAIN_FILE}...")
    df = pd.read_csv(TRAIN_FILE, parse_dates=['Datum'])

    # 3. TARGET DETECTION
    if 'Umsatz' in df.columns:
        target_col = 'Umsatz'
    elif 'Umsatz_label' in df.columns:
        target_col = 'Umsatz_label'
    else:
        print("\n❌ CRITICAL ERROR: Target column (Sales) is MISSING!")
        return

    # --- FIX 1: DROP ROWS WITH MISSING TARGETS (TEST DATA) ---
    print(f"   Original row count: {len(df)}")
    df = df.dropna(subset=[target_col])
    print(f"   Row count after removing NaNs: {len(df)}")
    # ---------------------------------------------------------

    # 4. FEATURE DEFINITION
    metadata_cols = [target_col, 'Umsatz', 'Umsatz_label', 'Datum', 'is_test', 'id', 'Warengruppe']
    drop_cols = [c for c in metadata_cols if c in df.columns]
    
    X = df.drop(columns=drop_cols)
    y = df[target_col]

    # --- FIX 2: HANDLE TEXT COLUMNS AUTOMATICALLY ---
    # XGBoost needs text converted to 'category' type
    for col in X.select_dtypes(include=['object', 'string']).columns:
        X[col] = X[col].astype('category')
    # ------------------------------------------------

    print(f"✅ Data Ready. Training on {X.shape[1]} features.")

    # 5. TRAIN & VALIDATE
    print("🔄 Running 5-Fold Cross-Validation (XGBoost)...")
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        enable_categorical=True,  # Allows reading the category columns
        tree_method='hist'        # Required for categorical support
    )
    
    scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=5)
    avg_rmse = np.mean(-scores)

    # 6. REPORT RESULTS
    print("-" * 40)
    print(f"🏆 XGBoost Average RMSE: {avg_rmse:.4f}")
    print("-" * 40)

    if avg_rmse < 164.2:
        print("✨ SUCCESS! This beats the Linear Baseline (164.2)!")
    else:
        print("📉 Result is higher than Linear Baseline (164.2). Needs tuning.")

if __name__ == "__main__":
    main()