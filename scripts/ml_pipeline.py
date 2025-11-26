"""
scripts/ml_pipeline.py

Performs the requested sequence:
1) Baseline: train LightGBM with TimeSeriesSplit (walk-forward CV). Report MAE, RMSE, MAPE.
2) Analyze features: near-zero variance, all zeros, multicollinearity (VIF), sparse one-hots.
3) Suggest and apply reductions: drop low-variance/all-zeros, merge rare wcode_* into wcode_rare, drop high-VIF features.
4) Retrain on cleaned set with the same CV and compare metrics.
5) Output report and feature importance lists.

Usage: python scripts/ml_pipeline.py

Outputs:
- processed_data/ml_report.txt
- processed_data/feature_importances_baseline.csv
- processed_data/feature_importances_cleaned.csv
- processed_data/ml_metrics_summary.json

"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / '1_DatasetCharacteristics' / 'processed_data'
TRAIN_FILE = DATA_DIR / 'train_features.csv'
OUT_DIR = DATA_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = OUT_DIR / 'ml_report.txt'
IMP_BASE = OUT_DIR / 'feature_importances_baseline.csv'
IMP_CLEAN = OUT_DIR / 'feature_importances_cleaned.csv'
METRICS = OUT_DIR / 'ml_metrics_summary.json'

# Parameters
N_SPLITS = 5
RANDOM_STATE = 42
VIF_THRESHOLD = 10.0
NZV_VARIANCE_THRESH = 1e-6
SPARSE_RATIO_THRESH = 0.01  # if nonzero fraction < threshold
WCODE_PREFIX = 'wcode_'
TOP_KEEP = 30  # number of top baseline features to preserve when reducing
# Grid search alphas for ridge
ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 200.0]

# XGBoost parameters
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'random_state': RANDOM_STATE,
    'verbosity': 0
}
# XGBoost hyperparameter grid
XGB_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.1, 0.3],
    'subsample': [0.8, 1.0]
}


def load_train():
    df = pd.read_csv(TRAIN_FILE, parse_dates=['Datum'])
    # sort by date to ensure time order
    df = df.sort_values('Datum').reset_index(drop=True)
    return df


def select_features(df):
    # Exclude index/target columns
    exclude = set(['Datum','is_test','Umsatz','Umsatz_label'])
    # Keep numeric columns and dummies (already numeric)
    cand = [c for c in df.columns if c not in exclude]
    return cand


def target_series(df):
    if 'Umsatz' in df.columns:
        return df['Umsatz']
    return df['Umsatz_label']


def _time_series_splits(n_samples, n_splits=N_SPLITS):
    # simple expanding window: generate n_splits folds where validation windows are contiguous
    # compute split indices by equally spacing end points
    indices = np.arange(n_samples)
    test_size = n_samples // (n_splits + 1)
    splits = []
    for i in range(n_splits):
        train_end = test_size * (i + 1)
        val_start = train_end
        val_end = val_start + test_size
        if val_end > n_samples:
            val_end = n_samples
        train_idx = indices[:train_end]
        val_idx = indices[val_start:val_end]
        if len(val_idx) == 0:
            continue
        splits.append((train_idx, val_idx))
    return splits


def time_series_cv_metrics(X, y, features, n_splits=N_SPLITS, alpha=1.0):
    n = len(X)
    splits = _time_series_splits(n, n_splits=n_splits)
    metrics = []
    importances = []
    # diagnostics per-fold
    fold_diags = []
    fold = 0
    for train_idx, val_idx in splits:
        fold += 1
        X_train = X.iloc[train_idx][features].fillna(0).to_numpy()
        X_val = X.iloc[val_idx][features].fillna(0).to_numpy()
        y_train = y.iloc[train_idx].to_numpy()
        y_val = y.iloc[val_idx].to_numpy()
        # ensure numeric dtype for linear algebra (defensive cast)
        try:
            X_train = X_train.astype(float)
            X_val = X_val.astype(float)
            y_train = y_train.astype(float)
            y_val = y_val.astype(float)
        except Exception:
            # if cast fails, raise helpful error for debugging
            raise ValueError('Could not convert training/validation arrays to float. Check feature dtypes.')
        # fit ridge regression via normal equations: (X^T X + alpha I)^{-1} X^T y
        # Add small regularization on intercept by not regularizing the first column (we'll add intercept separately)
        X_train_aug = np.hstack([np.ones((X_train.shape[0],1)), X_train])
        XtX = X_train_aug.T.dot(X_train_aug)
        # regularize all except intercept
        reg = alpha * np.eye(XtX.shape[0])
        reg[0,0] = 0.0
        # Solve ridge regression using augmented least-squares for numerical stability
        try:
            k = X_train_aug.shape[1]
            sqrt_alpha = np.sqrt(alpha)
            A = np.vstack([X_train_aug, sqrt_alpha * np.eye(k)])
            b = np.concatenate([y_train, np.zeros(k)])
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            X_val_aug = np.hstack([np.ones((X_val.shape[0],1)), X_val])
            y_pred = X_val_aug.dot(coef)
        except Exception:
            coef = np.full((X_train_aug.shape[1],), np.nan)
            y_pred = np.full((X_val.shape[0],), np.nan)
        # fold diagnostics
        try:
            # condition number of augmented matrix A used in lstsq
            cond = float(np.linalg.cond(A))
        except Exception:
            cond = float('nan')
        coef_has_nan = bool(~np.isfinite(coef).all())
        coef_norm = float(np.linalg.norm(coef)) if np.isfinite(np.linalg.norm(coef)) else float('nan')
        y_pred_has_nan = bool(~np.isfinite(y_pred).all())
        fold_diags.append({
            'fold': fold,
            'cond_XtX_reg': cond,
            'coef_has_nan': coef_has_nan,
            'coef_norm': coef_norm,
            'y_val_mean': float(np.nanmean(y_val)),
            'y_val_std': float(np.nanstd(y_val)),
            'y_pred_mean': float(np.nanmean(y_pred)),
            'y_pred_std': float(np.nanstd(y_pred)),
            'y_pred_has_nan': y_pred_has_nan
        })
        mae = float(np.mean(np.abs(y_val - y_pred)))
        rmse = float(np.sqrt(np.mean((y_val - y_pred) ** 2)))
        mape = float(np.mean(np.abs((y_val - y_pred) / np.where(y_val == 0, 1e-8, y_val))) * 100.0)
        metrics.append({'fold': fold, 'mae': mae, 'rmse': rmse, 'mape': mape})
        # feature importances approximated by absolute coefficient magnitudes (exclude intercept)
        imp = pd.Series(np.abs(coef[1:]), index=features)
        importances.append(imp)
    imp_df = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    mean_metrics = {k: float(np.mean([m[k] for m in metrics])) for k in ['mae', 'rmse', 'mape']}
    # write fold diagnostics
    try:
        diag_df = pd.DataFrame(fold_diags)
        diag_df.to_csv(OUT_DIR / 'ml_fold_diagnostics.csv', index=False)
    except Exception:
        pass
    return mean_metrics, imp_df


def xgboost_cv_metrics(X, y, features, n_splits=N_SPLITS, **xgb_params):
    """XGBoost cross-validation with time series splits."""
    n = len(X)
    splits = _time_series_splits(n, n_splits=n_splits)
    metrics = []
    importances = []
    fold_diags = []
    fold = 0
    
    for train_idx, val_idx in splits:
        fold += 1
        X_train = X.iloc[train_idx][features].fillna(0)
        X_val = X.iloc[val_idx][features].fillna(0)
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        
        try:
            # Create XGBoost model
            model = xgb.XGBRegressor(**XGB_PARAMS, **xgb_params)
            
            # Fit model
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            mae = float(np.mean(np.abs(y_val - y_pred)))
            rmse = float(np.sqrt(np.mean((y_val - y_pred) ** 2)))
            mape = float(np.mean(np.abs((y_val - y_pred) / np.where(y_val == 0, 1e-8, y_val))) * 100.0)
            
            metrics.append({'fold': fold, 'mae': mae, 'rmse': rmse, 'mape': mape})
            
            # Feature importances
            imp = pd.Series(model.feature_importances_, index=features)
            importances.append(imp)
            
            # Fold diagnostics
            fold_diags.append({
                'fold': fold,
                'best_iteration': int(model.best_iteration) if hasattr(model, 'best_iteration') else -1,
                'n_estimators_used': int(model.n_estimators),
                'y_val_mean': float(np.mean(y_val)),
                'y_val_std': float(np.std(y_val)),
                'y_pred_mean': float(np.mean(y_pred)),
                'y_pred_std': float(np.std(y_pred))
            })
            
        except Exception as e:
            print(f"XGBoost fold {fold} failed: {e}")
            # Fill with NaN metrics for failed fold
            metrics.append({'fold': fold, 'mae': float('nan'), 'rmse': float('nan'), 'mape': float('nan')})
            imp = pd.Series(np.full(len(features), np.nan), index=features)
            importances.append(imp)
    
    # Aggregate results
    if importances:
        imp_df = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    else:
        imp_df = pd.Series(index=features, dtype=float)
    
    mean_metrics = {k: float(np.nanmean([m[k] for m in metrics])) for k in ['mae', 'rmse', 'mape']}
    
    # Write XGBoost fold diagnostics
    try:
        if fold_diags:
            diag_df = pd.DataFrame(fold_diags)
            diag_df.to_csv(OUT_DIR / 'ml_xgboost_fold_diagnostics.csv', index=False)
    except Exception:
        pass
    
    return mean_metrics, imp_df


def analyze_features(X, features):
    report = {}
    sub = X[features]
    # near-zero variance
    variances = sub.var(numeric_only=True)
    nzv = variances[variances <= NZV_VARIANCE_THRESH].index.tolist()
    report['near_zero_variance'] = nzv
    # all zeros
    all_zero = [c for c in features if (sub[c].fillna(0)==0).all()]
    report['all_zero'] = all_zero
    # sparsity for one-hot categories
    sparse = []
    for c in features:
        nonzero_frac = (sub[c].fillna(0) != 0).mean()
        if nonzero_frac < SPARSE_RATIO_THRESH:
            sparse.append({'col':c,'nonzero_frac':float(nonzero_frac)})
    report['sparse_cols'] = sorted(sparse, key=lambda x: x['nonzero_frac'])
    # multicollinearity: compute VIF via R^2 of regressing each feature on others
    vif = {}
    for c in features:
        Xi = sub[features].copy().fillna(0)
        # Cast to float arrays for linear algebra; handle failures gracefully
        try:
            y_var = Xi[c].astype(float).values
            X_others = Xi.drop(columns=[c]).astype(float).values
        except Exception:
            # fallback: try converting column-by-column
            y_var = pd.to_numeric(Xi[c], errors='coerce').fillna(0).values.astype(float)
            X_others = np.zeros((len(Xi), max(0, len(features)-1)), dtype=float)
            if X_others.shape[1] > 0:
                cols = Xi.drop(columns=[c]).columns.tolist()
                for j, col in enumerate(cols):
                    X_others[:, j] = pd.to_numeric(Xi[col], errors='coerce').fillna(0).values.astype(float)
        if X_others.shape[1] == 0:
            vif[c] = 0.0
            continue
        # compute R^2 of regressing c on the other features using least squares
        coefs, *_ = np.linalg.lstsq(X_others, y_var, rcond=None)
        y_pred = X_others.dot(coefs)
        ss_res = np.sum((y_var - y_pred) ** 2)
        ss_tot = np.sum((y_var - np.mean(y_var)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        vif_val = 1.0 / (1.0 - r2) if (1.0 - r2) > 1e-8 else float('inf')
        vif[c] = float(vif_val)
    vif_s = pd.Series(vif).sort_values(ascending=False)
    report['vif'] = vif_s.to_dict()
    return report


def propose_feature_reduction(report, features):
    to_drop = set()
    # drop all-zero and near-zero variance
    to_drop.update(report.get('all_zero',[]))
    to_drop.update(report.get('near_zero_variance',[]))
    # drop features with extremely high VIF
    vif = report.get('vif',{})
    for c,v in vif.items():
        if v is None:
            continue
        if v > VIF_THRESHOLD:
            to_drop.add(c)
    # For sparse one-hot columns, propose grouping later
    sparse_cols = [d['col'] for d in report.get('sparse_cols',[])]
    return list(to_drop), sparse_cols


def group_rare_wcodes(X, features, sparse_cols):
    # Identify weather code one-hot columns starting with prefix
    wcode_cols = [c for c in features if c.startswith(WCODE_PREFIX)]
    # find rare ones (in sparse_cols)
    rare = [c for c in wcode_cols if c in sparse_cols]
    if not rare:
        return X, []
    # create new column 'wcode_rare' as sum of rare indicators
    X = X.copy()
    X['wcode_rare'] = X[rare].fillna(0).sum(axis=1)
    # drop rare columns
    X = X.drop(columns=rare)
    return X, rare


def drop_features(X, drop_list):
    X = X.drop(columns=[c for c in drop_list if c in X.columns])
    return X


def run_pipeline():
    df = load_train()
    features = select_features(df)
    y = target_series(df)
    X = df[features].copy()
    # Ensure numeric types
    X = X.fillna(0)
    # Convert non-numeric columns to numeric with factorize (preserve ordering across CV)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.factorize(X[c])[0]

    # Baseline training (use sanitized numeric X to avoid object-dtype issues)
    print('Preparing diagnostics and running baseline CV...')
    # write diagnostics: dtypes and a sample of the sanitized feature matrix
    debug_types = OUT_DIR / 'ml_debug_dtypes.csv'
    debug_sample = OUT_DIR / 'ml_debug_sample.csv'
    debug_non_numeric = OUT_DIR / 'ml_debug_non_numeric.txt'
    # recompute features from sanitized X to ensure alignment
    features = X.columns.tolist()
    try:
        pd.Series(X.dtypes.astype(str)).to_csv(debug_types, header=['dtype'])
        X.head(20).to_csv(debug_sample, index=False)
        non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        with open(debug_non_numeric, 'w') as f:
            for c in non_numeric:
                f.write(f"{c}\n")
    except Exception as e:
        print('Warning writing diagnostic files:', e)
    print('Diagnostics written:', debug_types.name, debug_sample.name)
    # filter out rows with missing target to avoid NaNs during CV
    target = y
    non_null_mask = target.notna()
    if non_null_mask.sum() < len(target):
        missing = len(target) - int(non_null_mask.sum())
        print(f'Note: {missing} rows have missing target and will be excluded from CV')
        X = X.loc[non_null_mask].reset_index(drop=True)
        y = target.loc[non_null_mask].reset_index(drop=True)
    # Grid-search alpha on sanitized X to find best alpha (lowest RMSE)
    print('Running alpha grid search for ridge regularization:', ALPHA_GRID)
    alpha_results = {}
    best_alpha = None
    best_rmse = float('inf')
    for a in ALPHA_GRID:
        try:
            metrics_a, imp_a = time_series_cv_metrics(X, y, features, alpha=a)
            alpha_results[a] = metrics_a
            print(f' alpha={a} -> rmse={metrics_a.get("rmse")}')
            if metrics_a.get('rmse', float('inf')) < best_rmse:
                best_rmse = metrics_a.get('rmse')
                best_alpha = a
        except Exception as e:
            print(' alpha', a, 'failed:', e)
    # write grid results
    try:
        import json
        (OUT_DIR / 'ridge_alpha_grid.json').write_text(json.dumps(alpha_results, indent=2))
    except Exception:
        pass
    if best_alpha is None:
        best_alpha = 1.0
    print('Best alpha chosen:', best_alpha, 'with RMSE:', best_rmse)
    # use best alpha for baseline Ridge
    baseline_metrics, imp_baseline = time_series_cv_metrics(X, y, features, alpha=best_alpha)
    imp_baseline.to_csv(IMP_BASE, header=['importance'])
    
    # Train XGBoost baseline with grid search
    print('Running XGBoost baseline with grid search...')
    xgb_results = {}
    best_xgb_params = None
    best_xgb_rmse = float('inf')
    
    # Simple grid search for XGBoost
    param_combinations = [
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8},
        {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 1.0},
        {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.3, 'subsample': 0.8},
        {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 1.0}
    ]
    
    for params in param_combinations:
        try:
            params_key = str(params)
            print(f' XGBoost params: {params}')
            xgb_metrics, xgb_imp = xgboost_cv_metrics(X, y, features, **params)
            xgb_results[params_key] = {'params': params, 'metrics': xgb_metrics}
            print(f' XGBoost rmse: {xgb_metrics.get("rmse")}')
            
            if xgb_metrics.get('rmse', float('inf')) < best_xgb_rmse:
                best_xgb_rmse = xgb_metrics.get('rmse')
                best_xgb_params = params
                baseline_xgb_metrics = xgb_metrics
                imp_xgb_baseline = xgb_imp
        except Exception as e:
            print(f' XGBoost params {params} failed: {e}')
    
    # Save XGBoost baseline importances
    if best_xgb_params is not None:
        imp_xgb_baseline.to_csv(OUT_DIR / 'feature_importances_xgb_baseline.csv', header=['importance'])
        print(f'Best XGBoost params: {best_xgb_params} with RMSE: {best_xgb_rmse}')
    else:
        print('XGBoost baseline training failed')
        baseline_xgb_metrics = {'mae': float('nan'), 'rmse': float('nan'), 'mape': float('nan')}
        imp_xgb_baseline = pd.Series(dtype=float)

    # Analyze features
    print('Analyzing features...')
    analysis = analyze_features(X, features)

    # Propose reductions
    drop_list, sparse_cols = propose_feature_reduction(analysis, features)
    # Preserve top baseline features to avoid removing highly predictive columns
    try:
        top_keep = imp_baseline.head(TOP_KEEP).index.tolist()
        drop_list = [c for c in drop_list if c not in top_keep]
        if top_keep:
            print(f'Preserving top {len(top_keep)} baseline features: {top_keep[:10]}{"..." if len(top_keep)>10 else ""}')
    except Exception:
        top_keep = []

    # Apply grouping of rare wcode columns (operate on sanitized X)
    X_grouped, rare_wcodes = group_rare_wcodes(X, features, sparse_cols)
    grouped_features = X_grouped.columns.tolist()
    # Drop proposed features (from original df)
    cleaned_df = df.copy()
    cleaned_df = drop_features(cleaned_df, drop_list)
    # If we grouped rare wcodes, align cleaned_df with grouping
    if rare_wcodes:
        existing_rare = [c for c in rare_wcodes if c in df.columns]
        if existing_rare:
            cleaned_df['wcode_rare'] = df[existing_rare].fillna(0).sum(axis=1)
            # drop only columns that actually exist in cleaned_df
            cleaned_df = cleaned_df.drop(columns=[c for c in existing_rare if c in cleaned_df.columns])

    cleaned_features = select_features(cleaned_df)
    # coerce non-numeric in cleaned_df similarly
    for c in cleaned_features:
        if c in cleaned_df.columns and not pd.api.types.is_numeric_dtype(cleaned_df[c]):
            cleaned_df[c] = pd.factorize(cleaned_df[c])[0]

    # Retrain on cleaned features
    print('Running cleaned CV with Ridge...')
    y_clean = target_series(cleaned_df)
    non_null_mask_clean = y_clean.notna()
    if non_null_mask_clean.sum() < len(y_clean):
        missing = len(y_clean) - int(non_null_mask_clean.sum())
        print(f'Note: {missing} rows have missing target in cleaned_df and will be excluded from CV')
        cleaned_df = cleaned_df.loc[non_null_mask_clean].reset_index(drop=True)
        y_clean = y_clean.loc[non_null_mask_clean].reset_index(drop=True)
    
    # Ridge on cleaned features
    cleaned_metrics, imp_cleaned = time_series_cv_metrics(cleaned_df, y_clean, cleaned_features, alpha=best_alpha)
    imp_cleaned.to_csv(IMP_CLEAN, header=['importance'])
    
    # XGBoost on cleaned features
    print('Running cleaned CV with XGBoost...')
    if best_xgb_params is not None:
        try:
            cleaned_xgb_metrics, imp_xgb_cleaned = xgboost_cv_metrics(cleaned_df, y_clean, cleaned_features, **best_xgb_params)
            imp_xgb_cleaned.to_csv(OUT_DIR / 'feature_importances_xgb_cleaned.csv', header=['importance'])
        except Exception as e:
            print(f'XGBoost cleaned training failed: {e}')
            cleaned_xgb_metrics = {'mae': float('nan'), 'rmse': float('nan'), 'mape': float('nan')}
            imp_xgb_cleaned = pd.Series(dtype=float)
    else:
        cleaned_xgb_metrics = {'mae': float('nan'), 'rmse': float('nan'), 'mape': float('nan')}
        imp_xgb_cleaned = pd.Series(dtype=float)

    # Save report
    report = {
        'alpha_grid': list(ALPHA_GRID),
        'best_alpha': best_alpha,
        'alpha_grid_results': alpha_results,
        'baseline_metrics_ridge': baseline_metrics,
        'cleaned_metrics_ridge': cleaned_metrics,
        'best_xgb_params': best_xgb_params,
        'xgb_grid_results': xgb_results,
        'baseline_metrics_xgb': baseline_xgb_metrics,
        'cleaned_metrics_xgb': cleaned_xgb_metrics,
        'dropped_features': drop_list,
        'rare_wcodes_grouped': rare_wcodes,
        'feature_count_before': len(features),
        'feature_count_after': len(cleaned_features)
    }
    with open(REPORT, 'w') as f:
        f.write('ML Pipeline Report\n')
        f.write('==================\n\n')
        
        # Ridge results
        f.write('Ridge Regression Results:\n')
        f.write('-' * 30 + '\n')
        f.write(f'Alpha grid: {ALPHA_GRID}\n')
        f.write(f'Best alpha selected: {best_alpha} (by CV RMSE)\n')
        f.write('\nBaseline Ridge metrics (CV mean):\n')
        f.write(json.dumps(baseline_metrics, indent=2))
        f.write('\n\nCleaned Ridge metrics (CV mean, Top-30 preserved):\n')
        f.write(json.dumps(cleaned_metrics, indent=2))
        
        # XGBoost results
        f.write('\n\nXGBoost Results:\n')
        f.write('-' * 30 + '\n')
        if best_xgb_params is not None:
            f.write(f'Best XGBoost params: {best_xgb_params}\n')
            f.write('\nBaseline XGBoost metrics (CV mean):\n')
            f.write(json.dumps(baseline_xgb_metrics, indent=2))
            f.write('\n\nCleaned XGBoost metrics (CV mean):\n')
            f.write(json.dumps(cleaned_xgb_metrics, indent=2))
        else:
            f.write('XGBoost training failed\n')
        
        # Model comparison
        f.write('\n\nModel Comparison (RMSE):\n')
        f.write('-' * 30 + '\n')
        f.write(f'Ridge Baseline: {baseline_metrics.get("rmse", "N/A"):.4f}\n')
        f.write(f'Ridge Cleaned:  {cleaned_metrics.get("rmse", "N/A"):.4f}\n')
        if best_xgb_params is not None:
            f.write(f'XGBoost Baseline: {baseline_xgb_metrics.get("rmse", "N/A"):.4f}\n')
            f.write(f'XGBoost Cleaned:  {cleaned_xgb_metrics.get("rmse", "N/A"):.4f}\n')
        
        # Feature engineering details
        f.write('\n\nFeature Engineering:\n')
        f.write('-' * 30 + '\n')
        f.write('Dropped features:\n')
        for c in drop_list:
            f.write(f'- {c}\n')
        f.write('\nRare weather-code columns grouped:\n')
        for c in rare_wcodes:
            f.write(f'- {c}\n')
        
        # Feature importances
        f.write('\n\nFeature Importances (Ridge baseline top 20):\n')
        f.write(imp_baseline.head(20).to_string())
        f.write('\n\nFeature Importances (Ridge cleaned top 20):\n')
        f.write(imp_cleaned.head(20).to_string())
        
        if best_xgb_params is not None and not imp_xgb_baseline.empty:
            f.write('\n\nFeature Importances (XGBoost baseline top 20):\n')
            f.write(imp_xgb_baseline.head(20).to_string())
            if not imp_xgb_cleaned.empty:
                f.write('\n\nFeature Importances (XGBoost cleaned top 20):\n')
                f.write(imp_xgb_cleaned.head(20).to_string())

    # Save metrics JSON
    with open(METRICS, 'w') as f:
        json.dump(report, f, indent=2)

    print('Report written to', REPORT)
    print('Importance CSVs written to', IMP_BASE, IMP_CLEAN)
    print('Metrics summary written to', METRICS)

if __name__ == '__main__':
    run_pipeline()
