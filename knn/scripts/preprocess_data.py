from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import pickle

KNN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(KNN_ROOT, 'data')
MODELS_DIR = os.path.join(KNN_ROOT, 'models')

# PCA configuration: apply PCA on wifi only using a variance threshold.
# Set to a float in (0,1] to retain that fraction of variance (e.g. 0.95),
# or set to 0 to disable PCA.
PCA_VARIANCE_THRESHOLD = 0.90

def determine_columns_based_on_nulls(dataframe: pd.DataFrame, percentage: float = 0.8) -> Tuple[List[str], Dict[str, float]]:
    """Based on the dataframe, only choose columns that have at least `percentage` non-null values."""
    n = len(dataframe)
    if n == 0:
        return [], {}
    # Series of null counts per column
    null_counts = dataframe.isnull().sum()
    # Fraction of nulls (0-1)
    null_frac = null_counts / n
    # Fraction of non-nulls (0-1)
    non_null_frac = 1.0 - null_frac

    threshold_frac = float(percentage)
    selected_columns: List[str] = [col for col in dataframe.columns if non_null_frac.get(col, 0.0) >= threshold_frac]

    # Only return null fractions for selected columns, rounded to 4 decimals
    null_percentages: Dict[str, float] = {col: round(float(null_frac[col]), 4) for col in selected_columns}

    return selected_columns, null_percentages

def normalize_wifi_global(wifi_df: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    """Normalize wifi columns using dataset-wide min/max and return those stats."""
    wifi_cols = [c for c in wifi_df.columns if c not in ('timestamp', 'x', 'y')]
    # Compute global min/max across all selected wifi columns (skip NaNs)
    global_min = wifi_df[wifi_cols].min().min(skipna=True)
    global_max = wifi_df[wifi_cols].max().max(skipna=True)
    if pd.isna(global_min):
        global_min = 0.0
    if pd.isna(global_max):
        global_max = global_min + 1.0
    denom = float(global_max - global_min) if global_max is not None and global_min is not None else 1.0
    if denom == 0:
        denom = 1.0

    # Replace nulls with dataset minimum
    wifi_df[wifi_cols] = wifi_df[wifi_cols].fillna(global_min)
    wifi_df[wifi_cols] = (wifi_df[wifi_cols] - global_min) / denom
    return wifi_df, float(global_min), float(global_max)

def normalize_light(light_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Normalize light columns using per-column min/max and return stats."""
    light_cols = [c for c in light_df.columns if c not in ('timestamp', 'x', 'y')]
    stats: Dict[str, Dict[str, float]] = {}
    for col in light_cols:
        col_min = light_df[col].min(skipna=True)
        col_max = light_df[col].max(skipna=True)
        if pd.isna(col_min):
            col_min = 0.0
        if pd.isna(col_max):
            col_max = col_min + 1.0
        denom = float(col_max - col_min) if col_max is not None and col_min is not None else 1.0
        if denom == 0:
            denom = 1.0
        # Replace nulls with column minimum
        light_df[col] = light_df[col].fillna(col_min)
        light_df[col] = (light_df[col] - col_min) / denom
        stats[col] = {'min': float(col_min) if col_min is not None else 0.0, 'max': float(col_max) if col_max is not None else 0.0}
    return light_df, stats

def standardize_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Standardize features to zero mean and unit variance per column."""

    if exclude_cols is None:
        exclude_cols = ['timestamp', 'x', 'y']

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in exclude_cols]

    out = df.copy()
    if not feature_cols:
        return out, pd.Series(dtype=float), pd.Series(dtype=float)

    means = out[feature_cols].mean(skipna=True)
    stds = out[feature_cols].std(skipna=True, ddof=0)
    stds = stds.replace(0, 1.0).fillna(1.0)

    out[feature_cols] = (out[feature_cols] - means) / stds
    return out, means, stds


def main() -> None:
    """
    1) Loads the wifi + light data
    2) Drops wifi columns with too many nulls (95 percent threshold)
    3) Normalizes wifi (global min/max) and light (per-column min/max)
    4) Standardizes both datasets (zero mean, unit variance)
    5) Applies PCA on wifi features using variance threshold
    6) Saves metadata of PCA (saves the algorithm so that it can be applied to new data.)
    7) Saves cleaned datasets into 3 files: 1) wifi_cleaned.csv (cleaned wifi data after PCA), 2) wifi_pre_pca.csv (cleaned wifi data before PCA), 3) light_cleaned.csv (cleaned light data)
    """
    output_dir = DATA_DIR
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    wifi_data = os.path.join(DATA_DIR, 'wifi.csv')
    light_data = os.path.join(DATA_DIR, 'light.csv')

    wifi_df = pd.read_csv(wifi_data)
    light_df = pd.read_csv(light_data)

    wifi_selected_cols, wifi_nulls = determine_columns_based_on_nulls(wifi_df, percentage=0.95)
    print(f"Total wifi columns before selection: {len(wifi_df.columns)}")
    print(f"Selected {len(wifi_selected_cols)} wifi columns based on null threshold:")
    for col in wifi_selected_cols:
        print(f"  {col}: {wifi_nulls[col]*100:.2f}% nulls")


    wifi_df = wifi_df[wifi_selected_cols]
    wifi_normalized, wifi_global_min, wifi_global_max = normalize_wifi_global(wifi_df.copy())
    light_normalized, light_norm_stats = normalize_light(light_df.copy())

    # Standardize (zero mean, unit variance) for both datasets, excluding coordinates/timestamp
    wifi_feature_cols = [c for c in wifi_normalized.columns if c not in ('timestamp', 'x', 'y')]
    light_feature_cols = [c for c in light_normalized.columns if c not in ('timestamp', 'x', 'y')]

    wifi_standardized, wifi_means, wifi_stds = standardize_features(
        wifi_normalized,
        feature_cols=wifi_feature_cols,
        exclude_cols=['timestamp', 'x', 'y'],
    )
    light_standardized, light_means, light_stds = standardize_features(
        light_normalized,
        feature_cols=light_feature_cols,
        exclude_cols=['timestamp', 'x', 'y'],
    )


    wifi_out = os.path.join(output_dir, 'wifi_cleaned.csv')
    light_out = os.path.join(output_dir, 'light_cleaned.csv')

    # By default save the standardized per-modality files. We will optionally
    # run PCA on the WiFi features only, using a variance threshold.
    # Helper to write light meta (no PCA)
    def write_light_meta(output_cols: List[str]):
        meta = {'columns': output_cols, 'num_columns': len(output_cols)}
        path = os.path.join(MODELS_DIR, 'spatial_light_meta.json')
        with open(path, 'w') as f:
            json.dump(meta, f, indent=2)

    # Track wifi metadata for both pre- and post-PCA representations
    wifi_meta = {
        'pre_pca': {
            'columns': wifi_feature_cols,
            'num_columns': len(wifi_feature_cols),
        },
        'post_pca': None,
        'pca': {
            'enabled': bool(PCA_VARIANCE_THRESHOLD and 0 < PCA_VARIANCE_THRESHOLD <= 1.0),
            'variance_threshold': PCA_VARIANCE_THRESHOLD if PCA_VARIANCE_THRESHOLD else None,
            'components_retained': None,
            'explained_variance_cumulative': None,
        },
    }

    # WiFi PCA using variance threshold if requested
    # Save pre-PCA standardized wifi for downstream training (GPs expect pre-PCA inputs)
    wifi_pre_pca_out = os.path.join(output_dir, 'wifi_pre_pca.csv')
    try:
        wifi_standardized.to_csv(wifi_pre_pca_out, index=False)
        # save scaler (means/stds) so other scripts can standardize the same way
        scaler_path = os.path.join(MODELS_DIR, 'spatial_wifi_scaler.pkl')
        try:
            with open(scaler_path, 'wb') as sf:
                payload = {
                    'means': {k: float(v) for k, v in wifi_means.to_dict().items()},
                    'stds': {k: float(v) for k, v in wifi_stds.to_dict().items()},
                    'global_min': float(wifi_global_min) if wifi_global_min is not None else None,
                    'global_max': float(wifi_global_max) if wifi_global_max is not None else None,
                }
                pickle.dump(payload, sf)
            print(f"Saved pre-PCA wifi standardized CSV -> {wifi_pre_pca_out} and scaler -> {scaler_path}")
        except Exception as e:
            print(f"Warning: failed to save wifi scaler: {e}")
    except Exception as e:
        print(f"Warning: failed to save pre-PCA wifi CSV: {e}")

    if PCA_VARIANCE_THRESHOLD and 0 < PCA_VARIANCE_THRESHOLD <= 1.0 and wifi_feature_cols:
        w_feat = wifi_standardized[wifi_feature_cols].fillna(0.0)
        pca_w = PCA(n_components=PCA_VARIANCE_THRESHOLD)
        w_pca_arr = pca_w.fit_transform(w_feat.values)
        n_components_w = w_pca_arr.shape[1]
        w_pca_cols = [f'wifi_pca_{i}' for i in range(n_components_w)]
        w_out_df = pd.DataFrame(w_pca_arr, columns=w_pca_cols)
        # attach coordinates if present
        if 'x' in wifi_standardized.columns and 'y' in wifi_standardized.columns:
            w_out_df.insert(0, 'y', wifi_standardized['y'].values)
            w_out_df.insert(0, 'x', wifi_standardized['x'].values)
        w_out_df.to_csv(wifi_out, index=False)
        wifi_meta['post_pca'] = {'columns': w_pca_cols, 'num_columns': n_components_w}
        wifi_meta['pca']['components_retained'] = n_components_w
        # Save the fitted PCA transformer so synthetic data can be projected into the same space
        pca_path = os.path.join(MODELS_DIR, 'spatial_wifi_pca.pkl')
        try:
            with open(pca_path, 'wb') as pf:
                pickle.dump(pca_w, pf)
            print(f"Saved WiFi PCA transformer -> {pca_path}")
        except Exception as e:
            print(f"Warning: failed to save PCA transformer: {e}")
        explained = pca_w.explained_variance_ratio_.cumsum()
        wifi_meta['pca']['explained_variance_cumulative'] = float(explained[-1]) if len(explained) else None
        print(f"Saved PCA-reduced wifi -> {wifi_out} (n_components={n_components_w}, cumvar={explained[-1]:.4f})")
    else:
        wifi_standardized.to_csv(wifi_out, index=False)
        wifi_meta['post_pca'] = dict(wifi_meta['pre_pca'])
        wifi_meta['pca'] = {
            'enabled': False,
            'variance_threshold': None,
            'components_retained': wifi_meta['pre_pca']['num_columns'],
            'explained_variance_cumulative': None,
        }
        print(f"Saved cleaned (standardized) wifi -> {wifi_out}")

    # Persist wifi meta (pre/post PCA info)
    wifi_meta_path = os.path.join(MODELS_DIR, 'spatial_wifi_meta.json')
    with open(wifi_meta_path, 'w') as wf:
        json.dump(wifi_meta, wf, indent=2)
    print(f"Saved WiFi metadata -> {wifi_meta_path}")

    # Always save light standardized (no PCA)
    light_standardized.to_csv(light_out, index=False)
    write_light_meta(light_feature_cols)

    # Save light normalization + standardization stats for realtime preprocessing
    light_scaler_path = os.path.join(MODELS_DIR, 'spatial_light_scaler.pkl')
    try:
        light_payload = {
            'mins': {col: light_norm_stats.get(col, {}).get('min', 0.0) for col in light_feature_cols},
            'maxs': {col: light_norm_stats.get(col, {}).get('max', 1.0) for col in light_feature_cols},
            'means': {k: float(v) for k, v in light_means.to_dict().items()},
            'stds': {k: float(v) if float(v) != 0 else 1.0 for k, v in light_stds.to_dict().items()},
        }
        with open(light_scaler_path, 'wb') as lf:
            pickle.dump(light_payload, lf)
        print(f"Saved light scaler stats -> {light_scaler_path}")
    except Exception as e:
        print(f"Warning: failed to save light scaler stats: {e}")

    print(f"Saved cleaned (standardized) light -> {light_out}")

if __name__ == '__main__':
    main()
