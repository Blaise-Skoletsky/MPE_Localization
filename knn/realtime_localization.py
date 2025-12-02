"""Realtime localization helper for streaming wifi+light data.

Workflow:
1. Fit a KNN regressor once using the same 0.4 real / 0.6 synthetic mix as kNN_Model.py.
2. Provide a `RealtimeLocalizer` object that exposes `predict_sample(sample_dict)` so callers (e.g.,
    Raspberry Pi streaming code) can pass raw measurements directly without CSV intermediaries.
3. Each call filters the measurement down to the wifi_* / light_* columns the model expects,
    skipping missing entries, and immediately returns the estimated (x, y).
4. Optional live plotting utilities accumulate predictions and render the trajectory in matplotlib.
5. Demo helpers remain available but the core API is now usable from any external data gatherer.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kNN_Model import build_feature_list, fit_knn_model, prepare_datasets, knn_predict_point, load_meta


def train_realtime_model(
    *,
    real_fraction: float,
    n_neighbors: int,
) -> Tuple[object, List[str]]:
    """Fit the KNN model using the specified real/synthetic split."""
    random_seed = 42
    features = build_feature_list()
    train_df, _, _ = prepare_datasets(
        real_fraction=real_fraction,
        test_size=0.2,
        val_size=0.2,
        train_size=None,
        random_seed=random_seed,
    )
    X = train_df[features].to_numpy(dtype=float)
    y = train_df[['x', 'y']].to_numpy(dtype=float)
    print(
        f"Training realtime KNN with {len(X)} samples | "
        f"features={len(features)} | real_fraction={real_fraction:.2f}"
    )
    model = fit_knn_model(X, y, n_neighbors=n_neighbors)
    return model, features


def filter_sample(sample: Dict[str, float], feature_order: List[str]) -> np.ndarray:
    """Extract required wifi_* and light_* readings, returning them as a vector.

    If a feature is missing or null it will be replaced with 0.0 (instead of skipping the sample).
    This keeps the stream robust when some sensors are unavailable.
    """
    values: List[float] = []
    for feat in feature_order:
        if feat not in sample or sample[feat] is None:
            # substitute missing features with 0.0
            values.append(0.0)
        else:
            values.append(float(sample[feat]))
    return np.asarray(values, dtype=float)


class RealtimeLocalizer:
    """Encapsulates the KNN model for realtime predictions"""

    def __init__(
        self,
        *,
        real_fraction: float = 0.4,
        n_neighbors: int = 4,
        enable_plot: bool = True,
    ) -> None:
        self.model, self.feature_order = train_realtime_model(
            real_fraction=real_fraction,
            n_neighbors=n_neighbors,
        )
        self.enable_plot = enable_plot
        self._xs: List[float] = []
        self._ys: List[float] = []
        self._scatter = None
        self._base_dir = os.path.dirname(os.path.abspath(__file__))
        self._data_dir = os.path.join(self._base_dir, 'data')
        self._models_dir = os.path.join(self._base_dir, 'models')
        self._wifi_pre_cols: List[str] = []
        self._wifi_post_cols: List[str] = []
        self._light_cols: List[str] = []
        self._wifi_scaler_means: Dict[str, float] = {}
        self._wifi_scaler_stds: Dict[str, float] = {}
        self._wifi_pca = None
        self._wifi_global_min: Optional[float] = None
        self._wifi_global_max: Optional[float] = None
        self._light_scaler_stats: Dict[str, Dict[str, float]] = {}
        self._load_preprocessing_assets()
        if enable_plot:
            self._init_plot()
        print(
            f"RealtimeLocalizer ready | features={len(self.feature_order)} | "
            f"plot={'on' if self.enable_plot else 'off'}"
        )

    def _init_plot(self) -> None:
        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(6, 6))
        self._ax.set_title('Realtime Localization')
        self._ax.set_xlabel('X (cm)')
        self._ax.set_ylabel('Y (cm)')
        self._ax.set_xlim(250, 750)
        self._ax.set_ylim(150, 600)
        self._scatter = self._ax.scatter([], [], c='tab:blue', s=25)

    def predict_sample(self, sample: Dict[str, float], update_plot: bool = True) -> Optional[Tuple[float, float]]:
        """Filter a raw measurement, predict (x,y), and optionally update the live plot."""
        vector = filter_sample(sample, self.feature_order)
        if vector is None:
            print("Skipping sample: missing required feature(s)")
            return None
        pred = self.model.predict(vector.reshape(1, -1))[0]
        x, y = float(pred[0]), float(pred[1])
        print(f"Predicted position -> x={x:.2f}, y={y:.2f}")
        if self.enable_plot and update_plot:
            self._xs.append(x)
            self._ys.append(y)
            assert self._scatter is not None
            self._scatter.set_offsets(np.column_stack((self._xs, self._ys)))
            self._fig.canvas.draw_idle()
            plt.pause(0.01)
        return x, y

    def predict_from_raw_combined(
        self,
        raw_sample: Dict[str, float],
        update_plot: bool = True,
    ) -> Optional[Tuple[float, float]]:
        """Preprocess a single dict containing raw wifi+light readings, then predict."""
        processed = self._preprocess_measurements(raw_sample, raw_sample)
        if processed is None:
            print("Skipping sample: unable to preprocess combined raw inputs")
            return None
        return self.predict_sample(processed, update_plot=update_plot)

    def predict_from_raw_modalities(
        self,
        wifi_sample: Dict[str, float],
        light_sample: Dict[str, float],
        update_plot: bool = False,
    ) -> Optional[Tuple[float, float]]:
        """Preprocess separate wifi/light dicts before running the realtime predictor."""
        processed = self._preprocess_measurements(wifi_sample, light_sample)
        if processed is None:
            print("Skipping sample: unable to preprocess modality-specific raw inputs")
            return None
        return self.predict_sample(processed, update_plot=update_plot)

    # --- Preprocessing helpers -------------------------------------------------

    def _load_preprocessing_assets(self) -> None:
        try:
            self._wifi_pre_cols = load_meta('wifi', subset='pre')
        except Exception:
            self._wifi_pre_cols = []
        try:
            self._wifi_post_cols = load_meta('wifi', subset='post')
        except Exception:
            self._wifi_post_cols = []
        try:
            self._light_cols = load_meta('light')
        except Exception:
            self._light_cols = []
        (
            self._wifi_scaler_means,
            self._wifi_scaler_stds,
            self._wifi_global_min,
            self._wifi_global_max,
        ) = self._load_wifi_scaler()
        self._wifi_pca = self._load_wifi_pca()
        self._light_scaler_stats = self._load_light_scaler()

    def _load_wifi_scaler(self) -> Tuple[Dict[str, float], Dict[str, float], Optional[float], Optional[float]]:
        scaler_path = os.path.join(self._models_dir, 'spatial_wifi_scaler.pkl')
        if not os.path.exists(scaler_path):
            print(f"Warning: missing wifi scaler at {scaler_path}; raw predictions may be inaccurate")
            return {}, {}, None, None
        try:
            with open(scaler_path, 'rb') as sf:
                data = pickle.load(sf)
            means = {str(k): float(v) for k, v in data.get('means', {}).items()}
            stds = {str(k): (float(v) if float(v) != 0 else 1.0) for k, v in data.get('stds', {}).items()}
            gmin = data.get('global_min')
            gmax = data.get('global_max')
            return means, stds, (float(gmin) if gmin is not None else None), (float(gmax) if gmax is not None else None)
        except Exception as exc:
            print(f"Warning: failed to load wifi scaler ({exc}); raw predictions may be inaccurate")
            return {}, {}, None, None

    def _load_wifi_pca(self):
        pca_path = os.path.join(self._models_dir, 'spatial_wifi_pca.pkl')
        if not os.path.exists(pca_path):
            print(f"Warning: missing wifi PCA at {pca_path}; predictor will use pre-PCA wifi features")
            return None
        try:
            with open(pca_path, 'rb') as pf:
                return pickle.load(pf)
        except Exception as exc:
            print(f"Warning: failed to load wifi PCA ({exc}); predictor will use pre-PCA wifi features")
            return None

    def _load_light_scaler(self) -> Dict[str, Dict[str, float]]:
        scaler_path = os.path.join(self._models_dir, 'spatial_light_scaler.pkl')
        if not os.path.exists(scaler_path):
            print(f"Warning: missing light scaler at {scaler_path}; light preprocessing will be skipped")
            return {}
        try:
            with open(scaler_path, 'rb') as lf:
                data = pickle.load(lf)
        except Exception as exc:
            print(f"Warning: failed to load light scaler ({exc}); light preprocessing will be skipped")
            return {}
        stats: Dict[str, Dict[str, float]] = {}
        mins = data.get('mins', {})
        maxs = data.get('maxs', {})
        means = data.get('means', {})
        stds = data.get('stds', {})
        for col in self._light_cols:
            stats[col] = {
                'min': float(mins.get(col, 0.0)),
                'max': float(maxs.get(col, 1.0)),
                'mean': float(means.get(col, 0.0)),
                'std': float(stds.get(col, 1.0)) or 1.0,
            }
        return stats

    def _lookup_feature_value(self, sample: Dict[str, float], key: str, prefix: str) -> Optional[float]:
        def _as_float(value: Optional[float]) -> Optional[float]:
            if value is None:
                return None
            try:
                fval = float(value)
            except (TypeError, ValueError):
                return None
            if np.isnan(fval):
                return None
            return fval

        if key in sample:
            val = _as_float(sample[key])
            if val is not None:
                return val
        prefixed = f'{prefix}_{key}' if not key.startswith(prefix + '_') else key
        if prefixed in sample:
            val = _as_float(sample[prefixed])
            if val is not None:
                return val
        if key.startswith(prefix + '_') and key in sample:
            val = _as_float(sample[key])
            if val is not None:
                return val
        return None

    def _preprocess_measurements(
        self,
        wifi_sample: Optional[Dict[str, float]],
        light_sample: Optional[Dict[str, float]],
    ) -> Optional[Dict[str, float]]:
        wifi_sample = wifi_sample or {}
        light_sample = light_sample or {}
        if not wifi_sample and not light_sample:
            return None

        processed: Dict[str, float] = {}

        # WiFi preprocessing
        if self._wifi_pre_cols:
            wifi_min = self._wifi_global_min if self._wifi_global_min is not None else 0.0
            wifi_max = self._wifi_global_max if self._wifi_global_max is not None else wifi_min + 1.0
            denom = wifi_max - wifi_min or 1.0
            wifi_values: List[float] = []
            for col in self._wifi_pre_cols:
                val = self._lookup_feature_value(wifi_sample, col, 'wifi')
                if val is None:
                    val = wifi_min
                val = (val - wifi_min) / denom
                mean = self._wifi_scaler_means.get(col)
                std = self._wifi_scaler_stds.get(col)
                if mean is not None and std is not None and std != 0:
                    val = (val - mean) / std
                wifi_values.append(float(val))
            if wifi_values:
                vector = np.asarray(wifi_values, dtype=float).reshape(1, -1)
                if self._wifi_pca is not None and self._wifi_post_cols:
                    try:
                        comps = self._wifi_pca.transform(vector)[0]
                    except Exception as exc:
                        print(f"Warning: PCA transform failed ({exc}); falling back to pre-PCA wifi features")
                        comps = vector[0]
                    for idx in range(min(len(comps), len(self._wifi_post_cols))):
                        name = self._wifi_post_cols[idx]
                        base = name[len('wifi_'):] if name.startswith('wifi_') else name
                        processed[f'wifi_{base}'] = float(comps[idx])
                else:
                    for idx, col in enumerate(self._wifi_pre_cols):
                        processed[f'wifi_{col}'] = float(vector[0, idx])

        # Light preprocessing (use light_sample when provided, otherwise fall back to wifi_sample that may contain light prefixed keys)
        light_source = light_sample if light_sample else wifi_sample
        if self._light_cols:
            for col in self._light_cols:
                val = self._lookup_feature_value(light_source, col, 'light') if light_source else None
                stats = self._light_scaler_stats.get(col)
                if stats:
                    fill = stats['min']
                    denom = stats['max'] - stats['min'] or 1.0
                    numeric = fill if val is None else float(val)
                    normalized = (numeric - stats['min']) / denom
                    std = stats['std'] if stats['std'] != 0 else 1.0
                    scaled = (normalized - stats['mean']) / std
                else:
                    scaled = 0.0 if val is None else float(val)
                base = col[len('light_'):] if col.startswith('light_') else col
                processed[f'light_{base}'] = float(scaled)

        return processed if processed else None

    def close_plot(self) -> None:
        if self.enable_plot:
            plt.ioff()
            plt.show()




def main():
    """Example entry point showing how to run predictions on raw wifi/light CSVs."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wifi_path = os.path.join(base_dir, 'data', 'wifi.csv')
    light_path = os.path.join(base_dir, 'data', 'light.csv')
    localizer = RealtimeLocalizer(enable_plot=True, real_fraction=0.4, n_neighbors=15)

    wifi_df = pd.read_csv(wifi_path)
    light_df = pd.read_csv(light_path)
    if len(wifi_df) != len(light_df):
        print(
            f"Warning: wifi rows={len(wifi_df)} != light rows={len(light_df)}; "
            "pairing by index up to min length"
        )
    n = min(len(wifi_df), len(light_df))
    print(f"Running realtime predictions on {n} paired raw samples...")

    for idx in range(n):
        wifi_row = wifi_df.iloc[idx].to_dict()
        light_row = light_df.iloc[idx].to_dict()
        pred = localizer.predict_from_raw_modalities(wifi_row, light_row)
        if pred is None:
            print(f"Row {idx}: prediction skipped")
        else:
            x, y = pred
            print(f"Row {idx}: x={x:.2f}, y={y:.2f}")

    localizer.close_plot()


if __name__ == '__main__':
    main()