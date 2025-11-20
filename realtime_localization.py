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
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kNN_Model import _build_feature_list, fit_knn_model, prepare_datasets


def train_realtime_model(
    *,
    real_fraction: float = 0.4,
    n_neighbors: int = 7,
    random_seed: int = 42,
) -> Tuple[object, List[str]]:
    """Fit the KNN model using the specified real/synthetic split."""
    features = _build_feature_list()
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


def filter_sample(sample: Dict[str, float], feature_order: List[str]) -> Optional[np.ndarray]:
    """Keep only required wifi_* and light_* readings, returning them as a vector.

    Missing features invalidate the sample; return None so the caller can skip.
    """
    values: List[float] = []
    for feat in feature_order:
        if feat not in sample or sample[feat] is None:
            return None
        values.append(float(sample[feat]))
    return np.asarray(values, dtype=float)


class RealtimeLocalizer:
    """Encapsulates the KNN model for realtime predictions"""

    def __init__(
        self,
        *,
        real_fraction: float = 0.4,
        n_neighbors: int = 7,
        random_seed: int = 42,
        enable_plot: bool = True,
        verbose: bool = True,
    ) -> None:
        self.model, self.feature_order = train_realtime_model(
            real_fraction=real_fraction,
            n_neighbors=n_neighbors,
            random_seed=random_seed,
        )
        self.enable_plot = enable_plot
        self.verbose = verbose
        self._xs: List[float] = []
        self._ys: List[float] = []
        self._scatter = None
        if enable_plot:
            self._init_plot()
        if self.verbose:
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
            if self.verbose:
                print("Skipping sample: missing required feature(s)")
            return None
        pred = self.model.predict(vector.reshape(1, -1))[0]
        x, y = float(pred[0]), float(pred[1])
        if self.verbose:
            print(f"Predicted position -> x={x:.2f}, y={y:.2f}")
        if self.enable_plot and update_plot:
            self._xs.append(x)
            self._ys.append(y)
            assert self._scatter is not None
            self._scatter.set_offsets(np.column_stack((self._xs, self._ys)))
            self._fig.canvas.draw_idle()
            plt.pause(0.01)
        return x, y

    def close_plot(self) -> None:
        if self.enable_plot:
            plt.ioff()
            plt.show()


def stream_localization(sample_stream: Iterable[Dict[str, float]], localizer: RealtimeLocalizer) -> None:
    """Convenience helper for running a generator through the localizer with plotting."""
    for sample in sample_stream:
        localizer.predict_sample(sample)
    localizer.close_plot()


def load_stream_from_csv(path: str) -> Iterable[Dict[str, float]]:
    """Example helper: iterate rows from a CSV as sample dicts."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    for _, row in df.iterrows():
        yield row.to_dict()


def load_stream_from_two_csvs(wifi_path: str, light_path: str) -> Iterable[Dict[str, float]]:
    """Iterate synchronized rows from wifi and light CSVs and yield a combined sample dict.

    Each yielded dict uses `wifi_<col>` and `light_<col>` keys matching model feature names.
    Rows are paired by their DataFrame index; lengths should be equal for the demo data.
    """
    wifi_df = pd.read_csv(wifi_path)
    light_df = pd.read_csv(light_path)
    if len(wifi_df) != len(light_df):
        print(f"Warning: wifi rows={len(wifi_df)} != light rows={len(light_df)}; pairing by index up to min length")
    n = min(len(wifi_df), len(light_df))
    print(f"Loaded and combining {n} rows from: {wifi_path} + {light_path}")
    for i in range(n):
        wrow = wifi_df.iloc[i]
        lrow = light_df.iloc[i]
        sample: Dict[str, float] = {}
        # prefix wifi columns
        for c in wifi_df.columns:
            if c in ('timestamp', 'x', 'y'):
                continue
            sample[f'wifi_{c}'] = wrow[c]
        # prefix light columns
        for c in light_df.columns:
            if c in ('timestamp', 'x', 'y'):
                continue
            sample[f'light_{c}'] = lrow[c]
        # include coordinates if present
        sample['x'] = float(wrow['x']) if 'x' in wifi_df.columns else float(lrow['x'])
        sample['y'] = float(wrow['y']) if 'y' in wifi_df.columns else float(lrow['y'])
        yield sample


def main():
    # Demo only: in real deployment the Raspberry Pi code would import this file,
    # instantiate RealtimeLocalizer, and call predict_sample(sample_dict) directly.
    localizer = RealtimeLocalizer(enable_plot=True, verbose=True)
    demo_stream = load_stream_from_two_csvs(os.path.join('data', 'wifi_cleaned.csv'), os.path.join('data', 'light_cleaned.csv'))
    stream_localization(demo_stream, localizer)


if __name__ == '__main__':
    main()