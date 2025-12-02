"""Generates synthetic data from model, saves it to csv"""

import json
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import gprlib
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())


def load_model_and_likelihood(kind: str) -> Tuple[object, object, List[str], List[str], torch.Tensor, torch.Tensor]:
    """Loads trained GPR model & likelihood for each type (models are trained independently) Given a parameter: kind, which can be wifi or light"""
   
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    meta_path = os.path.join(model_dir, f'spatial_{kind}_meta.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found for {kind}: {meta_path}")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    if kind == 'wifi':
        pre_cols = meta.get('pre_pca', {}).get('columns') or meta.get('columns') or meta.get('output_cols')
        post_cols = meta.get('post_pca', {}).get('columns') or pre_cols
    else:
        pre_cols = meta.get('columns') or meta.get('output_cols')
        post_cols = pre_cols
    if pre_cols is None:
        raise ValueError(f"Unable to determine columns for {kind} from {meta_path}")

    train_x_path = os.path.join(model_dir, f'spatial_{kind}_train_x.pt')
    train_y_path = os.path.join(model_dir, f'spatial_{kind}_train_y.pt')

    train_x = torch.load(train_x_path, map_location=device)
    train_y = torch.load(train_y_path, map_location=device)

    likelihood = gprlib.likelihoods.MultitaskGaussianLikelihood(num_tasks=len(pre_cols))
    mean = gprlib.means.MultitaskMean(gprlib.means.ZeroMean(), num_tasks=len(pre_cols))
    spatial_base = gprlib.kernels.MaternKernel(nu=2.5, ard=True)
    kernel = gprlib.kernels.MultitaskKernel(spatial_base, num_tasks=len(pre_cols), rank=min(5, len(pre_cols)))
    distribution = gprlib.distributions.MultitaskMultivariateNormal
    model = gprlib.ExactGP(train_x, train_y, likelihood, mean, kernel, distribution)

    model_path = os.path.join(model_dir, f'spatial_{kind}_model.pt')
    like_path = os.path.join(model_dir, f'spatial_{kind}_likelihood.pt')
    if not os.path.exists(model_path) or not os.path.exists(like_path):
        raise FileNotFoundError(f"Model or likelihood not found for {kind}; run train_model first.")

    # Load state dicts onto the chosen device and move model/likelihood there
    model.load_state_dict(torch.load(model_path, map_location=device))
    likelihood.load_state_dict(torch.load(like_path, map_location=device))
    model.to(device)
    likelihood.to(device)
    model.eval()
    likelihood.eval()

    return model, likelihood, pre_cols, post_cols


def randomly_generate_x_y_coordinates(n: int = 2000, x_range: Tuple[float, float] = (300, 700), y_range: Tuple[float, float] = (175, 575), seed: int = 42) -> List[Tuple[float, float]]:
    """Generate `n` (x,y) locations uniformly in the given ranges."""
    np.random.seed(seed)
    xs = np.random.uniform(x_range[0], x_range[1], size=n)
    ys = np.random.uniform(y_range[0], y_range[1], size=n)
    return list(zip(xs, ys))


def main():
    """Script to generate syntheic data points using trained GPR models."""


    # Default output path
    out = os.path.join('synthetic_data', 'synthetic_knn_points.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"Using device: {device}")

    # Load models & likelihoods (these will raise if still missing)
    wifi_model, wifi_like, wifi_pre_cols, wifi_post_cols = load_model_and_likelihood('wifi')
    light_model, light_like, light_pre_cols, light_post_cols = load_model_and_likelihood('light')

    # Attempt to load a saved WiFi PCA transformer (produced by preprocessing)
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    pca_path = os.path.join(model_dir, 'spatial_wifi_pca.pkl')
    pca = None
    if os.path.exists(pca_path):
        try:
            with open(pca_path, 'rb') as pf:
                pca = pickle.load(pf)
            print(f"Loaded WiFi PCA transformer from {pca_path}")
        except Exception as e:
            print(f"Warning: failed to load WiFi PCA transformer: {e}")

    # Load post-PCA meta for naming output columns that downstream (kNN) expects
    points = randomly_generate_x_y_coordinates(n=2000)
    rows = []
    print(f"Generating {len(points)} synthetic points and querying GP models — this may take a while.")
    for x, y in tqdm(points, desc='Generating'):
        xy = torch.tensor([[float(x), float(y)]], dtype=torch.float32, device=device)
        with torch.no_grad():
            post_w = wifi_like(wifi_model(xy))
            vec_w = post_w.mean.detach().cpu().numpy().reshape(-1)
            post_l = light_like(light_model(xy))
            vec_l = post_l.mean.detach().cpu().numpy().reshape(-1)

        entry = {}

        # Project into PCA space when possible
        expected_dim = None
        if pca is not None:
            expected_dim = (
                getattr(pca, 'n_features_in_', None)
                or getattr(pca, 'n_features_', None)
                or (pca.components_.shape[1] if hasattr(pca, 'components_') else None)
                or (len(pca.mean_) if hasattr(pca, 'mean_') else None)
            )

        if pca is not None and expected_dim == len(vec_w):
            comps = pca.transform(vec_w.reshape(1, -1))[0]
            target_cols = wifi_post_cols if wifi_post_cols and len(wifi_post_cols) == len(comps) else [f'wifi_pca_{i}' for i in range(len(comps))]
            for name, val in zip(target_cols, comps):
                base = name[len('wifi_'):] if str(name).startswith('wifi_') else name
                entry[f'wifi_{base}'] = float(val)
        else:
            if pca is None:
                reason = 'missing PCA transformer file (models/spatial_wifi_pca.pkl).'
            else:
                reason = f'vector dim {len(vec_w)} vs PCA expected {expected_dim}'
            print(f"Warning: WiFi PCA transformer missing or incompatible ({reason}); synthetic data will not include PCA columns.")

        # Light outputs (no PCA) — prefix consistently
        for col_name, value in zip(light_post_cols, vec_l):
            base = col_name
            if base.startswith('light_'):
                base = base[len('light_'):]
            entry[f'light_{base}'] = float(value)

        entry['x'] = float(x)
        entry['y'] = float(y)
        rows.append(entry)

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} synthetic points to {out}")


if __name__ == '__main__':
    main()
