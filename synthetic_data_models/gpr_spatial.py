import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
try:
    from . import gprlib  # package context
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import gprlib

def train_model(data_type):
    print('Starting: ')
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    if data_type == 'wifi':
        dataset = pd.read_csv(os.path.join(data_dir, "wifi_cleaned.csv"))
    elif data_type == 'light':
        dataset = pd.read_csv(os.path.join(data_dir, "light_cleaned.csv"))
    else:
        raise ValueError("data_type must be either 'wifi' or 'light'")

    device_cols = [col for col in dataset.columns if col not in ['timestamp', 'x', 'y']]

    dataset_shuffled = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(0.8 * len(dataset_shuffled))
    train = dataset_shuffled.iloc[:split_idx]
    test = dataset_shuffled.iloc[split_idx:]

    train_x = torch.tensor(train[['x', 'y']].values, dtype=torch.float32)
    train_y = torch.tensor(train[device_cols].values, dtype=torch.float32)
    test_x = torch.tensor(test[['x', 'y']].values, dtype=torch.float32)
    test_y = torch.tensor(test[device_cols].values, dtype=torch.float32)

    likelihood = gprlib.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y.shape[1])
    mean = gprlib.means.MultitaskMean(gprlib.means.ZeroMean(), num_tasks=train_y.shape[1])
    spatial_base = gprlib.kernels.MaternKernel(nu=2.5, ard=True)
    kernel = gprlib.kernels.MultitaskKernel(spatial_base, num_tasks=train_y.shape[1], rank=5)
    distribution = gprlib.distributions.MultitaskMultivariateNormal
    model = gprlib.ExactGP(train_x, train_y, likelihood, mean, kernel, distribution)

    mll = gprlib.mlls.ExactMarginalLogLikelihood(likelihood, model)
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train(); likelihood.train()
    epochs = 1000 if data_type == 'wifi' else 2500 if data_type == 'light' else 1000
    rng = tqdm(range(epochs), desc="Training")
    for _ in rng:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        rng.set_postfix(loss=loss.item())
        optimizer.step()

    model.eval(); likelihood.eval()
    with torch.no_grad():
        preds = likelihood(model(test_x))
        mae = torch.mean(torch.abs(preds.mean - test_y)).item()
        print(f"Mean Absolute Error (MAE): {mae}")

    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, f'spatial_{data_type}_model.pt'))
    torch.save(likelihood.state_dict(), os.path.join(model_dir, f'spatial_{data_type}_likelihood.pt'))
    meta = {"output_cols": device_cols, "num_outputs": len(device_cols)}
    with open(os.path.join(model_dir, f'spatial_{data_type}_meta.json'), 'w') as f:
        json.dump(meta, f)
    torch.save(train_x, os.path.join(model_dir, f'spatial_{data_type}_train_x.pt'))
    torch.save(train_y, os.path.join(model_dir, f'spatial_{data_type}_train_y.pt'))
    print(f"Model artifacts saved for '{data_type}'.")
    return model, likelihood, device_cols


def predict_from_saved(data_type, x, y):
    """Load saved model & likelihood for sensor type and predict at (x,y)."""
    model_dir = 'models'
    meta_path = os.path.join(model_dir, f'spatial_{data_type}_meta.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    output_cols = meta['output_cols']; num_outputs = meta['num_outputs']

    train_x_path = os.path.join(model_dir, f'spatial_{data_type}_train_x.pt')
    train_y_path = os.path.join(model_dir, f'spatial_{data_type}_train_y.pt')
    train_x = torch.load(train_x_path, map_location='cpu')
    train_y = torch.load(train_y_path, map_location='cpu')

    likelihood = gprlib.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_outputs)
    mean = gprlib.means.MultitaskMean(gprlib.means.ZeroMean(), num_tasks=num_outputs)
    spatial_base = gprlib.kernels.MaternKernel(nu=2.5, ard=True)
    kernel = gprlib.kernels.MultitaskKernel(spatial_base, num_tasks=num_outputs, rank=min(5, num_outputs))
    distribution = gprlib.distributions.MultitaskMultivariateNormal
    model = gprlib.ExactGP(train_x, train_y, likelihood, mean, kernel, distribution)

    model.load_state_dict(torch.load(os.path.join(model_dir, f'spatial_{data_type}_model.pt'), map_location='cpu'))
    likelihood.load_state_dict(torch.load(os.path.join(model_dir, f'spatial_{data_type}_likelihood.pt'), map_location='cpu'))
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        xy = torch.tensor([[x, y]], dtype=torch.float32)
        posterior = likelihood(model(xy))
        vec = posterior.mean.numpy().reshape(-1)
    return {c: float(v) for c, v in zip(output_cols, vec)}


if __name__ == '__main__':
    # Example minimal usage
    train_model('wifi')
    print(predict_from_saved('wifi', 300.0, 175.0))
    train_model('light')
    print(predict_from_saved('light', 300.0, 175.0))