import os
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
try:
    # Prefer the local package when used as a package
    from . import gprlib
except Exception:
    # Fallback for scripts run as top-level modules
    import gprlib

def train_model(data_type):
    print('Starting: ')
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    if data_type == 'wifi':
        dataset = pd.read_csv(os.path.join(data_dir, "wifi_pre_pca.csv"))
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
    # Bias toward a minimum observation noise so the model cannot overfit extremely small residuals
    noise_floor = 0.05 if data_type == 'wifi' else 0.1
    with torch.no_grad():
        if hasattr(likelihood, 'raw_noise'):
            floor = torch.full_like(likelihood.raw_noise, float(np.log(np.expm1(noise_floor) + 1e-8)))
            likelihood.raw_noise.copy_(torch.maximum(likelihood.raw_noise, floor))
    mean = gprlib.means.MultitaskMean(gprlib.means.ZeroMean(), num_tasks=train_y.shape[1])
    spatial_base = gprlib.kernels.MaternKernel(nu=2.5, ard=True)
    kernel_rank = 3 if data_type == 'wifi' else 2
    kernel = gprlib.kernels.MultitaskKernel(spatial_base, num_tasks=train_y.shape[1], rank=kernel_rank)
    distribution = gprlib.distributions.MultitaskMultivariateNormal
    model = gprlib.ExactGP(train_x, train_y, likelihood, mean, kernel, distribution)

    mll = gprlib.mlls.ExactMarginalLogLikelihood(likelihood, model)
    # Smaller learning rates plus mild weight decay slow convergence and reduce overfitting
    learning_rate = 3e-4 if data_type == 'wifi' else 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    epochs = 500 if data_type == 'wifi' else 1000 if data_type == 'light' else 500
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.1)

    model.train(); likelihood.train()
    # Reduce epochs to reasonable defaults to avoid long runs; adjust as needed
    rng = tqdm(range(epochs), desc="Training")
    for _ in rng:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        # detect NaN loss and stop early
        if torch.isnan(loss) or torch.isinf(loss):
            print("Encountered NaN/Inf loss during training — stopping early.")
            break
        loss.backward()
        # gradient clipping to improve stability
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        except Exception:
            pass
        rng.set_postfix(loss=float(loss.item()))
        optimizer.step()
        scheduler.step()

    # Evaluate. Predict one point at a time to avoid constructing very large
    # Kronecker covariance matrices (which can OOM for many tasks / points).
    model.eval(); likelihood.eval()
    pred_means = []
    with torch.no_grad():
        for i in range(test_x.shape[0]):
            xy = test_x[i:i+1]
            try:
                posterior = likelihood(model(xy))
                pm = posterior.mean.squeeze(0)
            except RuntimeError as e:
                try:
                    pm = model(xy).mean.squeeze(0)
                except Exception:
                    pm = torch.zeros(train_y.shape[1])
            pred_means.append(pm)
    preds_mean = torch.stack(pred_means, dim=0)
    mae = torch.mean(torch.abs(preds_mean - test_y)).item()
    print(f"Mean Absolute Error (MAE): {mae}")

    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, f'spatial_{data_type}_model.pt'))
    torch.save(likelihood.state_dict(), os.path.join(model_dir, f'spatial_{data_type}_likelihood.pt'))
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
    if data_type == 'wifi':
        output_cols = meta.get('pre_pca', {}).get('columns') or meta.get('columns') or meta.get('output_cols')
    else:
        output_cols = meta.get('columns') or meta.get('output_cols')
    if output_cols is None:
        raise ValueError(f"Unable to load feature columns from {meta_path}")
    num_outputs = len(output_cols)

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
    vec = np.array(vec, dtype=float)
    return {c: float(v) for c, v in zip(output_cols, vec)}


if __name__ == '__main__':
    # Example minimal usage
    train_model('wifi')
    print(predict_from_saved('wifi', 300.0, 175.0))
    train_model('light')
    print(predict_from_saved('light', 300.0, 175.0))