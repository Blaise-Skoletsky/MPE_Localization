import os
import random
import json
from typing import Tuple, Optional, List
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
from synthetic_data_models.gpr_spatial import train_model, predict_from_saved


def ensure_trained():
	"""Ensures that the synthetic models have been trained, with all parameter files being saved."""
	model_dir = 'models'
	def needs_train(kind):
		required = [
			os.path.join(model_dir, f'spatial_{kind}_meta.json'),
			os.path.join(model_dir, f'spatial_{kind}_model.pt'),
			os.path.join(model_dir, f'spatial_{kind}_likelihood.pt'),
			os.path.join(model_dir, f'spatial_{kind}_train_x.pt'),
			os.path.join(model_dir, f'spatial_{kind}_train_y.pt'),
		]
		return any(not os.path.exists(p) for p in required)

	if needs_train('wifi'):
		train_model('wifi')
	if needs_train('light'):
		train_model('light')


def generate_random_points(num_points: int = 100) -> pd.DataFrame:
	"""Generates synthetic readings, using (x, y) coordinates, randomly sampled within the range of real data."""
	ensure_trained()
	rows = []
	for _ in range(num_points):
		x = random.uniform(300, 700)
		y = random.uniform(175, 575)
		wifi = predict_from_saved('wifi', x, y)
		light = predict_from_saved('light', x, y)
		combined = {f'wifi_{k}': v for k, v in wifi.items()}
		combined.update({f'light_{k}': v for k, v in light.items()})
		combined['x'] = x
		combined['y'] = y
		rows.append(combined)
	df = pd.DataFrame(rows)
	return df


def _load_meta(kind: str) -> List[str]:
	meta_path = os.path.join('models', f'spatial_{kind}_meta.json')
	with open(meta_path, 'r') as f:
		meta = json.load(f)
	return meta['output_cols']


def _build_feature_list() -> List[str]:
	wifi_cols = _load_meta('wifi')
	light_cols = _load_meta('light')
	return [f'wifi_{c}' for c in wifi_cols] + [f'light_{c}' for c in light_cols]


def _load_real_combined() -> pd.DataFrame:
	"""Load and combine real wifi+light by (x,y), prefixing columns, keeping all rows (including duplicates)."""
	wifi_cols = _load_meta('wifi')
	light_cols = _load_meta('light')
	wifi_df = pd.read_csv(os.path.join('data', 'wifi_cleaned.csv'), usecols=['x', 'y'] + wifi_cols)
	light_df = pd.read_csv(os.path.join('data', 'light_cleaned.csv'), usecols=['x', 'y'] + light_cols)
	wifi_df = wifi_df.rename(columns={c: f'wifi_{c}' for c in wifi_cols})
	light_df = light_df.rename(columns={c: f'light_{c}' for c in light_cols})
	combined = pd.merge(wifi_df, light_df, on=['x', 'y'], how='inner')
	features = _build_feature_list()
	combined = combined.dropna(subset=features)
	return combined


def prepare_datasets(
	real_fraction: float,
	test_size: float = 0.2,
	val_size: float = 0.2,
	train_size: Optional[int] = None,
	random_seed: int = 42,
):
	"""Return (train_df, val_df, test_df) for a given real/synthetic mix."""
	ensure_trained()
	real_df = _load_real_combined()
	features = _build_feature_list()

	real_temp, real_test = train_test_split(real_df, test_size=test_size, random_state=random_seed, shuffle=True)
	val_rel = val_size / (1.0 - test_size)
	real_train_pool, real_val = train_test_split(real_temp, test_size=val_rel, random_state=random_seed, shuffle=True)

	if train_size is None:
		train_size = len(real_train_pool)
	real_fraction = max(0.0, min(1.0, real_fraction))
	n_real = min(int(round(train_size * real_fraction)), len(real_train_pool))
	n_synth = max(0, train_size - n_real)

	if n_real > 0:
		real_subset = real_train_pool.sample(n=n_real, random_state=random_seed, replace=False)
	else:
		real_subset = real_train_pool.iloc[0:0]

	if n_synth > 0:
		synth_df = generate_random_points(num_points=n_synth)
		for c in features:
			if c not in synth_df.columns:
				synth_df[c] = 0.0
		synth_df = synth_df[['x', 'y'] + features]
	else:
		synth_df = real_train_pool.iloc[0:0][['x', 'y']]
		for c in features:
			synth_df[c] = []

	train_df = pd.concat([real_subset[['x', 'y'] + features], synth_df], axis=0, ignore_index=True)
	train_df = train_df.dropna(subset=features)
	real_val = real_val[['x', 'y'] + features].copy()
	real_test = real_test[['x', 'y'] + features].copy()

	return train_df, real_val, real_test


def fit_knn_model(X_train: np.ndarray, y_train: np.ndarray, n_neighbors: int = 7) -> Pipeline:
	"""Fit a scaled KNN regressor for (x,y) prediction from wifi+light features."""
	pipe = Pipeline([
		('scaler', StandardScaler(with_mean=True, with_std=True)),
		('knn', KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance', p=2, n_jobs=-1))
	])
	pipe.fit(X_train, y_train)
	return pipe


def knn_predict_point(model: Pipeline, point_features: np.ndarray) -> Tuple[float, float]:
	"""Predict (x,y) for a single feature vector using the fitted KNN pipeline."""
	pred = model.predict(point_features.reshape(1, -1))[0]
	return float(pred[0]), float(pred[1])





def main():
	features = _build_feature_list()
	splits = [0.2, 0.4, 0.6, 0.8]
	print(f"Evaluating KNN across real fractions: {splits}")
	for frac in splits:
		train_df, val_df, test_df = prepare_datasets(real_fraction=frac, test_size=0.2, val_size=0.2, train_size=None, random_seed=42)
		X_train = train_df[features].to_numpy(dtype=float)
		y_train = train_df[['x', 'y']].to_numpy(dtype=float)
		X_val = val_df[features].to_numpy(dtype=float)
		y_val = val_df[['x', 'y']].to_numpy(dtype=float)
		X_test = test_df[features].to_numpy(dtype=float)
		y_test = test_df[['x', 'y']].to_numpy(dtype=float)

		print(f"\n=== Real fraction: {frac:.2f} | Synth: {1-frac:.2f} ===")
		print(f"Train set size: {len(X_train)} | Validation set size: {len(X_val)} | Test set size: {len(X_test)}")

		model = fit_knn_model(X_train, y_train, n_neighbors=7)

		val_pred = model.predict(X_val) 
		test_pred = model.predict(X_test)
		val_mae = mean_absolute_error(y_val, val_pred)
		print(f"Validation MAE (avg over x,y): {val_mae:.3f} on {len(X_val)} samples")
		test_mae = mean_absolute_error(y_test, test_pred)
		print(f"Test MAE (avg over x,y): {test_mae:.3f} on {len(X_test)} samples")


if __name__ == '__main__':
	main()
