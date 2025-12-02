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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SYNTHETIC_DIR = os.path.join(BASE_DIR, 'synthetic_data')


def _resolve_path(path: str) -> str:
	"""Resolve a relative path against the knn module directory."""
	return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


def generate_random_points(num_points: int = 100) -> pd.DataFrame:
	"""Generates synthetic readings. (num_points)"""

	synth_path = os.path.join(SYNTHETIC_DIR, 'synthetic_knn_points.csv')
	if not os.path.exists(synth_path):
		raise FileNotFoundError(f"Synthetic data file not found, or data not yet generated : {synth_path}")

	features = build_feature_list()
	df = pd.read_csv(synth_path)

	#if we don't have enough points, we can resample (try not to do this!)
	replace = num_points > len(df)
	sampled = df.sample(n=num_points, replace=replace, random_state=42).reset_index(drop=True)
	return sampled[['x', 'y'] + features]


def load_meta(kind: str, subset: str = 'default') -> List[str]:
	"""Load feature columns for the requested sensor kind.

	For wifi, `subset` can be 'pre' (pre-PCA) or 'post' (default, PCA features).
	For other kinds, the stored columns are returned regardless of subset.
	"""
	meta_path = os.path.join(MODELS_DIR, f'spatial_{kind}_meta.json')
	with open(meta_path, 'r') as f:
		meta = json.load(f)

	if kind == 'wifi' and isinstance(meta, dict):
		if subset == 'pre':
			cols = meta.get('pre_pca', {}).get('columns')
			if cols:
				return cols
		# default to post-PCA columns
		cols = meta.get('post_pca', {}).get('columns')
		if cols:
			return cols
		# fallback to pre-PCA if post missing
		cols = meta.get('pre_pca', {}).get('columns')
		if cols:
			return cols

	# Generic fallback for light or legacy schemas
	if 'columns' in meta:
		return meta['columns']
	if 'output_cols' in meta:
		return meta['output_cols']
	raise KeyError(f"Could not determine columns for kind={kind} in {meta_path}")


def build_feature_list() -> List[str]:
	wifi_cols = load_meta('wifi', subset='post')
	light_cols = load_meta('light')

	def _base_name(col: str, prefix: str) -> str:
		if col.startswith(prefix + '_'):
			return col[len(prefix) + 1 :]
		return col

	wifi_bases = [_base_name(c, 'wifi') for c in wifi_cols]
	light_bases = [_base_name(c, 'light') for c in light_cols]

	# Always expose features in the combined dataframe as prefixed names
	return [f'wifi_{b}' for b in wifi_bases] + [f'light_{b}' for b in light_bases]


def load_real_combined() -> pd.DataFrame:
	"""Load and combine real wifi+light by (x,y), prefixing columns, keeping all rows (including duplicates)."""
	wifi_meta = load_meta('wifi', subset='post')
	light_meta = load_meta('light')

	# Read full CSVs so we can map meta names to actual columns robustly
	wifi_df = pd.read_csv(os.path.join(DATA_DIR, 'wifi_cleaned.csv'))
	light_df = pd.read_csv(os.path.join(DATA_DIR, 'light_cleaned.csv'))

	def _base_name(col: str, prefix: str) -> str:
		if col.startswith(prefix + '_'):
			return col[len(prefix) + 1 :]
		return col

	# Build selection frames with consistent prefixed column names
	wifi_sel = wifi_df[['x', 'y']].copy() if 'x' in wifi_df.columns and 'y' in wifi_df.columns else pd.DataFrame()
	for c in wifi_meta:
		base = _base_name(c, 'wifi')
		target = f'wifi_{base}'
		if c in wifi_df.columns:
			wifi_sel[target] = wifi_df[c]
		elif base in wifi_df.columns:
			wifi_sel[target] = wifi_df[base]
		elif target in wifi_df.columns:
			wifi_sel[target] = wifi_df[target]
		else:
			wifi_sel[target] = 0.0

	light_sel = light_df[['x', 'y']].copy() if 'x' in light_df.columns and 'y' in light_df.columns else pd.DataFrame()
	for c in light_meta:
		base = _base_name(c, 'light')
		target = f'light_{base}'
		if c in light_df.columns:
			light_sel[target] = light_df[c]
		elif base in light_df.columns:
			light_sel[target] = light_df[base]
		elif target in light_df.columns:
			light_sel[target] = light_df[target]
		else:
			light_sel[target] = 0.0

	combined = pd.merge(wifi_sel, light_sel, on=['x', 'y'], how='inner')
	features = build_feature_list()
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
	real_df = load_real_combined()
	features = build_feature_list()

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

	train_df = real_subset[['x', 'y'] + features].copy()
	if n_synth > 0:
		synth_df = generate_random_points(num_points=n_synth)
		synth_df = synth_df[['x', 'y'] + features]
		train_df = pd.concat([train_df, synth_df], axis=0, ignore_index=True)
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


def neighbor_distances(
	model: Pipeline,
	point_features: np.ndarray,
	n_neighbors: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Return (distances, indices) for the neighbors used in prediction order."""
	scaler = model.named_steps.get('scaler')
	knn = model.named_steps.get('knn')
	if knn is None or not hasattr(knn, 'kneighbors'):
		raise ValueError('Pipeline missing a KNeighborsRegressor step named "knn"')
	vector = point_features.reshape(1, -1)
	if scaler is not None:
		vector = scaler.transform(vector)
	k = n_neighbors or getattr(knn, 'n_neighbors', None)
	if k is None:
		raise ValueError('Unable to determine number of neighbors for inspection')
	distances, indices = knn.kneighbors(vector, n_neighbors=k, return_distance=True)
	return distances[0], indices[0]


def search_best_k(
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_val: np.ndarray,
	y_val: np.ndarray,
	k_start: int = 1,
	k_stop: int = 31,
	k_step: int = 2,
) -> Tuple[int, float, List[Tuple[int, float]]]:
	if X_val.size == 0:
		raise ValueError("Validation set is empty; cannot run hyperparameter search.")
	for k in range(k_start, k_stop + 1, k_step):
		model = fit_knn_model(X_train, y_train, n_neighbors=k)
		val_pred = model.predict(X_val)
		mae = mean_absolute_error(y_val, val_pred)
		print(f"Tested k={k} | Validation MAE: {mae:.3f}")



def knn_predict_point(
	model: Pipeline,
	point_features: np.ndarray,
	log_neighbors: bool = False,
) -> Tuple[float, float]:
	"""Predict (x,y) for one feature vector and optionally log neighbor distances."""
	pred = model.predict(point_features.reshape(1, -1))[0]
	if log_neighbors:
		dists, idxs = neighbor_distances(model, point_features)
		print(f"Neighbor distances: {np.array2string(dists, precision=3)}")
		print(f"Neighbor indices: {idxs.tolist()}")
	return float(pred[0]), float(pred[1])


def load_paired_csvs(wifi_path: str, light_path: str) -> pd.DataFrame:
	"""Load two CSVs (wifi, light) that contain x,y and feature columns and return combined prefixed DataFrame.

	The function expects the wifi/light CSVs to include columns matching the metadata output columns
	from the trained GP (those listed in `models/spatial_{kind}_meta.json`). Returned DataFrame contains
	columns: ['x','y', 'wifi_<col>...', 'light_<col>...'].
	"""
	wifi_meta = load_meta('wifi', subset='post')
	light_meta = load_meta('light')

	wifi_df = pd.read_csv(_resolve_path(wifi_path))
	light_df = pd.read_csv(_resolve_path(light_path))

	# Ensure x,y exist
	if 'x' not in wifi_df.columns or 'y' not in wifi_df.columns:
		raise ValueError(f"wifi CSV missing 'x' or 'y' columns: {wifi_path}")
	if 'x' not in light_df.columns or 'y' not in light_df.columns:
		raise ValueError(f"light CSV missing 'x' or 'y' columns: {light_path}")

	def _base_name(col: str, prefix: str) -> str:
		if col.startswith(prefix + '_'):
			return col[len(prefix) + 1 :]
		return col

	wifi_sel = wifi_df[['x', 'y']].copy()
	for c in wifi_meta:
		base = _base_name(c, 'wifi')
		target = f'wifi_{base}'
		if c in wifi_df.columns:
			wifi_sel[target] = wifi_df[c]
		elif base in wifi_df.columns:
			wifi_sel[target] = wifi_df[base]
		elif target in wifi_df.columns:
			wifi_sel[target] = wifi_df[target]
		else:
			wifi_sel[target] = 0.0

	light_sel = light_df[['x', 'y']].copy()
	for c in light_meta:
		base = _base_name(c, 'light')
		target = f'light_{base}'
		if c in light_df.columns:
			light_sel[target] = light_df[c]
		elif base in light_df.columns:
			light_sel[target] = light_df[base]
		elif target in light_df.columns:
			light_sel[target] = light_df[target]
		else:
			light_sel[target] = 0.0

	combined = pd.merge(wifi_sel, light_sel, on=['x', 'y'], how='inner')
	features = build_feature_list()
	for col in features:
		if col not in combined.columns:
			combined[col] = 0.0
	combined[features] = combined[features].fillna(0.0)
	return combined[['x', 'y'] + features]


def test_on_labeled(wifi_path: str, light_path: str, real_fraction: float = 0.4, n_neighbors: int = 7):
	"""Train the KNN (using the same mixing policy) and evaluate on labeled wifi/light CSVs.

	Returns a dict with MAE statistics and prints the summary.
	"""
	# Prepare training data (mix of real/synthetic) using the existing prepare_datasets()
	train_df, val_df, test_df = prepare_datasets(real_fraction=real_fraction, test_size=0.2, val_size=0.2, train_size=None, random_seed=42)
	features = build_feature_list()
	X_train = train_df[features].to_numpy(dtype=float)
	y_train = train_df[['x', 'y']].to_numpy(dtype=float)
	model = fit_knn_model(X_train, y_train, n_neighbors=n_neighbors)

	# Load labeled CSVs
	labeled = load_paired_csvs(wifi_path, light_path)
	X_labeled = labeled[features].to_numpy(dtype=float)
	y_true = labeled[['x', 'y']].to_numpy(dtype=float)

	# Predict and score
	if len(X_labeled) == 0:
		print("No labeled samples found to evaluate.")
		return {'mae_overall': None, 'mae_x': None, 'mae_y': None, 'n_samples': 0}

	preds = model.predict(X_labeled)
	mae_overall = mean_absolute_error(y_true, preds)
	mae_x = mean_absolute_error(y_true[:, 0], preds[:, 0])
	mae_y = mean_absolute_error(y_true[:, 1], preds[:, 1])

	print(f"Labeled evaluation on files: {wifi_path} + {light_path}")
	print(f"Samples: {len(X_labeled)} | MAE overall: {mae_overall:.3f} | MAE_x: {mae_x:.3f} | MAE_y: {mae_y:.3f}")
	return {'mae_overall': float(mae_overall), 'mae_x': float(mae_x), 'mae_y': float(mae_y), 'n_samples': int(len(X_labeled))}





def main():
	features = build_feature_list()
	splits = [1, .8, .6, .4, .2, 0]
	print(f"\nEvaluating KNN across real fractions: {splits}")
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

		search_best_k(X_train, y_train, X_val, y_val)
		model = fit_knn_model(X_train, y_train, n_neighbors=7)
		val_pred = model.predict(X_val)
		test_pred = model.predict(X_test)
		val_mae = mean_absolute_error(y_val, val_pred)
		print(f"Validation MAE (avg over x,y): {val_mae:.3f} on {len(X_val)} samples")
		test_mae = mean_absolute_error(y_test, test_pred)
		print(f"Test MAE (avg over x,y): {test_mae:.3f} on {len(X_test)} samples")


if __name__ == '__main__':
	main()
