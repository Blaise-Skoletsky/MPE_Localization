"""Generate presentation-ready visualizations for the KNN localization project.

This script produces multiple figures that summarize the training pipeline:

1. Hyper-parameter (k) sweeps across multiple real/synthetic splits.
2. WiFi column null-percentage chart showing which signals were discarded.
3. Distribution comparisons between real measurements and GPR-generated samples.
4. Spatial accuracy heatmap highlighting where the model performs best/worst.
5. Additional plots: error histogram & predicted-vs-true scatter for quick sanity checks.

All figures are saved under ``figures/presentation`` relative to this file.
"""

from __future__ import annotations

import os
import pickle
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.metrics import mean_absolute_error

from kNN_Model import (
	build_feature_list,
	fit_knn_model,
	prepare_datasets,
	load_real_combined,
	load_paired_csvs,
	load_meta,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SYNTHETIC_PATH = os.path.join(BASE_DIR, 'synthetic_data', 'synthetic_knn_points.csv')
FIG_DIR = os.path.join(BASE_DIR, 'figures', 'presentation')
OLD_WIFI_PRE = os.path.join(DATA_DIR, 'old_wifi_cleaned.csv')
NEW_WIFI_PRE = os.path.join(DATA_DIR, 'wifi_pre_pca.csv')
OLD_LIGHT_CLEAN = os.path.join(DATA_DIR, 'old_light_cleaned.csv')
NEW_WIFI_CLEAN = os.path.join(DATA_DIR, 'wifi_cleaned.csv')
NEW_LIGHT_CLEAN = os.path.join(DATA_DIR, 'light_cleaned.csv')
WIFI_SCALER_PATH = os.path.join(MODELS_DIR, 'spatial_wifi_scaler.pkl')
LIGHT_SCALER_PATH = os.path.join(MODELS_DIR, 'spatial_light_scaler.pkl')
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style='whitegrid')


def _base_wifi_name(col: str) -> str:
	return col[5:] if col.startswith('wifi_') else col


def _base_light_name(col: str) -> str:
	return col[6:] if col.startswith('light_') else col


@lru_cache(maxsize=1)
def _load_wifi_scaler_payload() -> Tuple[Dict[str, float], Dict[str, float], Optional[float], Optional[float]]:
	if not os.path.exists(WIFI_SCALER_PATH):
		raise FileNotFoundError(f"Missing wifi scaler metadata at {WIFI_SCALER_PATH}")
	with open(WIFI_SCALER_PATH, 'rb') as sf:
		payload = pickle.load(sf)
	means = { _base_wifi_name(k): float(v) for k, v in payload.get('means', {}).items() }
	stds = { _base_wifi_name(k): (float(v) if float(v) != 0 else 1.0) for k, v in payload.get('stds', {}).items() }
	global_min = payload.get('global_min')
	global_max = payload.get('global_max')
	return (
		means,
		stds,
		float(global_min) if global_min is not None else None,
		float(global_max) if global_max is not None else None,
	)


@lru_cache(maxsize=1)
def _load_light_scaler_payload() -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
	if not os.path.exists(LIGHT_SCALER_PATH):
		raise FileNotFoundError(f"Missing light scaler metadata at {LIGHT_SCALER_PATH}")
	with open(LIGHT_SCALER_PATH, 'rb') as lf:
		payload = pickle.load(lf)
	mins = {str(k): float(v) for k, v in payload.get('mins', {}).items()}
	maxs = {str(k): float(v) for k, v in payload.get('maxs', {}).items()}
	means = {str(k): float(v) for k, v in payload.get('means', {}).items()}
	stds = {str(k): (float(v) if float(v) != 0 else 1.0) for k, v in payload.get('stds', {}).items()}
	return mins, maxs, means, stds


def standardize_wifi_with_metadata(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
	"""Apply the stored min/max + z-score stats to a *raw* wifi dataframe."""
	if df.empty:
		raise ValueError('Wifi dataframe is empty; cannot standardize')
	means, stds, global_min, global_max = _load_wifi_scaler_payload()
	result = df.copy()
	if global_min is None or global_max is None or global_max == global_min:
		raise ValueError('Wifi scaler metadata lacks valid global min/max; cannot standardize')
	denom = global_max - global_min
	if denom == 0:
		raise ValueError('Wifi scaler metadata produced invalid denominator; cannot standardize')
	fill_value = global_min if global_min is not None else 0.0
	for feat in features:
		if feat not in result.columns:
			raise KeyError(f"Wifi column '{feat}' missing; cannot standardize")
		base = _base_wifi_name(feat)
		series = result[feat].astype(float)
		series = series.fillna(fill_value)
		series = (series - fill_value) / denom
		mean = means.get(base)
		std = stds.get(base, 1.0) or 1.0
		if mean is not None:
			series = (series - mean) / std
		result[feat] = series
	return result


def standardize_light_with_metadata(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
	if df.empty:
		raise ValueError('Light dataframe is empty; cannot standardize')
	mins, maxs, means, stds = _load_light_scaler_payload()
	result = df.copy()
	for feat in features:
		if feat not in result.columns:
			raise KeyError(f"Light column '{feat}' missing; cannot standardize")
		base = _base_light_name(feat)
		col_min = mins.get(base)
		col_max = maxs.get(base)
		if col_min is None or col_max is None or col_max == col_min:
			raise ValueError(f"Light scaler stats invalid for column '{base}'")
		series = result[feat].astype(float).fillna(col_min)
		series = (series - col_min) / (col_max - col_min)
		mean = means.get(base)
		std = stds.get(base, 1.0)
		if mean is None or std is None or std == 0:
			raise ValueError(f"Light standardization stats missing for column '{base}'")
		series = (series - mean) / std
		result[feat] = series
	return result


def _save_current_fig(name: str) -> None:
	path = os.path.join(FIG_DIR, name)
	plt.tight_layout()
	plt.savefig(path, dpi=200)
	print(f"Saved figure -> {path}")


def run_hyperparameter_search(
	real_fracs: Sequence[float],
	k_values: Sequence[int],
	random_seed: int = 42,
) -> pd.DataFrame:
	"""Run KNN sweeps for each real fraction and return a tidy DataFrame of MAEs."""

	features = build_feature_list()
	rows: List[Dict[str, float]] = []
	for frac in real_fracs:
		train_df, val_df, _ = prepare_datasets(
			real_fraction=frac,
			test_size=0.2,
			val_size=0.2,
			train_size=None,
			random_seed=random_seed,
		)
		if len(val_df) == 0:
			continue
		X_train = train_df[features].to_numpy(dtype=float)
		y_train = train_df[['x', 'y']].to_numpy(dtype=float)
		X_val = val_df[features].to_numpy(dtype=float)
		y_val = val_df[['x', 'y']].to_numpy(dtype=float)

		for k in k_values:
			model = fit_knn_model(X_train, y_train, n_neighbors=k)
			preds = model.predict(X_val)
			mae = mean_absolute_error(y_val, preds)
			rows.append({'real_fraction': frac, 'k': k, 'mae': float(mae)})
			print(f"real_fraction={frac:.2f} | k={k} | val MAE={mae:.4f}")

	return pd.DataFrame(rows)


def plot_hyperparameter_search(df: pd.DataFrame) -> None:
	if df.empty:
		print('No hyperparameter results to plot.')
		return
	plt.figure(figsize=(10, 6))
	sns.lineplot(data=df, x='k', y='mae', hue='real_fraction', marker='o')
	plt.title('Validation MAE vs. k across real-data fractions')
	plt.xlabel('Number of Neighbors (k)')
	plt.ylabel('Validation MAE (avg over x,y)')
	plt.legend(title='Real Fraction')
	_save_current_fig('hyperparameter_sweep.png')
	plt.close()


def plot_wifi_null_summary(wifi_csv_path: str, top_n: int = 25) -> None:
	if not os.path.exists(wifi_csv_path):
		print(f"Cannot find wifi CSV at {wifi_csv_path}; skipping null-percentage chart.")
		return
	wifi_df = pd.read_csv(wifi_csv_path)
	feature_cols = [c for c in wifi_df.columns if c not in ('timestamp', 'x', 'y')]
	if not feature_cols:
		print('No wifi feature columns detected.')
		return
	null_pct = wifi_df[feature_cols].isna().mean().sort_values(ascending=False)
	top_series = null_pct.head(top_n)

	plt.figure(figsize=(14, 6))
	plt.subplot(1, 2, 1)
	sns.barplot(x=top_series.values * 100, y=top_series.index, color='tab:green')
	plt.title(f'Top {min(top_n, len(top_series))} WiFi columns by % null')
	plt.xlabel('Null Percentage (%)')
	plt.ylabel('WiFi Column')

	plt.subplot(1, 2, 2)
	values_pct = null_pct.values * 100
	sns.histplot(values_pct, bins=20, color='tab:blue')
	plt.axvline(values_pct.mean(), color='r', linestyle='--', label='Mean')
	plt.axvline(np.median(values_pct), color='g', linestyle=':', label='Median')
	plt.title('Distribution of null percentages across all WiFi columns')
	plt.xlabel('Null Percentage (%)')
	plt.ylabel('Column count')
	plt.legend()

	_save_current_fig('wifi_null_summary.png')
	plt.close()


def plot_real_vs_synth_distributions(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> None:
	if real_df.empty or synth_df.empty:
		print('Cannot plot distributions: real or synthetic dataframe empty.')
		return
	selected_cols = ['x', 'y']
	# Add a couple sensor features if available
	for candidate in ['wifi_pca_0', 'wifi_pca_1', 'light_f1', 'light_clear']:
		if candidate in real_df.columns and candidate in synth_df.columns:
			selected_cols.append(candidate)
	selected_cols = list(dict.fromkeys(selected_cols))  # preserve order, remove dups

	n_cols = len(selected_cols)
	n_rows = int(np.ceil(n_cols / 2))
	plt.figure(figsize=(12, 4 * n_rows))
	for idx, col in enumerate(selected_cols, start=1):
		plt.subplot(n_rows, 2, idx)
		sns.kdeplot(real_df[col], label='Real', fill=True, common_norm=False)
		sns.kdeplot(synth_df[col], label='Synthetic', fill=True, common_norm=False)
		plt.title(f'Distribution: {col}')
		plt.xlabel(col)
		if idx == 1:
			plt.legend()
	_save_current_fig('real_vs_synth_distributions.png')
	plt.close()


def build_wifi_pre_feature_list() -> List[str]:
	wifi_cols = load_meta('wifi', subset='pre')
	features: List[str] = []
	for col in wifi_cols:
		base = col
		if col.startswith('wifi_'):
			base = col[5:]
		features.append(f'wifi_{base}')
	return features


def build_light_feature_list() -> List[str]:
	light_cols = load_meta('light')
	features: List[str] = []
	for col in light_cols:
		base = col
		if col.startswith('light_'):
			base = col[6:]
		features.append(f'light_{base}')
	return features


def load_wifi_pre_dataset(path: str, features: List[str]) -> pd.DataFrame:
	if not os.path.exists(path):
		raise FileNotFoundError(f"Missing wifi pre-PCA file: {path}")
	df = pd.read_csv(path)
	if 'x' not in df.columns or 'y' not in df.columns:
		raise ValueError(f"wifi dataset at {path} lacks x/y columns")
	result = df[['x', 'y']].copy()
	col_payload: Dict[str, pd.Series] = {}
	for feat in features:
		base = feat[5:]
		if feat in df.columns:
			col_payload[feat] = df[feat]
		elif base in df.columns:
			col_payload[feat] = df[base]
		else:
			raise KeyError(f"wifi column '{base}' missing from {path}")
	if col_payload:
		result = pd.concat([result, pd.DataFrame(col_payload)], axis=1)
	return result


def load_light_dataset(path: str, features: List[str]) -> pd.DataFrame:
	if not os.path.exists(path):
		raise FileNotFoundError(f"Missing light file: {path}")
	df = pd.read_csv(path)
	if 'x' not in df.columns or 'y' not in df.columns:
		raise ValueError(f"light dataset at {path} lacks x/y columns")
	result = df[['x', 'y']].copy()
	col_payload: Dict[str, pd.Series] = {}
	for feat in features:
		base = _base_light_name(feat)
		if feat in df.columns:
			col_payload[feat] = df[feat]
		elif base in df.columns:
			col_payload[feat] = df[base]
		else:
			raise KeyError(f"light column '{base}' missing from {path}")
	if col_payload:
		result = pd.concat([result, pd.DataFrame(col_payload)], axis=1)
	return result


def load_combined_dataset(wifi_path: str, light_path: str) -> pd.DataFrame:
	if not os.path.exists(wifi_path) or not os.path.exists(light_path):
		print(f"Missing combined dataset files: {wifi_path}, {light_path}")
		return pd.DataFrame()
	return load_paired_csvs(wifi_path, light_path)



def compute_old_vs_new_differences() -> Tuple[pd.DataFrame, List[str]]:
	wifi_features = build_wifi_pre_feature_list()
	light_features = build_light_feature_list()
	new_wifi = load_wifi_pre_dataset(NEW_WIFI_PRE, wifi_features)
	new_light = load_light_dataset(NEW_LIGHT_CLEAN, light_features)
	old_wifi = load_wifi_pre_dataset(OLD_WIFI_PRE, wifi_features)
	old_light = load_light_dataset(OLD_LIGHT_CLEAN, light_features)
	old_wifi = standardize_wifi_with_metadata(old_wifi, wifi_features)
	old_light = standardize_light_with_metadata(old_light, light_features)
	new_combined = pd.merge(new_wifi, new_light, on=['x', 'y'], how='inner')
	old_combined = pd.merge(old_wifi, old_light, on=['x', 'y'], how='inner')
	features = wifi_features + light_features
	merged = pd.merge(
		new_combined[['x', 'y'] + features],
		old_combined[['x', 'y'] + features],
		on=['x', 'y'],
		suffixes=('_new', '_old'),
	)
	if merged.empty:
		raise ValueError('No overlapping (x,y) coordinates between old and new datasets.')
	available = [feat for feat in features if f'{feat}_new' in merged.columns and f'{feat}_old' in merged.columns]
	return merged, available


def plot_feature_shift_summary(
	merged: pd.DataFrame,
	features: List[str],
	label: str,
	top_n: int = 10,
) -> Tuple[List[str], pd.DataFrame]:
	if merged.empty:
		return [], pd.DataFrame()
	shift_rows = []
	for feat in features:
		new_col = f'{feat}_new'
		old_col = f'{feat}_old'
		if new_col not in merged.columns or old_col not in merged.columns:
			continue
		if merged[new_col].isna().all() and merged[old_col].isna().all():
			continue
		diff = merged[new_col] - merged[old_col]
		shift_rows.append({
			'feature': feat,
			'domain': 'wifi' if feat.startswith('wifi_') else 'light',
			'mean_abs_diff': float(np.nanmean(np.abs(diff))),
			'median_diff': float(np.nanmedian(diff)),
		})
	shift_df = pd.DataFrame(shift_rows).sort_values('mean_abs_diff', ascending=False)
	if shift_df.empty:
		print('No overlapping feature columns for old/new comparison.')
		return [], shift_df
	top_feats = shift_df.head(top_n)
	plt.figure(figsize=(12, 7))
	sns.barplot(data=top_feats, x='mean_abs_diff', y='feature', hue='domain', dodge=False)
	plt.title(f'Top feature shifts between old and new datasets (by mean |Δ|) [{label}]')
	plt.xlabel('Mean absolute difference')
	plt.ylabel('Feature')
	_save_current_fig(f'old_vs_new_feature_shift_{label}.png')
	plt.close()
	return top_feats['feature'].tolist(), shift_df


def plot_shift_overview_histogram(shift_df: pd.DataFrame, label: str) -> None:
	if shift_df.empty:
		return
	edges = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, np.inf]
	labels = []
	for idx in range(len(edges) - 1):
		start = edges[idx]
		end = edges[idx + 1]
		if np.isinf(end):
			labels.append(f'>= {start:.2f}')
		else:
			labels.append(f'{start:.2f}-{end:.2f}')
	shift_df = shift_df.copy()
	shift_df['mae_bin'] = pd.cut(
		shift_df['mean_abs_diff'],
		edges,
		labels=labels,
		include_lowest=True,
		right=False,
	)
	overview = shift_df['mae_bin'].value_counts().reindex(labels, fill_value=0).reset_index()
	overview.columns = ['bin', 'count']
	plt.figure(figsize=(12, 5))
	sns.barplot(data=overview, x='bin', y='count', color='tab:blue')
	plt.xticks(rotation=45, ha='right')
	plt.xlabel('Mean absolute difference bucket (|Δ|)')
	plt.ylabel('Feature count')
	plt.title(f'Feature shift distribution by |Δ| bucket [{label}]')
	_save_current_fig(f'old_vs_new_feature_shift_overview_{label}.png')
	plt.close()


def plot_full_feature_shift_chart(shift_df: pd.DataFrame, label: str) -> None:
	if shift_df.empty:
		return
	sorted_df = shift_df.sort_values('mean_abs_diff', ascending=False)
	fig_height = max(6, len(sorted_df) * 0.18)
	plt.figure(figsize=(12, fig_height))
	sns.barplot(data=sorted_df, x='mean_abs_diff', y='feature', hue='domain', dodge=False)
	plt.xlabel('Mean absolute difference (|Δ|)')
	plt.ylabel('Feature')
	plt.title(f'Per-feature distribution shift overview [{label}]')
	plt.legend(title='Domain', loc='upper right')
	plt.tight_layout()
	_save_current_fig(f'old_vs_new_feature_shift_full_{label}.png')
	plt.close()



def plot_feature_distribution_overlays(
	merged: pd.DataFrame,
	features: List[str],
	selected_features: List[str],
	label: str,
	suffix: str = 'top',
	max_plots: int = 6,
) -> None:
	if merged.empty or not selected_features:
		return
	plot_feats = selected_features[:max_plots]
	n_cols = 2
	n_rows = int(np.ceil(len(plot_feats) / n_cols))
	plt.figure(figsize=(12, 4 * n_rows))
	for idx, feat in enumerate(plot_feats, start=1):
		plt.subplot(n_rows, n_cols, idx)
		new_vals = merged[f'{feat}_new'].dropna()
		old_vals = merged[f'{feat}_old'].dropna()
		if not new_vals.empty:
			sns.kdeplot(new_vals, label='New dataset', fill=True)
		if not old_vals.empty:
			sns.kdeplot(old_vals, label='Old dataset', fill=True)
		plt.title(f'{feat} distribution shift')
		plt.xlabel(feat)
		if idx == 1:
			plt.legend()
	_save_current_fig(f'old_vs_new_distributions_{suffix}_{label}.png')
	plt.close()


def plot_feature_difference_map(merged: pd.DataFrame, features: List[str], label: str) -> None:
	if merged.empty:
		return
	print("1")
	diff_df = merged[['x', 'y']].copy()
	abs_diff_payload: Dict[str, pd.Series] = {}
	for feat in features:
		new_col = f'{feat}_new'
		old_col = f'{feat}_old'
		if new_col not in merged.columns or old_col not in merged.columns:
			continue
		col_name = f'{feat}_abs_diff'
		abs_diff_payload[col_name] = np.abs(merged[new_col] - merged[old_col])
	if not abs_diff_payload:
		return
	print("2")
	diff_df = pd.concat([diff_df, pd.DataFrame(abs_diff_payload)], axis=1)
	feature_cols = list(abs_diff_payload.keys())
	diff_df['mean_abs_diff'] = diff_df[feature_cols].mean(axis=1)
	vmax = float(diff_df['mean_abs_diff'].max()) if feature_cols else 0.0
	if not np.isfinite(vmax) or vmax == 0:
		vmax = 0.5
	clamped_vmax = max(0.5, min(1.0, vmax))
	print("3")
	norm = Normalize(vmin=0.0, vmax=clamped_vmax)
	print("4")
	plt.figure(figsize=(8, 6))
	sc = plt.scatter(diff_df['x'], diff_df['y'], c=diff_df['mean_abs_diff'], cmap='magma', s=50, edgecolor='none', norm=norm)
	for x, y, val in zip(diff_df['x'], diff_df['y'], diff_df['mean_abs_diff']):
		plt.text(x, y + 8, f"{val:.2f}", color='black', ha='center', va='bottom', fontsize=8)
	plt.colorbar(sc, label='Mean |Δfeature| across modalities')
	plt.title(f'Spatial map of feature drift (old vs new) [{label}]')
	plt.xlabel('X (cm)')
	plt.ylabel('Y (cm)')
	_save_current_fig(f'old_vs_new_spatial_drift_{label}.png')
	plt.close()


def run_feature_drift_pipeline(label: str) -> None:
	merged_df, overlap_features = compute_old_vs_new_differences()
	#if merged_df.empty or not overlap_features:
	#		return
	#top_features, shift_df = plot_feature_shift_summary(merged_df, overlap_features, label=label, top_n=6)
	#plot_full_feature_shift_chart(shift_df, label)
	#plot_shift_overview_histogram(shift_df, label)
	#if top_features:
	#		plot_feature_distribution_overlays(merged_df, overlap_features, top_features, label=label, suffix='top', max_plots=6)
	plot_feature_difference_map(merged_df, overlap_features, label=label)

def train_baseline_model(
	frac: float = 0.4,
	n_neighbors: int = 13,
	random_seed: int = 42,
) -> Tuple[object, List[str], pd.DataFrame, pd.DataFrame]:
	features = build_feature_list()
	train_df, _, test_df = prepare_datasets(
		real_fraction=frac,
		test_size=0.2,
		val_size=0.2,
		train_size=None,
		random_seed=random_seed,
	)
	X_train = train_df[features].to_numpy(dtype=float)
	y_train = train_df[['x', 'y']].to_numpy(dtype=float)

	model = fit_knn_model(X_train, y_train, n_neighbors=n_neighbors)

	X_test = test_df[features].to_numpy(dtype=float)
	y_test = test_df[['x', 'y']].to_numpy(dtype=float)
	preds = model.predict(X_test)
	errors = np.linalg.norm(preds - y_test, axis=1)

	results = test_df[['x', 'y']].copy()
	results['error_cm'] = errors
	preds_df = pd.DataFrame({
		'pred_x': preds[:, 0],
		'pred_y': preds[:, 1],
		'true_x': y_test[:, 0],
		'true_y': y_test[:, 1],
		'error_cm': errors,
	})
	return model, features, results, preds_df


def evaluate_on_dataframe(
	model: object,
	features: List[str],
	df: pd.DataFrame,
	label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	if df.empty:
		raise ValueError(f"Cannot evaluate {label}: dataframe is empty")
	df = df.copy()
	missing = [col for col in features if col not in df.columns]
	for col in missing:
		df[col] = 0.0
	X = df[features].fillna(0.0).to_numpy(dtype=float)
	y = df[['x', 'y']].to_numpy(dtype=float)
	preds = model.predict(X)
	errors = np.linalg.norm(preds - y, axis=1)
	results = df[['x', 'y']].copy()
	results['error_cm'] = errors
	preds_df = pd.DataFrame({
		'pred_x': preds[:, 0],
		'pred_y': preds[:, 1],
		'true_x': y[:, 0],
		'true_y': y[:, 1],
		'error_cm': errors,
	})
	return results, preds_df


def plot_accuracy_map(results: pd.DataFrame, filename: str, title: str) -> None:
	plt.figure(figsize=(8, 6))
	sc = plt.scatter(results['x'], results['y'], c=results['error_cm'], cmap='coolwarm', s=60, edgecolor='k')
	plt.colorbar(sc, label='Euclidean Error (cm)')
	plt.title(title)
	plt.xlabel('X (cm)')
	plt.ylabel('Y (cm)')
	_save_current_fig(filename)
	plt.close()


def plot_error_heatmap(results: pd.DataFrame, filename: str, title: str, bins: int = 15) -> None:
	if results.empty:
		return
	grid = results.copy()
	grid['x_bin'] = pd.cut(grid['x'], bins=bins)
	grid['y_bin'] = pd.cut(grid['y'], bins=bins)
	pivot = grid.pivot_table(index='y_bin', columns='x_bin', values='error_cm', aggfunc='mean')
	plt.figure(figsize=(8, 6))
	sns.heatmap(pivot.iloc[::-1], cmap='coolwarm', cbar_kws={'label': 'Mean error (cm)'})
	plt.title(title)
	plt.xlabel('X bin')
	plt.ylabel('Y bin')
	_save_current_fig(filename)
	plt.close()


def plot_error_histogram(errors: Iterable[float], tag: str) -> None:
	errors = np.asarray(list(errors), dtype=float)
	plt.figure(figsize=(8, 5))
	sns.histplot(errors, bins=30, kde=True, color='tab:purple')
	plt.title(f'Distribution of absolute errors ({tag})')
	plt.xlabel('Euclidean error (cm)')
	plt.ylabel('Count')
	_save_current_fig(f'error_histogram_{tag}.png')
	plt.close()


def plot_pred_vs_true(df: pd.DataFrame, tag: str) -> None:
	plt.figure(figsize=(12, 5))
	plt.subplot(1, 2, 1)
	sns.scatterplot(x='true_x', y='pred_x', data=df, alpha=0.6)
	plt.plot([df['true_x'].min(), df['true_x'].max()], [df['true_x'].min(), df['true_x'].max()], 'k--')
	plt.xlabel('True X')
	plt.ylabel('Predicted X')
	plt.title(f'Predicted vs True X ({tag})')

	plt.subplot(1, 2, 2)
	sns.scatterplot(x='true_y', y='pred_y', data=df, alpha=0.6, color='tab:orange')
	plt.plot([df['true_y'].min(), df['true_y'].max()], [df['true_y'].min(), df['true_y'].max()], 'k--')
	plt.xlabel('True Y')
	plt.ylabel('Predicted Y')
	plt.title(f'Predicted vs True Y ({tag})')

	_save_current_fig(f'pred_vs_true_{tag}.png')
	plt.close()


def main() -> None:
	real_fracs = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
	k_values = list(range(1, 26, 2))

	#print('Running hyperparameter sweeps...')
	#hp_df = run_hyperparameter_search(real_fracs, k_values)
	#plot_hyperparameter_search(hp_df)
#
	#print('Plotting wifi null percentages...')
	#wifi_csv = os.path.join(DATA_DIR, 'wifi.csv')
	#plot_wifi_null_summary(wifi_csv)
#
	#print('Comparing real vs synthetic distributions...')
	#real_df = load_real_combined()
	#synth_df = pd.read_csv(SYNTHETIC_PATH) if os.path.exists(SYNTHETIC_PATH) else pd.DataFrame()
	#plot_real_vs_synth_distributions(real_df, synth_df)

	print('Analyzing historical (old vs new) feature drift (standardized view only)...')
	run_feature_drift_pipeline(label='standardized')

	#print('Training baseline model and evaluating on real test split...')
	#model, features, real_results, real_preds = train_baseline_model()
	#plot_accuracy_map(real_results, 'accuracy_map_real.png', 'Spatial accuracy (real test set)')
	#plot_error_heatmap(real_results, 'accuracy_heatmap_real.png', 'Mean error heatmap (real test set)')
	#plot_error_histogram(real_results['error_cm'], 'real')
	#plot_pred_vs_true(real_preds, 'real')

	#if not synth_df.empty:
	#	print('Evaluating baseline model on synthetic dataset...')
	#	synth_results, synth_preds = evaluate_on_dataframe(model, features, synth_df, 'synthetic')
	#	plot_accuracy_map(synth_results, 'accuracy_map_synthetic.png', 'Spatial accuracy (synthetic samples)')
	#	plot_error_heatmap(synth_results, 'accuracy_heatmap_synthetic.png', 'Mean error heatmap (synthetic samples)')
	#	plot_error_histogram(synth_results['error_cm'], 'synthetic')
	#	plot_pred_vs_true(synth_preds, 'synthetic')
	#else:
	#	print('Synthetic dataset not found; skipping synthetic evaluation plots.')
#
	print('All figures saved under:', FIG_DIR)


if __name__ == '__main__':
	main()
