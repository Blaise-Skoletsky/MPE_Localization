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
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from kNN_Model import (
	build_feature_list,
	fit_knn_model,
	prepare_datasets,
	load_real_combined,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SYNTHETIC_PATH = os.path.join(BASE_DIR, 'synthetic_data', 'synthetic_knn_points.csv')
FIG_DIR = os.path.join(BASE_DIR, 'figures', 'presentation')
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style='whitegrid')


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
	sns.barplot(x=top_series.values * 100, y=top_series.index, palette='viridis')
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

	print('Running hyperparameter sweeps...')
	hp_df = run_hyperparameter_search(real_fracs, k_values)
	plot_hyperparameter_search(hp_df)

	print('Plotting wifi null percentages...')
	wifi_csv = os.path.join(DATA_DIR, 'wifi.csv')
	plot_wifi_null_summary(wifi_csv)

	print('Comparing real vs synthetic distributions...')
	real_df = load_real_combined()
	synth_df = pd.read_csv(SYNTHETIC_PATH) if os.path.exists(SYNTHETIC_PATH) else pd.DataFrame()
	plot_real_vs_synth_distributions(real_df, synth_df)

	print('Training baseline model and evaluating on real test split...')
	model, features, real_results, real_preds = train_baseline_model()
	plot_accuracy_map(real_results, 'accuracy_map_real.png', 'Spatial accuracy (real test set)')
	plot_error_heatmap(real_results, 'accuracy_heatmap_real.png', 'Mean error heatmap (real test set)')
	plot_error_histogram(real_results['error_cm'], 'real')
	plot_pred_vs_true(real_preds, 'real')

	if not synth_df.empty:
		print('Evaluating baseline model on synthetic dataset...')
		synth_results, synth_preds = evaluate_on_dataframe(model, features, synth_df, 'synthetic')
		plot_accuracy_map(synth_results, 'accuracy_map_synthetic.png', 'Spatial accuracy (synthetic samples)')
		plot_error_heatmap(synth_results, 'accuracy_heatmap_synthetic.png', 'Mean error heatmap (synthetic samples)')
		plot_error_histogram(synth_results['error_cm'], 'synthetic')
		plot_pred_vs_true(synth_preds, 'synthetic')
	else:
		print('Synthetic dataset not found; skipping synthetic evaluation plots.')

	print('All figures saved under:', FIG_DIR)


if __name__ == '__main__':
	main()
