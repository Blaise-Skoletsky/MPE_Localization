# Models Metadata

This folder now keeps a single metadata file per modality:

- `spatial_wifi_meta.json` – contains both the **pre-PCA** sensor columns (the ones saved in `data/wifi_pre_pca.csv` and used to train the GP models) and the **post-PCA** columns (the dimensionality-reduced features saved in `data/wifi_cleaned.csv`, used by the KNN + realtime localization pipelines). The JSON shape is:

```json
{
  "pre_pca": {
    "columns": ["<original_wifi_col>", "..."],
    "num_columns": <int>
  },
  "post_pca": {
    "columns": ["wifi_pca_0", "wifi_pca_1", "..."],
    "num_columns": <int>
  },
  "pca": {
    "enabled": true,
    "variance_threshold": 0.9,
    "components_retained": 24,
    "explained_variance_cumulative": 0.95
  }
}
```

- `spatial_light_meta.json` – light has no PCA, so it simply stores `{ "columns": [...], "num_columns": N }`.

Companion files:

- `data/wifi_pre_pca.csv` – standardized wifi data before PCA (used for GP training).
- `data/wifi_cleaned.csv` – PCA-projected wifi data (consumed by KNN + realtime localization).
- `models/spatial_wifi_scaler.pkl` – the per-column means/stds (plus global min/max) applied before PCA.
- `models/spatial_wifi_pca.pkl` – the fitted PCA transformer used to project wifi readings.
- `models/spatial_light_scaler.pkl` – per-column min/max/mean/std statistics used to normalize + standardize light readings.

Code can now pick whichever representation it needs by reading the appropriate section from `spatial_wifi_meta.json`.
