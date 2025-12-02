import pandas as pd

paths = {
    'wifi_pre_pca': 'data/wifi_pre_pca.csv',
    'wifi_cleaned': 'data/wifi_cleaned.csv',
    'light': 'data/light_cleaned.csv'
}

for name, path in paths.items():
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(name, 'missing:', e)
        continue
    coords = df[[c for c in ['x','y'] if c in df.columns]]
    vals = df.drop(columns=[c for c in ['timestamp','x','y'] if c in df.columns])
    print(f"-- {name} -- rows={len(df)} unique_coords={len(coords.drop_duplicates()) if not coords.empty else 'n/a'} cols={vals.shape[1]}")
    if not coords.empty:
        print('  sample coords unique counts:', coords['x'].nunique(), coords['y'].nunique())
    var = vals.var()
    print('  per-col var min/max:', float(var.min()), float(var.max()))
    print('  any-zero-variance cols:', int((var == 0).sum()))
    print('  first 3 rows mean of vals:', vals.iloc[:3].mean().values[:5])
    print()
