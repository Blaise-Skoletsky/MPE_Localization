"""
使用 new_light_cleaned.csv 訓練 Random Forest 模型 (80/20 分割)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("使用 new_light_cleaned.csv 訓練 Random Forest (80/20 分割)")
print("="*70)

# 1. 載入資料
print("\n【步驟 1】載入資料...")
data = pd.read_csv('new_light_cleaned.csv')
print(f"資料形狀: {data.shape}")
print(f"樣本數: {len(data)}")
print(f"X 範圍: {data['x'].min()} ~ {data['x'].max()}")
print(f"Y 範圍: {data['y'].min()} ~ {data['y'].max()}")
print(f"特徵欄位: {[col for col in data.columns if col not in ['timestamp', 'x', 'y']]}")

# 2. 準備特徵和標籤
print("\n【步驟 2】準備特徵...")
X = data.drop(['timestamp', 'x', 'y'], axis=1, errors='ignore')
y = data[['x', 'y']]

feature_names = list(X.columns)
print(f"特徵數: {len(feature_names)}")
print(f"特徵: {feature_names}")

# 3. 80/20 分割
print("\n【步驟 3】分割資料 (80% 訓練, 20% 測試)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"訓練集: {len(X_train)} 樣本")
print(f"測試集: {len(X_test)} 樣本")

# 4. 訓練模型
print("\n【步驟 4】訓練 Random Forest 模型...")
model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
)

model.fit(X_train, y_train)
print("\n模型訓練完成!")

# 5. 預測
print("\n【步驟 5】評估模型...")

# 訓練集預測
y_train_pred = model.predict(X_train)
train_distances = np.sqrt(
    (y_train['x'].values - y_train_pred[:, 0])**2 + 
    (y_train['y'].values - y_train_pred[:, 1])**2
)

# 測試集預測
y_test_pred = model.predict(X_test)
test_distances = np.sqrt(
    (y_test['x'].values - y_test_pred[:, 0])**2 + 
    (y_test['y'].values - y_test_pred[:, 1])**2
)

# 6. 顯示結果
print("\n" + "="*70)
print("訓練集結果")
print("="*70)
print(f"X 座標 MAE: {mean_absolute_error(y_train['x'], y_train_pred[:, 0]):.2f} cm")
print(f"Y 座標 MAE: {mean_absolute_error(y_train['y'], y_train_pred[:, 1]):.2f} cm")
print(f"平均定位誤差: {train_distances.mean():.2f} cm")
print(f"中位數誤差: {np.median(train_distances):.2f} cm")
print(f"誤差 <= 100cm (1m): {(train_distances <= 100).sum() / len(train_distances) * 100:.2f}%")
print(f"誤差 <= 200cm (2m): {(train_distances <= 200).sum() / len(train_distances) * 100:.2f}%")

print("\n" + "="*70)
print("測試集結果")
print("="*70)
print(f"X 座標 MAE: {mean_absolute_error(y_test['x'], y_test_pred[:, 0]):.2f} cm")
print(f"Y 座標 MAE: {mean_absolute_error(y_test['y'], y_test_pred[:, 1]):.2f} cm")
print(f"平均定位誤差: {test_distances.mean():.2f} cm")
print(f"中位數誤差: {np.median(test_distances):.2f} cm")
print(f"誤差 <= 100cm (1m): {(test_distances <= 100).sum() / len(test_distances) * 100:.2f}%")
print(f"誤差 <= 200cm (2m): {(test_distances <= 200).sum() / len(test_distances) * 100:.2f}%")

print(f"\n預測 X 範圍: {y_test_pred[:, 0].min():.1f} ~ {y_test_pred[:, 0].max():.1f}")
print(f"預測 Y 範圍: {y_test_pred[:, 1].min():.1f} ~ {y_test_pred[:, 1].max():.1f}")

# 7. 儲存模型
print("\n【步驟 6】儲存模型...")
model_file = 'new_light_rf_model.joblib'
features_file = 'new_light_rf_features.json'

joblib.dump(model, model_file)
with open(features_file, 'w') as f:
    json.dump(feature_names, f)

print(f"模型已儲存: {model_file}")
print(f"特徵已儲存: {features_file}")

# 8. 儲存測試集預測結果
results_df = pd.DataFrame({
    'actual_x': y_test['x'].values,
    'actual_y': y_test['y'].values,
    'predicted_x': y_test_pred[:, 0],
    'predicted_y': y_test_pred[:, 1],
    'error_x': np.abs(y_test['x'].values - y_test_pred[:, 0]),
    'error_y': np.abs(y_test['y'].values - y_test_pred[:, 1]),
    'euclidean_distance_error': test_distances
})

results_file = 'new_light_rf_predictions.csv'
results_df.to_csv(results_file, index=False)
print(f"預測結果已儲存: {results_file}")

# 9. 特徵重要性
print("\n【步驟 7】特徵重要性分析...")
importance_x = model.estimators_[0].feature_importances_
importance_y = model.estimators_[1].feature_importances_
avg_importance = (importance_x + importance_y) / 2

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance_X': importance_x,
    'Importance_Y': importance_y,
    'Avg_Importance': avg_importance
}).sort_values('Avg_Importance', ascending=False)

print("\n特徵重要性排序:")
print(feature_importance.to_string(index=False))

feature_importance.to_csv('new_light_rf_feature_importance.csv', index=False)
print("\n特徵重要性已儲存: new_light_rf_feature_importance.csv")

# 10. 生成視覺化
print("\n【步驟 8】生成視覺化...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. 真實 vs 預測位置
ax1 = axes[0, 0]
ax1.scatter(y_test['x'], y_test['y'], c='blue', s=30, alpha=0.5, label='真實位置')
ax1.scatter(y_test_pred[:, 0], y_test_pred[:, 1], c='red', s=30, alpha=0.5, marker='x', label='預測位置')
for i in range(0, len(y_test), 5):
    ax1.plot([y_test.iloc[i]['x'], y_test_pred[i, 0]], 
             [y_test.iloc[i]['y'], y_test_pred[i, 1]], 
             'gray', alpha=0.2, linewidth=0.5)
ax1.set_xlabel('X 座標 (cm)')
ax1.set_ylabel('Y 座標 (cm)')
ax1.set_title(f'測試集: 真實 vs 預測位置 (n={len(y_test)})', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(250, 750)
ax1.set_ylim(125, 625)

# 2. 誤差熱力圖
ax2 = axes[0, 1]
scatter = ax2.scatter(y_test['x'], y_test['y'], c=test_distances, 
                      cmap='RdYlGn_r', s=40, alpha=0.7)
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('定位誤差 (cm)')
ax2.set_xlabel('X 座標 (cm)')
ax2.set_ylabel('Y 座標 (cm)')
ax2.set_title('測試集誤差分布', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(250, 750)
ax2.set_ylim(125, 625)

# 3. 誤差直方圖
ax3 = axes[0, 2]
ax3.hist(test_distances, bins=50, edgecolor='black', alpha=0.7, color='purple')
ax3.axvline(test_distances.mean(), color='red', linestyle='--', 
            linewidth=2, label=f'平均: {test_distances.mean():.2f} cm')
ax3.axvline(np.median(test_distances), color='green', linestyle='--', 
            linewidth=2, label=f'中位數: {np.median(test_distances):.2f} cm')
ax3.set_xlabel('歐式距離誤差 (cm)')
ax3.set_ylabel('樣本數')
ax3.set_title('測試集誤差分布', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 4. X-Y 預測準確度
ax4 = axes[1, 0]
ax4.scatter(y_test['x'], y_test_pred[:, 0], alpha=0.5, s=20, label='X 座標')
ax4.scatter(y_test['y'], y_test_pred[:, 1], alpha=0.5, s=20, label='Y 座標')
min_val = min(y_test['x'].min(), y_test['y'].min())
max_val = max(y_test['x'].max(), y_test['y'].max())
ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美預測')
ax4.set_xlabel('實際座標 (cm)')
ax4.set_ylabel('預測座標 (cm)')
ax4.set_title('座標預測準確度', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. 特徵重要性
ax5 = axes[1, 1]
colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
bars = ax5.barh(feature_importance['Feature'], feature_importance['Avg_Importance'], color=colors)
ax5.set_xlabel('平均重要性')
ax5.set_title('特徵重要性', fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')
ax5.invert_yaxis()

# 6. 各位置平均誤差
ax6 = axes[1, 2]
location_errors = results_df.groupby(['actual_x', 'actual_y']).agg({
    'euclidean_distance_error': ['mean', 'count']
}).reset_index()
location_errors.columns = ['x', 'y', 'mean_error', 'count']

scatter2 = ax6.scatter(location_errors['x'], location_errors['y'], 
                       c=location_errors['mean_error'], 
                       s=location_errors['count'] * 10,
                       cmap='RdYlGn_r', alpha=0.7, edgecolors='black', linewidths=0.5)
cbar2 = plt.colorbar(scatter2, ax=ax6)
cbar2.set_label('平均誤差 (cm)')
ax6.set_xlabel('X 座標 (cm)')
ax6.set_ylabel('Y 座標 (cm)')
ax6.set_title('各位置平均誤差', fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.set_xlim(250, 750)
ax6.set_ylim(125, 625)

plt.tight_layout()
plt.savefig('new_light_rf_results.png', dpi=300, bbox_inches='tight')
print("已儲存: new_light_rf_results.png")

plt.show()

print("\n" + "="*70)
print("完成!")
print("="*70)
