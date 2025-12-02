"""
RF 模型推論程式

使用訓練好的 Random Forest 模型進行位置推論
支援 WiFi、Light 以及加權融合的推論方式
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class RFLocalizer:
    """Random Forest 定位推論器"""
    
    def __init__(
        self,
        wifi_model_path: str = None,
        wifi_features_path: str = None,
        light_model_path: str = None,
        light_features_path: str = None,
        wifi_weight: float = 0.5,
        light_weight: float = 0.5,
        verbose: bool = True
    ):
        """
        初始化推論器
        
        Parameters:
        -----------
        wifi_model_path : str
            WiFi RF 模型路徑 (.joblib)
        wifi_features_path : str
            WiFi 特徵列表路徑 (.json)
        light_model_path : str
            Light RF 模型路徑 (.joblib)
        light_features_path : str
            Light 特徵列表路徑 (.json)
        wifi_weight : float
            WiFi 模型的權重 (0~1)
        light_weight : float
            Light 模型的權重 (0~1)
        verbose : bool
            是否顯示詳細資訊
        """
        self.verbose = verbose
        
        # 設定預設路徑
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        if wifi_model_path is None:
            wifi_model_path = os.path.join(base_dir, 'new_wifi_rf_model.joblib')
        if wifi_features_path is None:
            wifi_features_path = os.path.join(base_dir, 'new_wifi_rf_features.json')
        if light_model_path is None:
            light_model_path = os.path.join(base_dir, 'new_light_rf_model.joblib')
        if light_features_path is None:
            light_features_path = os.path.join(base_dir, 'new_light_rf_features.json')
        
        # 載入模型
        self.wifi_model = None
        self.wifi_features = None
        self.light_model = None
        self.light_features = None
        
        # 載入 WiFi 模型
        if os.path.exists(wifi_model_path) and os.path.exists(wifi_features_path):
            self.wifi_model = joblib.load(wifi_model_path)
            with open(wifi_features_path, 'r') as f:
                self.wifi_features = json.load(f)
            if self.verbose:
                print(f"[WiFi] 模型載入成功，特徵數: {len(self.wifi_features)}")
        else:
            if self.verbose:
                print(f"[WiFi] 模型檔案不存在，跳過載入")
        
        # 載入 Light 模型
        if os.path.exists(light_model_path) and os.path.exists(light_features_path):
            self.light_model = joblib.load(light_model_path)
            with open(light_features_path, 'r') as f:
                self.light_features = json.load(f)
            if self.verbose:
                print(f"[Light] 模型載入成功，特徵數: {len(self.light_features)}")
        else:
            if self.verbose:
                print(f"[Light] 模型檔案不存在，跳過載入")
        
        # 設定權重
        self.set_weights(wifi_weight, light_weight)
        
        if self.verbose:
            print(f"[權重] WiFi: {self.wifi_weight:.2f}, Light: {self.light_weight:.2f}")
    
    def set_weights(self, wifi_weight: float, light_weight: float) -> None:
        """
        設定加權權重
        
        Parameters:
        -----------
        wifi_weight : float
            WiFi 模型權重
        light_weight : float
            Light 模型權重
        """
        total = wifi_weight + light_weight
        if total == 0:
            raise ValueError("權重總和不能為 0")
        
        # 正規化權重
        self.wifi_weight = wifi_weight / total
        self.light_weight = light_weight / total
    
    def _prepare_wifi_features(
        self, 
        data: Union[Dict, pd.DataFrame, np.ndarray],
        fill_missing: float = -100.0
    ) -> np.ndarray:
        """
        準備 WiFi 特徵向量
        
        Parameters:
        -----------
        data : dict, DataFrame, or ndarray
            輸入資料
        fill_missing : float
            缺失值填充值
            
        Returns:
        --------
        np.ndarray : 特徵向量
        """
        if self.wifi_features is None:
            raise ValueError("WiFi 模型未載入")
        
        if isinstance(data, dict):
            features = []
            for feat in self.wifi_features:
                if feat in data:
                    features.append(float(data[feat]))
                else:
                    features.append(fill_missing)
            return np.array(features).reshape(1, -1)
        
        elif isinstance(data, pd.DataFrame):
            # 確保所有需要的特徵都存在
            for feat in self.wifi_features:
                if feat not in data.columns:
                    data[feat] = fill_missing
            return data[self.wifi_features].values
        
        elif isinstance(data, np.ndarray):
            return data.reshape(1, -1) if data.ndim == 1 else data
        
        else:
            raise TypeError(f"不支援的資料類型: {type(data)}")
    
    def _prepare_light_features(
        self, 
        data: Union[Dict, pd.DataFrame, np.ndarray],
        fill_missing: float = 0.0
    ) -> np.ndarray:
        """
        準備 Light 特徵向量
        
        Parameters:
        -----------
        data : dict, DataFrame, or ndarray
            輸入資料
        fill_missing : float
            缺失值填充值
            
        Returns:
        --------
        np.ndarray : 特徵向量
        """
        if self.light_features is None:
            raise ValueError("Light 模型未載入")
        
        if isinstance(data, dict):
            features = []
            for feat in self.light_features:
                if feat in data:
                    features.append(float(data[feat]))
                else:
                    features.append(fill_missing)
            return np.array(features).reshape(1, -1)
        
        elif isinstance(data, pd.DataFrame):
            # 確保所有需要的特徵都存在
            for feat in self.light_features:
                if feat not in data.columns:
                    data[feat] = fill_missing
            return data[self.light_features].values
        
        elif isinstance(data, np.ndarray):
            return data.reshape(1, -1) if data.ndim == 1 else data
        
        else:
            raise TypeError(f"不支援的資料類型: {type(data)}")
    
    def predict_wifi(
        self, 
        data: Union[Dict, pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        使用 WiFi 模型進行預測
        
        Parameters:
        -----------
        data : dict, DataFrame, or ndarray
            WiFi RSSI 資料
            
        Returns:
        --------
        np.ndarray : 預測座標 [[x, y], ...]
        """
        if self.wifi_model is None:
            raise ValueError("WiFi 模型未載入")
        
        X = self._prepare_wifi_features(data)
        return self.wifi_model.predict(X)
    
    def predict_light(
        self, 
        data: Union[Dict, pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        使用 Light 模型進行預測
        
        Parameters:
        -----------
        data : dict, DataFrame, or ndarray
            光感測器資料
            
        Returns:
        --------
        np.ndarray : 預測座標 [[x, y], ...]
        """
        if self.light_model is None:
            raise ValueError("Light 模型未載入")
        
        X = self._prepare_light_features(data)
        return self.light_model.predict(X)
    
    def predict_weighted(
        self, 
        wifi_data: Union[Dict, pd.DataFrame, np.ndarray] = None,
        light_data: Union[Dict, pd.DataFrame, np.ndarray] = None,
        wifi_weight: float = None,
        light_weight: float = None
    ) -> np.ndarray:
        """
        使用加權融合進行預測
        
        Parameters:
        -----------
        wifi_data : dict, DataFrame, or ndarray
            WiFi RSSI 資料
        light_data : dict, DataFrame, or ndarray
            光感測器資料
        wifi_weight : float, optional
            WiFi 權重 (覆蓋預設值)
        light_weight : float, optional
            Light 權重 (覆蓋預設值)
            
        Returns:
        --------
        np.ndarray : 加權融合後的預測座標 [[x, y], ...]
        """
        # 使用提供的權重或預設權重
        w_wifi = wifi_weight if wifi_weight is not None else self.wifi_weight
        w_light = light_weight if light_weight is not None else self.light_weight
        
        # 正規化權重
        total = w_wifi + w_light
        w_wifi /= total
        w_light /= total
        
        predictions = []
        weights = []
        
        # WiFi 預測
        if wifi_data is not None and self.wifi_model is not None:
            pred_wifi = self.predict_wifi(wifi_data)
            predictions.append(pred_wifi)
            weights.append(w_wifi)
        
        # Light 預測
        if light_data is not None and self.light_model is not None:
            pred_light = self.predict_light(light_data)
            predictions.append(pred_light)
            weights.append(w_light)
        
        if len(predictions) == 0:
            raise ValueError("沒有可用的預測結果")
        
        if len(predictions) == 1:
            return predictions[0]
        
        # 正規化權重
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # 加權融合
        result = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            result += pred * weight
        
        return result
    
    def predict(
        self,
        wifi_data: Union[Dict, pd.DataFrame, np.ndarray] = None,
        light_data: Union[Dict, pd.DataFrame, np.ndarray] = None,
        mode: str = 'weighted'
    ) -> np.ndarray:
        """
        統一的預測介面
        
        Parameters:
        -----------
        wifi_data : dict, DataFrame, or ndarray
            WiFi RSSI 資料
        light_data : dict, DataFrame, or ndarray
            光感測器資料
        mode : str
            預測模式: 'wifi', 'light', 'weighted'
            
        Returns:
        --------
        np.ndarray : 預測座標 [[x, y], ...]
        """
        if mode == 'wifi':
            return self.predict_wifi(wifi_data)
        elif mode == 'light':
            return self.predict_light(light_data)
        elif mode == 'weighted':
            return self.predict_weighted(wifi_data, light_data)
        else:
            raise ValueError(f"不支援的模式: {mode}")
    
    def evaluate(
        self,
        wifi_data: pd.DataFrame = None,
        light_data: pd.DataFrame = None,
        y_true: np.ndarray = None,
        mode: str = 'weighted'
    ) -> Dict[str, float]:
        """
        評估模型效能
        
        Parameters:
        -----------
        wifi_data : DataFrame
            WiFi 測試資料
        light_data : DataFrame
            Light 測試資料
        y_true : ndarray
            真實座標 [[x, y], ...]
        mode : str
            預測模式
            
        Returns:
        --------
        dict : 評估指標
        """
        y_pred = self.predict(wifi_data, light_data, mode=mode)
        
        # 計算歐式距離誤差
        distances = np.sqrt(
            (y_true[:, 0] - y_pred[:, 0])**2 + 
            (y_true[:, 1] - y_pred[:, 1])**2
        )
        
        metrics = {
            'mean_error': np.mean(distances),
            'median_error': np.median(distances),
            'std_error': np.std(distances),
            'max_error': np.max(distances),
            'min_error': np.min(distances),
            'mae_x': np.mean(np.abs(y_true[:, 0] - y_pred[:, 0])),
            'mae_y': np.mean(np.abs(y_true[:, 1] - y_pred[:, 1])),
            'accuracy_1m': (distances <= 100).sum() / len(distances) * 100,
            'accuracy_2m': (distances <= 200).sum() / len(distances) * 100,
        }
        
        return metrics


def demo_single_prediction():
    """示範單筆資料推論"""
    print("\n" + "="*70)
    print("示範：單筆資料推論")
    print("="*70)
    
    # 初始化推論器
    localizer = RFLocalizer(wifi_weight=0.6, light_weight=0.4)
    
    # 模擬 WiFi 資料 (RSSI 值)
    wifi_sample = {
        "00:24:6C:26:87:60": -100.0,
        "00:24:6C:26:87:61": -100.0,
        "00:24:6C:26:87:62": -78.0,
        "30:87:D9:31:95:E9": -60.0,
        "30:87:D9:31:95:EC": -81.0,
        # ... 其他 AP RSSI
    }
    
    # 模擬 Light 資料
    light_sample = {
        "f1": 737.0,
        "f2": 2539.0,
        "f3": 2447.0,
        "f4": 2021.0,
        "f5": 9204.0,
        "f6": 5918.0,
        "f7": 6267.0,
        "f8": 2213.0,
        "clear": 9040.0,
        "nir": 760.0
    }
    
    # WiFi 預測
    try:
        pred_wifi = localizer.predict_wifi(wifi_sample)
        print(f"\n[WiFi 預測] X: {pred_wifi[0, 0]:.2f} cm, Y: {pred_wifi[0, 1]:.2f} cm")
    except Exception as e:
        print(f"\n[WiFi 預測] 失敗: {e}")
    
    # Light 預測
    try:
        pred_light = localizer.predict_light(light_sample)
        print(f"[Light 預測] X: {pred_light[0, 0]:.2f} cm, Y: {pred_light[0, 1]:.2f} cm")
    except Exception as e:
        print(f"[Light 預測] 失敗: {e}")
    
    # 加權預測
    try:
        pred_weighted = localizer.predict_weighted(wifi_sample, light_sample)
        print(f"[加權預測] X: {pred_weighted[0, 0]:.2f} cm, Y: {pred_weighted[0, 1]:.2f} cm")
        print(f"           (WiFi 權重: {localizer.wifi_weight:.2f}, Light 權重: {localizer.light_weight:.2f})")
    except Exception as e:
        print(f"[加權預測] 失敗: {e}")


def demo_batch_prediction():
    """示範批次資料推論與評估"""
    print("\n" + "="*70)
    print("示範：批次資料推論與評估")
    print("="*70)
    
    # 初始化推論器
    localizer = RFLocalizer(wifi_weight=0.5, light_weight=0.5)
    
    # 載入測試資料
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wifi_data_path = os.path.join(base_dir, 'new_wifi_cleaned.csv')
    light_data_path = os.path.join(base_dir, 'new_light_cleaned.csv')
    
    if not os.path.exists(wifi_data_path) or not os.path.exists(light_data_path):
        print("測試資料檔案不存在，跳過批次測試")
        return
    
    # 載入資料
    wifi_df = pd.read_csv(wifi_data_path)
    light_df = pd.read_csv(light_data_path)
    
    # 取得真實座標
    y_true = wifi_df[['x', 'y']].values
    
    # 準備特徵
    wifi_features = wifi_df.drop(['timestamp', 'x', 'y'], axis=1, errors='ignore')
    light_features = light_df.drop(['timestamp', 'x', 'y'], axis=1, errors='ignore')
    
    # 測試不同的權重組合
    weight_combinations = [
        (1.0, 0.0, 'WiFi only'),
        (0.0, 1.0, 'Light only'),
        (0.5, 0.5, 'Equal weight (0.5/0.5)'),
        (0.7, 0.3, 'WiFi heavy (0.7/0.3)'),
        (0.3, 0.7, 'Light heavy (0.3/0.7)'),
    ]
    
    print("\n不同權重組合的評估結果:")
    print("-" * 70)
    print(f"{'模式':<25} {'平均誤差':>12} {'中位誤差':>12} {'1m內準確率':>12} {'2m內準確率':>12}")
    print("-" * 70)
    
    for w_wifi, w_light, name in weight_combinations:
        localizer.set_weights(w_wifi, w_light)
        
        if w_wifi == 0:
            mode = 'light'
            metrics = localizer.evaluate(
                light_data=light_features, 
                y_true=y_true, 
                mode=mode
            )
        elif w_light == 0:
            mode = 'wifi'
            metrics = localizer.evaluate(
                wifi_data=wifi_features, 
                y_true=y_true, 
                mode=mode
            )
        else:
            metrics = localizer.evaluate(
                wifi_data=wifi_features,
                light_data=light_features,
                y_true=y_true,
                mode='weighted'
            )
        
        print(f"{name:<25} {metrics['mean_error']:>10.2f}cm {metrics['median_error']:>10.2f}cm "
              f"{metrics['accuracy_1m']:>11.1f}% {metrics['accuracy_2m']:>11.1f}%")
    
    print("-" * 70)


def demo_realtime_simulation():
    """模擬即時定位場景"""
    print("\n" + "="*70)
    print("示範：即時定位模擬")
    print("="*70)
    
    # 初始化推論器
    localizer = RFLocalizer(wifi_weight=0.6, light_weight=0.4, verbose=False)
    
    # 載入資料作為模擬
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wifi_df = pd.read_csv(os.path.join(base_dir, 'new_wifi_cleaned.csv'))
    light_df = pd.read_csv(os.path.join(base_dir, 'new_light_cleaned.csv'))
    
    # 模擬連續定位 (取前 10 筆)
    print("\n模擬即時定位 (前 10 筆資料):")
    print("-" * 70)
    print(f"{'時間戳':>15} {'真實 X':>10} {'真實 Y':>10} {'預測 X':>10} {'預測 Y':>10} {'誤差':>10}")
    print("-" * 70)
    
    for i in range(min(10, len(wifi_df))):
        # 取得當前樣本
        wifi_sample = wifi_df.iloc[i].drop(['timestamp', 'x', 'y'], errors='ignore').to_dict()
        light_sample = light_df.iloc[i].drop(['timestamp', 'x', 'y'], errors='ignore').to_dict()
        
        true_x = wifi_df.iloc[i]['x']
        true_y = wifi_df.iloc[i]['y']
        timestamp = wifi_df.iloc[i]['timestamp']
        
        # 進行預測
        pred = localizer.predict_weighted(wifi_sample, light_sample)
        pred_x, pred_y = pred[0, 0], pred[0, 1]
        
        # 計算誤差
        error = np.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2)
        
        print(f"{timestamp:>15} {true_x:>10.1f} {true_y:>10.1f} "
              f"{pred_x:>10.1f} {pred_y:>10.1f} {error:>9.1f}cm")
    
    print("-" * 70)


if __name__ == '__main__':
    print("="*70)
    print("RF 模型推論程式示範")
    print("="*70)
    
    # 1. 單筆資料推論示範
    demo_single_prediction()
    
    # 2. 批次資料推論與評估
    demo_batch_prediction()
    
    # 3. 即時定位模擬
    demo_realtime_simulation()
    
    print("\n" + "="*70)
    print("示範完成!")
    print("="*70)
