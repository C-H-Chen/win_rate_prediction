import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.inspection import PartialDependenceDisplay

class DragonWinRateAnalyzer:
    def __init__(self, model_path='dragon_model.pkl', scaler_path='dragon_scaler.pkl'):
        """
        初始化分析器，設定屬性並執行前置
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        # 定義核心物件
        self.dragon_model = None
        self.scaler = None
        self.predict_data = None
        self.features = []
        self.features_range = {}
        self.predictions_dict = {}

        # 啟動時自動載入模型與資料
        self._load_models()
        self._initialize_data()

    def _load_models(self):
        """載入模型與標準化工具"""
        self.dragon_model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

    def _initialize_data(self):
        """私有方法：初始化基準預測資料與特徵範圍"""
        data = {
            'FIP': [3.300939],
            'ERA': [3.38],
            'P_BABIP': [0.286468],
            'P_BB%': [0.077940],
            'OPS': [0.668],
            'OBP': [0.315],
            'ISO': [0.097],
            'H_K%': [0.191627],
            'Defense%': [0.982],
            'wOBA': [0.295344],
            'WHIP': [1.512000]
        }
        self.predict_data = pd.DataFrame(data)

        # 特徵
        self.features = ['FIP', 'ERA', 'P_BABIP', 'P_BB%', 'OPS', 'OBP', 'ISO', 'H_K%', 'Defense%', 'wOBA', 'WHIP']

        # 範圍
        self.features_range = {
            'FIP': np.linspace(3.5, 4, 5),
            'ERA': np.linspace(3.5, 4, 5),
            'P_BABIP': np.linspace(0.28, 0.36, 5),
            'P_BB%': np.linspace(0.5, 1.5, 5),
            'OPS':  np.linspace(0.74, 0.78, 5),
            'OBP':  np.linspace(0.2, 0.4, 5),
            'ISO':  np.linspace(0.06, 0.1, 5),
            'H_K%': np.linspace(0.05, 0.2, 5),
            'Defense%': np.linspace(0.8, 0.9, 5),
            'wOBA': np.linspace(0.327, 0.340, 5),
            'WHIP': np.linspace(1.0, 1.4, 5)
        }

    def run_sensitivity_analysis(self):
        """靈敏度分析：對每個特徵的不同值進行預測並繪圖"""
        for feature in self.features:
            feature_values = self.features_range[feature]  # 獲取當前特徵範圍
            predictions = []

            for value in feature_values:    
                temp_data = self.predict_data.copy() # 複製預測資料並修改指定特徵值
                temp_data[feature] = value  # 將當前特徵值設為新值

                # 標準化
                temp_data_scaled = self.scaler.transform(temp_data)
                temp_data_scaled = pd.DataFrame(temp_data_scaled, columns=temp_data.columns)

                # 預測並存儲結果
                prediction = self.dragon_model.predict(temp_data_scaled)
                predictions.append(prediction[0])

            self.predictions_dict[feature] = predictions

            # 繪圖
            self._plot_marginal_effect(feature, feature_values, predictions)

    def _plot_marginal_effect(self, feature, feature_values, predictions):
        """繪製每個特徵的邊際效應圖"""
        plt.figure(figsize=(8, 6))
        plt.plot(feature_values, predictions, marker='o', linestyle='-', color='b')
        plt.xlabel(feature)
        plt.ylabel('Predicted Win Rate')
        plt.title(f'Marginal Effect of {feature} on Predicted Win Rate')
        plt.grid(True)
        plt.show()

    def predict_baseline_scenario(self):
        """
        封裝保留日後可擴充獨立測試基礎預測值
        """
        '''
        predict_data = {
            'FIP': [3.300939],
            'ERA': [3.38],
            'P_BABIP': [0.286468],
            'P_BB%': [0.077940],
            'OPS': [0.668],
            'OBP': [0.315],
            'ISO': [0.097],
            'H_K%': [0.191627],
            'Defense%': [0.982],
            'wOBA': [0.295344],
            'WHIP': [1.512000]
        }
        predict_data_df = pd.DataFrame(predict_data)

        predict_data_scaled = self.scaler.transform(predict_data_df)

        predictions = self.dragon_model.predict(predict_data_scaled)

        print("假如味全龍球員個人成績都與去年相當，那麼2025年勝率預計是")
        print(predictions)
        '''
        pass


if __name__ == "__main__":
    # 建立分析器實例並執行分析
    analyzer = DragonWinRateAnalyzer()
    analyzer.run_sensitivity_analysis()
