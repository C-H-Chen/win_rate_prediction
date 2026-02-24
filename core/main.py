import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import learning_curve
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler


class CPBLWinRatePredictor:
    """
    中華職棒勝率預測分析管線 (CPBL Win Rate Prediction Pipeline)
    負責資料讀取、特徵工程、模型訓練調參、評估與可視化。
    """
    
    def __init__(self, file_path):
        # 檔案路徑與基礎狀態初始化
        self.file_path = file_path
        self.scaler = StandardScaler()
        self.best_model = None
        self.predictions = None
        
        # 資料集存放區
        self.df = None
        self.x_train_scaled = None
        self.x_test_scaled = None
        self.y_train = None
        self.y_test = None

    def execute_pipeline(self):
        """執行完整的機器學習管線"""
        self.load_and_preprocess_data()
        self.prepare_train_test_split()
        self.build_and_tune_model()
        self.evaluate_model()
        self.extract_feature_importances()
        self.visualize_results()

    def load_and_preprocess_data(self):
        """讀取資料並執行特徵工程"""
        
        # 讀取 Excel 檔案
        xls = pd.ExcelFile(self.file_path)
        
        # 球員打擊(2010-24) 欄位缺失值的行位置:第 862 行 缺失值位置: ['銀棒指數']
        # 球員守備(2010-24) 欄位缺失值的行位置:第 2907 行 缺失值位置: ['守備率 ']
        # 98味全龍異常值
        # 手動處理完畢
        
        # 讀取工作表
        team_performance_df = pd.read_excel(xls, sheet_name='球隊成績')
        team_pitching_df = pd.read_excel(xls, sheet_name='球隊投手')
        team_hitting_df = pd.read_excel(xls, sheet_name='球隊打擊')
        team_defense_df = pd.read_excel(xls, sheet_name='球隊守備')
        player_hitting_df = pd.read_excel(xls, sheet_name='球員打擊(2010-24)')
        player_pitching_df = pd.read_excel(xls, sheet_name='球員投手(2010-24)')
        player_defense_df = pd.read_excel(xls, sheet_name='球員守備(2010-24)')

        # === 特徵工程 ===
        # 處理年度格式，判斷是否為上下年度
        team_performance_df['年度'] = team_performance_df['年度'].apply(lambda x: str(x).split('(')[0].strip() if '(' in str(x) else str(x))

        # 篩選年份範圍
        team_performance_df = team_performance_df[team_performance_df['年度'].astype(int).between(2010, 2024)]
        team_pitching_df = team_pitching_df[team_pitching_df['年度'].astype(int).between(2010, 2024)]
        team_defense_df = team_defense_df[team_defense_df['年度'].astype(int).between(2010, 2024)]

        # 轉換年度為字串
        team_performance_df['年度'] = team_performance_df['年度'].astype(str)
        team_pitching_df['年度'] = team_pitching_df['年度'].astype(str)
        team_hitting_df['年度'] = team_hitting_df['年度'].astype(str)
        team_defense_df['年度'] = team_defense_df['年度'].astype(str)
        player_pitching_df['年度'] = player_pitching_df['年度'].astype(str)
        player_hitting_df['年度'] = player_hitting_df['年度'].astype(str)
        player_defense_df['年度'] = player_defense_df['年度'].astype(str)

        # 計算全年
        team_performance_df_agg = team_performance_df.groupby(['球隊', '年度']).agg(
            全年勝率=('勝率', lambda x: x.sum() / 2),
            全年主場勝=('主場勝', 'sum'),
            全年主場敗=('主場敗', 'sum'),
            全年客場勝=('客場勝', 'sum'),
            全年客場敗=('客場敗', 'sum')
        ).reset_index()

        avg_era = team_pitching_df.groupby('年度')['防禦率'].mean()
        avg_fip = team_pitching_df.groupby('年度').apply(lambda x: ((13 * x['被全壘打'] + 3 * x['四壞球'] - 2 * x['奪三振']) / x['局數'] + 3.10).mean())
        avg_ops = team_hitting_df.groupby('年度').apply(lambda x: (x['上壘率'] + x['長打率']).mean())

        # 球隊成績
        team_performance_df_agg['Home-Away_diff'] = (team_performance_df_agg['全年主場勝'] - team_performance_df_agg['全年主場敗']) - (team_performance_df_agg['全年客場勝'] - team_performance_df_agg['全年客場敗']) 

        # 球隊投手
        team_pitching_df['FIP'] = ((13 * team_pitching_df['被全壘打'] + 3 * team_pitching_df['四壞球'] - 2 * team_pitching_df['奪三振']) / team_pitching_df['局數'] + 3.10) # / team_pitching_df['年度'].map(avg_fip))
        team_pitching_df['P_BABIP'] = (team_pitching_df['被安打'] - team_pitching_df['被全壘打']) / (team_pitching_df['面對打席'] - team_pitching_df['被全壘打'] - team_pitching_df['奪三振'] - team_pitching_df['四壞球'])
        team_pitching_df['ERA+'] = 100 * (team_pitching_df['年度'].map(avg_era) / team_pitching_df['防禦率'])
        team_pitching_df['K%'] = team_pitching_df['奪三振'] / team_pitching_df['面對打席']
        team_pitching_df['P_BB%'] = team_pitching_df['四壞球'] / team_pitching_df['面對打席']
        team_pitching_df['K/BB'] = (team_pitching_df['奪三振'] / team_pitching_df['四壞球'])
        team_pitching_df['HR%'] = team_pitching_df['被全壘打'] / team_pitching_df['面對打席']

        # 球隊打者
        team_hitting_df['OPS'] = (team_hitting_df['上壘率'] + team_hitting_df['長打率']) # / team_hitting_df['年度'].map(avg_ops))
        team_hitting_df['H_K%'] = team_hitting_df['三振'] / team_hitting_df['打數']
        team_hitting_df['SH/AB'] = team_hitting_df['犧牲短打'] / team_hitting_df['打數']
        team_hitting_df['SB/G'] = team_hitting_df['盜壘成功'] / team_hitting_df['出賽數']
        team_hitting_df['ISO'] = team_hitting_df['長打率'] - team_hitting_df['打擊率']
        team_hitting_df['PPG'] = team_hitting_df['得分'] - team_hitting_df['出賽數']

        # 球員打擊
        player_hitting_df_agg = player_hitting_df.groupby(['球隊', '年度']).agg(
            PA=('打席', 'sum'),
            AB=('打數', 'sum'),
            SH=('犧短', 'sum'),
            SF=('犧飛', 'sum'),
            one=('一安', 'sum'),
            double=('二安', 'sum'),
            triple=('三安', 'sum'),
            HR=('全壘打', 'sum'),
            BB=('四壞球', 'sum'),
            HBP=('死球', 'sum'),
            K=('被三振', 'sum'),
            SB_percent=('盜壘率', 'mean'),
        ).reset_index()
        player_hitting_df_agg['SH%'] = (player_hitting_df_agg['SH'] + player_hitting_df_agg['SF']) / player_hitting_df_agg['PA']
        player_hitting_df_agg['wOBA'] = ((0.69 * player_hitting_df_agg['BB'] + 0.72 * player_hitting_df_agg['HBP'] 
                                       + 0.9 * player_hitting_df_agg['one'] + 1.25 * player_hitting_df_agg['double']
                                       + 1.6 * player_hitting_df_agg['triple'] + 1.8 * player_hitting_df_agg['HR']) 
                                       / (player_hitting_df_agg['AB'] + player_hitting_df_agg['BB'] + player_hitting_df_agg['SF'] + player_hitting_df_agg['HBP']))
        player_hitting_df_agg['H_BB%'] = player_hitting_df_agg['BB'] / player_hitting_df_agg['AB']
        player_hitting_df_agg['H_K/BB'] = player_hitting_df_agg['K'] / player_hitting_df_agg['BB']

        # 球員投手         
        player_pitching_df_agg = player_pitching_df.groupby(['球隊', '年度']).agg(
            WHIP=('每局被上壘率', 'mean'),
            HBP=('死球', 'sum'),
        ).reset_index()

        # 球員守備
        player_defense_df_agg = player_defense_df.groupby(['球隊', '年度']).agg(
            SB =('被盜成功', 'sum'),
            CS=('盜壘阻殺', 'sum'),
        ).reset_index()
        player_defense_df_agg['CS%'] = player_defense_df_agg['CS'] /(player_defense_df_agg['SB'] + player_defense_df_agg['CS'])
        player_defense_df_agg = player_defense_df_agg.merge(team_pitching_df[['球隊', '年度', '局數']], on=['球隊', '年度'], how='left')
        player_defense_df_agg['SBA'] = (player_defense_df_agg['CS'] + (player_defense_df_agg['SB']) / player_defense_df_agg['局數'])

        # 球隊守備
        team_defense_df['DP/DO'] = team_defense_df['雙殺'] / team_defense_df['守備機會']
        team_defense_df = team_defense_df.merge(player_pitching_df_agg[['球隊', '年度', 'HBP']], on=['球隊', '年度'], how='left')
        team_defense_df = team_defense_df.merge(team_pitching_df[['球隊', '年度', '四壞球', '面對打席', '被全壘打', '被安打']], on=['球隊', '年度'], how='left')
        team_defense_df['DER'] = 1 - ((team_defense_df['被安打'] + team_defense_df['失誤'] - team_defense_df['被全壘打']) 
                                   / (team_defense_df['面對打席'] - team_defense_df['四壞球'] - team_defense_df['HBP'] - team_defense_df['被全壘打']))                    

        # === 合併資料 ===
        df = team_performance_df_agg[['球隊', '年度', '全年勝率']].rename(columns={'全年勝率': 'Win_Rate'}).copy()
        df = df.merge(team_pitching_df[['球隊', '年度', 'FIP', '防禦率', 'P_BABIP', 'P_BB%']].rename(columns={'防禦率': 'ERA'})
                 , on=['球隊', '年度'], how='left')
        df = df.merge(team_hitting_df[['球隊', '年度', 'OPS', '上壘率', 'ISO','H_K%']]
                 .rename(columns={'上壘率': 'OBP'}), on=['球隊', '年度'], how='left')
        df = df.merge(team_defense_df[['球隊', '年度', '守備率']], on=['球隊', '年度'],how='left').rename(columns={'守備率': 'Defense%'})
        df = df.merge(player_hitting_df_agg[['球隊', '年度', 'wOBA']], on=['球隊', '年度'], how='left')
        df = df.merge(player_pitching_df_agg[['球隊', '年度', 'WHIP']], on=['球隊', '年度'], how='left')
        
        df = df.drop_duplicates()
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(df)
        
        # 確保資料是按照年份排序
        self.df = df.sort_values(by=['年度'])

    def prepare_train_test_split(self):
        """處理訓練/測試集切分與標準化"""
        train_size = int(len(self.df) * 0.7)
        x_train = self.df.iloc[:train_size].drop(columns=['球隊', '年度', 'Win_Rate'])
        x_test = self.df.iloc[train_size:].drop(columns=['球隊', '年度', 'Win_Rate'])
        self.y_train = self.df.iloc[:train_size]['Win_Rate']
        self.y_test = self.df.iloc[train_size:]['Win_Rate']

        # 標準化
        x_train_scaled_array = self.scaler.fit_transform(x_train)
        x_test_scaled_array = self.scaler.transform(x_test)
        
        self.x_train_scaled = pd.DataFrame(x_train_scaled_array, columns=x_train.columns)
        self.x_test_scaled = pd.DataFrame(x_test_scaled_array, columns=x_test.columns)

    def build_and_tune_model(self):
        """定義模型並執行RandomizedSearchCV"""
        # 定義基學習器（第一層）
        base_learners = [
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('xgb', XGBRegressor(n_estimators=100, random_state=42)),
            ('elasticnet', ElasticNet(alpha=1.0, l1_ratio=0.5))
        ]

        # 定義元學習器（第二層）
        meta_learner = LinearRegression()

        # 建立集成模型（Stacking）
        stacking_model = StackingRegressor(estimators=base_learners, final_estimator=meta_learner)

        param_grid = {
            'rf__n_estimators': [100, 200, 300],
            'rf__max_depth': [3, 5, 10, 15],
            'rf__min_samples_split': [5, 10, 15],
            'rf__min_samples_leaf': [2, 3, 4],
            
            'xgb__n_estimators': [100, 200, 300],
            'xgb__learning_rate': [0.01, 0.05, 0.1],
            'xgb__max_depth': [3, 5, 10, 15],
            'xgb__colsample_bytree': [0.5, 0.6, 0.7, 0.8],
            'xgb__subsample': [0.6, 0.7, 0.8],
            'xgb__reg_alpha': [0, 0.1, 0.5, 1.0], 
            'xgb__reg_lambda': [0, 0.1, 0.5, 1.0], 
            
            'elasticnet__alpha': [0.01, 0.1, 1, 10, 100],
            'elasticnet__l1_ratio': [0.1, 0.3, 0.5, 0.7]
        }

        # RandomizedSearchCV 調參
        random_search = RandomizedSearchCV(stacking_model, param_distributions=param_grid, n_iter=50, 
                                           scoring='neg_mean_squared_error', cv=5, random_state=42, n_jobs=-1)
        random_search.fit(self.x_train_scaled, self.y_train)
        
        self.best_model = random_search.best_estimator_
        print("最佳參數組合：", random_search.best_params_)

    def evaluate_model(self):
        """評估模型表現 (交叉驗證及測試集表現)"""
        cv_scores = cross_val_score(self.best_model, self.x_train_scaled, self.y_train, cv=5, 
                                     scoring='neg_mean_squared_error')
        print("交叉驗證的均方誤差（Negative MSE）:", cv_scores)
        print("平均均方誤差:", cv_scores.mean())
        print("標準差:", cv_scores.std())

        # 訓練模型並預測
        self.best_model.fit(self.x_train_scaled, self.y_train)
        self.predictions = self.best_model.predict(self.x_test_scaled)

        # 測試集的指標
        mse = mean_squared_error(self.y_test, self.predictions)
        rmse = mean_squared_error(self.y_test, self.predictions, squared=False)
        mae = mean_absolute_error(self.y_test, self.predictions)
        r2 = r2_score(self.y_test, self.predictions)

        print(f"測試集 MSE: {mse}")
        print(f"測試集 RMSE: {rmse}")
        print(f"測試集 MAE: {mae}")
        print(f"測試集 R²: {r2}")

    def extract_feature_importances(self):
        """提取並印出基學習器的特徵重要性"""
        feature_importances = {}

        for name, model in self.best_model.named_estimators_.items():
            if hasattr(model, 'feature_importances_'):
                feature_importances[name] = pd.DataFrame({
                    '特徵': self.x_train_scaled.columns,
                    '重要性': model.feature_importances_
                }).sort_values(by='重要性', ascending=False)
            else:
                print(f"{name} 模型沒有特徵重要性屬性。")
                
        for name, importance_df in feature_importances.items():
            print(f"\n{name} 的特徵重要性排序:")
            print(importance_df)

    def visualize_results(self):
        """可視化實際與預測勝率對比"""
        comparison_df = pd.DataFrame({'Actual Win Rate': self.y_test, 'Predicted Win Rate': self.predictions})
        print("實際與預測勝率的對比:")
        print(comparison_df)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.y_test, y=self.predictions)
        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], color='red', linestyle='--')
        plt.xlabel("Actual Win Rate")
        plt.ylabel("Predicted Win Rate")
        plt.title("Actual vs Predicted Win Rates")
        plt.show()

    def _untested_prediction_block(self):
        """
        預測測試區塊，封裝在類別視情況啟用
        """
        '''    
        # 預測集
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
        # 特徵列表
        features = ['FIP', 'ERA', 'P_BABIP', 'P_BB%', 'OPS', 'OBP', 'ISO', 'H_K%', 'Defense%', 'wOBA', 'WHIP']
        predict_data = pd.DataFrame(predict_data, columns=features)

        # 標準化
        predict_data_scaled = self.scaler.transform(predict_data)

        # 預測 (此處的 dragon_model 對應目前的 self.best_model)
        predictions = self.best_model.predict(predict_data_scaled)

        # 顯示預測結果
        print("假如味全龍球員個人成績都與去年相當，那麼2025年勝率預計是")
        print(predictions)
        joblib.dump(self.best_model, 'dragon_model.pkl')
        joblib.dump(self.scaler, 'dragon_scaler.pkl')  # 儲存標準化器 
        plt.show()

          
        predict_data = {
            'FIP': [3.278216],
            'ERA': [4.436521739],
            'P_BABIP': [0.288024557],
            'P_BB%': [0.097196031],
            'OPS': [0.58925],
            'OBP': [0.28125],
            'ISO': [0.080928571],
            'H_K%': [0.236983443],
            'Defense%': [0.89575],
            'wOBA': [0.261990221],
            'WHIP': [1.453913043]
        }'''


# =====================================================================
# 執行與呼叫方式
# =====================================================================
if __name__ == "__main__":
    file_path = "CPBL數據 _面試用.xlsx"
    predictor = CPBLWinRatePredictor(file_path)
    predictor.execute_pipeline()
