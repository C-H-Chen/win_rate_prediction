import pingouin as pg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import f_regression
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

class DataLoader:
    """讀取 Excel 檔案與各個工作表"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def load_data(self) -> dict:
        xls = pd.ExcelFile(self.file_path)
        return {
            'team_performance': pd.read_excel(xls, sheet_name='球隊成績'),
            'team_pitching': pd.read_excel(xls, sheet_name='球隊投手'),
            'team_hitting': pd.read_excel(xls, sheet_name='球隊打擊'),
            'team_defense': pd.read_excel(xls, sheet_name='球隊守備'),
            'player_hitting': pd.read_excel(xls, sheet_name='球員打擊(2010-24)'),
            'player_pitching': pd.read_excel(xls, sheet_name='球員投手(2010-24)'),
            'player_defense': pd.read_excel(xls, sheet_name='球員守備(2010-24)')
        }


class DataPreprocessor:
    """資料清理、特徵工程與資料合併"""
    def __init__(self, data_dict: dict):
        self.team_performance_df = data_dict['team_performance']
        self.team_pitching_df = data_dict['team_pitching']
        self.team_hitting_df = data_dict['team_hitting']
        self.team_defense_df = data_dict['team_defense']
        self.player_hitting_df = data_dict['player_hitting']
        self.player_pitching_df = data_dict['player_pitching']
        self.player_defense_df = data_dict['player_defense']

    def process_and_merge(self) -> pd.DataFrame:
        self._clean_and_filter_years()
        self._calculate_team_metrics()
        self._calculate_player_metrics()
        return self._merge_dataframes()

    def _clean_and_filter_years(self):
        # 處理年度格式，判斷是否為上下年度
        self.team_performance_df['年度'] = self.team_performance_df['年度'].apply(
            lambda x: str(x).split('(')[0].strip() if '(' in str(x) else str(x)
        )

        # 篩選年份範圍
        self.team_performance_df = self.team_performance_df[self.team_performance_df['年度'].astype(int).between(2010, 2024)]
        self.team_pitching_df = self.team_pitching_df[self.team_pitching_df['年度'].astype(int).between(2010, 2024)]
        self.team_defense_df = self.team_defense_df[self.team_defense_df['年度'].astype(int).between(2010, 2024)]

        # 轉換年度為字串
        self.team_performance_df['年度'] = self.team_performance_df['年度'].astype(str)
        self.team_pitching_df['年度'] = self.team_pitching_df['年度'].astype(str)
        self.team_hitting_df['年度'] = self.team_hitting_df['年度'].astype(str)
        self.team_defense_df['年度'] = self.team_defense_df['年度'].astype(str)
        self.player_pitching_df['年度'] = self.player_pitching_df['年度'].astype(str)
        self.player_hitting_df['年度'] = self.player_hitting_df['年度'].astype(str)
        self.player_defense_df['年度'] = self.player_defense_df['年度'].astype(str)

    def _calculate_team_metrics(self):
        # 計算全年
        self.team_performance_df_agg = self.team_performance_df.groupby(['球隊', '年度']).agg(
            全年勝率=('勝率', lambda x: x.sum() / 2),
            全年主場勝=('主場勝', 'sum'),
            全年主場敗=('主場敗', 'sum'),
            全年客場勝=('客場勝', 'sum'),
            全年客場敗=('客場敗', 'sum')
        ).reset_index()

        avg_era = self.team_pitching_df.groupby('年度')['防禦率'].mean()
        avg_fip = self.team_pitching_df.groupby('年度').apply(
            lambda x: ((13 * x['被全壘打'] + 3 * x['四壞球'] - 2 * x['奪三振']) / x['局數'] + 3.10).mean()
        )
        avg_ops = self.team_hitting_df.groupby('年度').apply(
            lambda x: (x['上壘率'] + x['長打率']).mean()
        )

        # 球隊成績
        self.team_performance_df_agg['Home-Away_diff'] = (
            (self.team_performance_df_agg['全年主場勝'] - self.team_performance_df_agg['全年主場敗']) - 
            (self.team_performance_df_agg['全年客場勝'] - self.team_performance_df_agg['全年客場敗'])
        )

        # 球隊投手
        self.team_pitching_df['FIP'] = ((13 * self.team_pitching_df['被全壘打'] + 3 * self.team_pitching_df['四壞球'] - 2 * self.team_pitching_df['奪三振']) / self.team_pitching_df['局數'] + 3.10)
        self.team_pitching_df['P_BABIP'] = (self.team_pitching_df['被安打'] - self.team_pitching_df['被全壘打']) / (self.team_pitching_df['面對打席'] - self.team_pitching_df['被全壘打'] - self.team_pitching_df['奪三振'] - self.team_pitching_df['四壞球'])
        self.team_pitching_df['ERA+'] = 100 * (self.team_pitching_df['年度'].map(avg_era) / self.team_pitching_df['防禦率'])
        self.team_pitching_df['K%'] = self.team_pitching_df['奪三振'] / self.team_pitching_df['面對打席']
        self.team_pitching_df['P_BB%'] = self.team_pitching_df['四壞球'] / self.team_pitching_df['面對打席']
        self.team_pitching_df['K/BB'] = (self.team_pitching_df['奪三振'] / self.team_pitching_df['四壞球'])
        self.team_pitching_df['HR%'] = self.team_pitching_df['被全壘打'] / self.team_pitching_df['面對打席']

        # 球隊打者
        self.team_hitting_df['OPS'] = (self.team_hitting_df['上壘率'] + self.team_hitting_df['長打率'])
        self.team_hitting_df['H_K%'] = self.team_hitting_df['三振'] / self.team_hitting_df['打數']
        self.team_hitting_df['SH/AB'] = self.team_hitting_df['犧牲短打'] / self.team_hitting_df['打數']
        self.team_hitting_df['SB/G'] = self.team_hitting_df['盜壘成功'] / self.team_hitting_df['出賽數']
        self.team_hitting_df['ISO'] = self.team_hitting_df['長打率'] - self.team_hitting_df['打擊率']
        self.team_hitting_df['PPG'] = self.team_hitting_df['得分'] - self.team_hitting_df['出賽數']

    def _calculate_player_metrics(self):
        # 球員打擊
        self.player_hitting_df_agg = self.player_hitting_df.groupby(['球隊', '年度']).agg(
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
        
        self.player_hitting_df_agg['SH%'] = (self.player_hitting_df_agg['SH'] + self.player_hitting_df_agg['SF']) / self.player_hitting_df_agg['PA']
        self.player_hitting_df_agg['wOBA'] = ((0.69 * self.player_hitting_df_agg['BB'] + 0.72 * self.player_hitting_df_agg['HBP'] 
                               + 0.9 * self.player_hitting_df_agg['one'] + 1.25 * self.player_hitting_df_agg['double']
                               + 1.6 * self.player_hitting_df_agg['triple'] + 1.8 * self.player_hitting_df_agg['HR']) 
                               / (self.player_hitting_df_agg['AB'] + self.player_hitting_df_agg['BB'] + self.player_hitting_df_agg['SF'] + self.player_hitting_df_agg['HBP']))
        self.player_hitting_df_agg['H_BB%'] = self.player_hitting_df_agg['BB'] / self.player_hitting_df_agg['AB']
        self.player_hitting_df_agg['H_K/BB'] = self.player_hitting_df_agg['K'] / self.player_hitting_df_agg['BB']

        # 球員投手           
        self.player_pitching_df_agg = self.player_pitching_df.groupby(['球隊', '年度']).agg(
            WHIP=('每局被上壘率', 'mean'),
            HBP=('死球', 'sum'),
        ).reset_index()

        # 球員守備
        self.player_defense_df_agg = self.player_defense_df.groupby(['球隊', '年度']).agg(
            SB =('被盜成功', 'sum'),
            CS=('盜壘阻殺', 'sum'),
        ).reset_index()
        self.player_defense_df_agg['CS%'] = self.player_defense_df_agg['CS'] / (self.player_defense_df_agg['SB'] + self.player_defense_df_agg['CS'])
        self.player_defense_df_agg = self.player_defense_df_agg.merge(self.team_pitching_df[['球隊', '年度', '局數']], on=['球隊', '年度'], how='left')
        self.player_defense_df_agg['SBA'] = (self.player_defense_df_agg['CS'] + (self.player_defense_df_agg['SB']) / self.player_defense_df_agg['局數'])

        # 球隊守備
        self.team_defense_df['DP/DO'] = self.team_defense_df['雙殺'] / self.team_defense_df['守備機會']
        self.team_defense_df = self.team_defense_df.merge(self.player_pitching_df_agg[['球隊', '年度', 'HBP']], on=['球隊', '年度'], how='left')
        self.team_defense_df = self.team_defense_df.merge(self.team_pitching_df[['球隊', '年度', '四壞球', '面對打席', '被全壘打', '被安打']], on=['球隊', '年度'], how='left')
        self.team_defense_df['DER'] = 1 - ((self.team_defense_df['被安打'] + self.team_defense_df['失誤'] - self.team_defense_df['被全壘打']) 
                                   / (self.team_defense_df['面對打席'] - self.team_defense_df['四壞球'] - self.team_defense_df['HBP'] - self.team_defense_df['被全壘打']))

    def _merge_dataframes(self) -> pd.DataFrame:
        df = self.team_performance_df_agg[['球隊', '年度', '全年勝率']].rename(columns={'全年勝率': 'Win_Rate'}).copy()
        df = df.merge(self.team_pitching_df[['球隊', '年度', 'FIP', '防禦率', 'P_BABIP', 'P_BB%']].rename(columns={'防禦率': 'ERA'}), on=['球隊', '年度'], how='left')
        df = df.merge(self.team_hitting_df[['球隊', '年度', 'OPS', '上壘率', 'ISO','H_K%']].rename(columns={'上壘率': 'OBP'}), on=['球隊', '年度'], how='left')
        df = df.merge(self.team_defense_df[['球隊', '年度', '守備率']], on=['球隊', '年度'], how='left').rename(columns={'守備率': 'Defense%'})
        df = df.merge(self.player_hitting_df_agg[['球隊', '年度', 'wOBA']], on=['球隊', '年度'], how='left')
        df = df.merge(self.player_pitching_df_agg[['球隊', '年度', 'WHIP']], on=['球隊', '年度'], how='left')
        
        df = df.drop_duplicates()
        
        # 資料按照年份排序
        df = df.sort_values(by=['年度'])
        
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(df)
        
        return df


class DataAnalyzer:
    """資料切割、標準化、統計分析與視覺化繪圖"""
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.x_train_scaled = None
        self.x_test_scaled = None
        self.all_columns = None

    def prepare_data(self):
        # 確定訓練集和測試集的分割點
        train_size = int(len(self.df) * 0.7)
        self.x_train = self.df.iloc[:train_size].drop(columns=['球隊', '年度', 'Win_Rate'])
        self.x_test = self.df.iloc[train_size:].drop(columns=['球隊', '年度', 'Win_Rate'])
        self.y_train = self.df.iloc[:train_size]['Win_Rate']
        self.y_test = self.df.iloc[train_size:]['Win_Rate']

        # 標準化
        scaler = StandardScaler()
        x_train_scaled_arr = scaler.fit_transform(self.x_train)
        x_test_scaled_arr = scaler.transform(self.x_test)
        
        self.x_train_scaled = pd.DataFrame(x_train_scaled_arr, columns=self.x_train.columns)
        self.x_test_scaled = pd.DataFrame(x_test_scaled_arr, columns=self.x_test.columns)
        self.all_columns = self.x_train_scaled.select_dtypes(include=[float, int]).columns.tolist()

    def plot_correlation(self):
        # 特徵相關性
        correlation_matrix = self.x_train_scaled.select_dtypes(include=[float, int]).corr()
        print(correlation_matrix)
        plt.tight_layout()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

    def calculate_partial_correlation(self):
        y_col = self.y_train 
        y_col_name = y_col.name if isinstance(y_col, pd.Series) else y_col

        # 計算每個 x 與固定的 y 之間的偏相關
        results = []
        for feature in self.all_columns:
            pcorr_result = pg.partial_corr(data=self.x_train_scaled.join(y_col), x=feature, y=y_col_name)
            results.append(pcorr_result)
            print(f"偏相關分析: 特徵 {feature} 與 {y_col_name} 的結果: \n{pcorr_result}\n")

        results_df = pd.concat(results, ignore_index=True)
        print(results_df)

    def analyze_f_scores(self):
        # F score
        f_scores, p_values = f_regression(self.x_train, self.y_train)

        # 顯示F分數和P值
        for name, f_score, p_value in zip(self.all_columns, f_scores, p_values):
            print(f"{name}: F-Score = {f_score:.4f}, P-value = {p_value:.4f}")
            
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # F分數條形圖
        ax[0].bar(self.all_columns, f_scores, color='skyblue')
        ax[0].set_title('F-Score for Each Feature')
        ax[0].set_xlabel('Features')
        ax[0].set_ylabel('F-Score')
        
        # P值條形圖
        ax[1].bar(self.all_columns, p_values, color='lightcoral')
        ax[1].set_title('P-Value for Each Feature')
        ax[1].set_xlabel('Features')
        ax[1].set_ylabel('P-Value')

        plt.tight_layout()

    def analyze_linear_regression(self):
        # 斜率
        lin_reg = LinearRegression()
        lin_reg.fit(self.x_train, self.y_train)

        # 輸出特徵係數
        slopes = lin_reg.coef_
        features = self.x_train.columns  
        for name, slope in zip(features, slopes):
            print(f"{name}: {slope:.4f}")
            
        # 視覺化
        plt.figure()
        plt.bar(features, slopes, color='skyblue')
        plt.xlabel('Features')
        plt.ylabel('Coefficient (Slope)')
        plt.title('Linear Regression Coefficients (Slopes)')

    def test_hypothesis_elastic_net(self):
        # 假設檢定
        # Elastic Net Logistic Regression
        elastic_net = ElasticNetCV(cv=5, l1_ratio=0.5, max_iter=10000)
        elastic_net.fit(self.x_train_scaled, self.y_train)

        # 使用statsmodels進行假設檢定
        x_train_const = sm.add_constant(self.x_train_scaled)  # 添加截距
        
        # 確保索引一致
        y_train_reset = self.y_train.reset_index(drop=True)
        x_train_const_reset = x_train_const.reset_index(drop=True)

        logit_model = sm.Logit(y_train_reset, x_train_const_reset)
        result = logit_model.fit()

        # OLS 檢定結果摘要
        print(result.summary())

        # 提取回歸係數、P 值、置信區間
        coef = result.params[1:]  # 排除截距項
        p_values = result.pvalues[1:]
        conf = result.conf_int().iloc[1:]  # 排除截距項的置信區間
        conf.columns = ['Lower CI', 'Upper CI']

        # 創建DataFrame存儲特徵、係數、P 值、置信區間
        df_coef = pd.DataFrame({
            'Feature': self.x_train.columns,
            'Coefficient': coef.values,
            'P-Value': p_values.values,
            'Lower CI': conf['Lower CI'].values,
            'Upper CI': conf['Upper CI'].values
        })

        # 回歸係數與置信區間圖表
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")

        # 條形圖顯示回歸係數
        sns.barplot(x='Coefficient', y='Feature', data=df_coef, palette='coolwarm', hue='Coefficient', legend=False)

        # 橫向置信區間
        for i in range(df_coef.shape[0]):
            plt.plot([df_coef['Lower CI'].iloc[i], df_coef['Upper CI'].iloc[i]], [i, i], color='black')
            
        # 標記顯著性 (P 值 < 0.05)
        for i in range(len(df_coef)):
            if df_coef['P-Value'].iloc[i] < 0.05:
                plt.text(df_coef['Coefficient'].iloc[i], i, '*', color='red', va='center')

        # 標題與軸標籤
        plt.title('Regression Hypothesis Tests for Elastic Net Logistic Regression Model')
        plt.xlabel('Coefficient Value')
        plt.ylabel('Feature')
        plt.axvline(0, color='black', linewidth=0.8)  # 中心線

        plt.tight_layout()

    def check_multicollinearity(self):
        # 檢測多重共線性
        vif_data = pd.DataFrame()
        vif_data["feature"] = self.x_train.columns
        vif_data["VIF"] = [variance_inflation_factor(self.x_train_scaled.values, i) for i in range(self.x_train_scaled.shape[1])]
        print(vif_data)

    def run_all_analysis(self):
        """依序執行所有分析與繪圖"""
        self.plot_correlation()
        self.calculate_partial_correlation()
        self.analyze_f_scores()
        self.analyze_linear_regression()
        self.test_hypothesis_elastic_net()
        self.check_multicollinearity()
        plt.show()


class CPBLAnalysisPipeline:
    """統整運作流程"""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def run(self):
        # 讀取資料
        loader = DataLoader(self.file_path)
        data_dict = loader.load_data()

        # 資料預處理與合併
        preprocessor = DataPreprocessor(data_dict)
        final_df = preprocessor.process_and_merge()

        # 模型分析與視覺化
        analyzer = DataAnalyzer(final_df)
        analyzer.prepare_data()
        analyzer.run_all_analysis()

# ==========================================
# 程式執行進入點
# ==========================================
if __name__ == "__main__":
    FILE_PATH = "CPBL數據 _面試用.xlsx"
    pipeline = CPBLAnalysisPipeline(FILE_PATH)
    pipeline.run()
