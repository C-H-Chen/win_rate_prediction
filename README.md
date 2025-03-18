# 簡介:

針對預測球隊年度勝率之問題，嘗試藉此掌握影響勝率之關鍵因子，洞察各項因子隨著數值變化，會如何影響勝率，達到設定球員篩選標準之目的。    

下圖為專案架構: 
![architecture](https://github.com/user-attachments/assets/fc9347be-2b0c-4e75-98ec-d470fd48f17b)  

# 數據清洗:  

發現"球員打擊(2010-24)"的銀棒指數第862行與"球員守備(2010-24)"的守備率欄位第2907行有缺失值，以填入實際正確數值做處理。  

發現"1998味全龍"多個球隊相關的數值出現異常值，但本研究決定以2010-24年的數據進行研究，故不影響。  

###### <h4> 基於10年代以前的數據有一些場外人為操控的因素，再加上成本考量，最終選擇以2010-24年的數據進行研究。

# 資料探索(EDA): 

此步驟主要使用三種技術: 

###### <h4>相關性分析 & 多重共線性檢測(VIF):

目的是找出各個特徵之間的相關性，以了解它們是如何相互影響，是否存在高度相關性。  

###### <h4>F檢定:  

目的是檢測特徵與目標變數之間是否有顯著關聯，識別哪些特徵對目標變數具有顯著的解釋能力，進而篩選出最具影響力的特徵。

###### <h5>此步驟之實際成果會放在後續的"成果分析"一併作說明。  

# 特徵工程:  

預測勝率之最終目的是預測一支球隊的表現，故在特徵的選擇上必須符合預測目的，避免直接把跟Y高度連動的變數當成X (得分、失分、WAR...)。  

基於以上原則進行特徵衍生&選擇，透過標準化處理特徵值的波動，並反覆評估資料探索與模型訓練的結果，最終選擇以下特徵:  

###### <h4>投球: 'FIP', 'ERA', 'BABIP', 'BB%', 'WHIP'  

###### <h4>打擊: 'OPS', 'OBP', 'ISO', 'K%', 'wOBA'  

###### <h4>守備: 'Defense%'  

共11個特徵。

# 模型建置:  

考量該題目的非線性關係與複雜模式，最終選擇能夠綜合多個基學習器優勢的"集成模型（Stacking）"，盼透過多樣性來提高預測的準確性與穩定性。

###### <h4>基學習器:  

###### <h5>RandomForest, XGBoost, ElasticNet。  

選擇這三個模型作為基學習器，主要是因為RandomForest和XGBoost能夠捕捉數據中的非線性模式和複雜關係，而ElasticNet則專注於處理線性關係與解決多重共線性問題，盼透過這樣的方式在處理複雜與高維度的數據時，能夠提供穩定、精確且具有較強泛化能力的預測結果。

###### <h4>元學習器:  

###### <h5>LinearRegression。

考量計算效率、過擬合風險與穩定性等因素，最終選擇LinearRegression作為元學習器。

###### <h4>參數調整:  

考量運算成本，最終選擇RandomizedSearchCV技術進行參數調整。

###### <h4>交叉驗證:  

透過此方式評估模型的表現。下圖皆為模型評估的結果。  

![結果1](https://github.com/user-attachments/assets/bffb6612-024d-45d5-ab8f-25d030c7ea4b)  

![實際vs預測勝率](https://github.com/user-attachments/assets/8a34a003-48a0-4327-9385-f3b31800b100)  

![結果2](https://github.com/user-attachments/assets/b53d05b5-57be-4e18-8772-18ec08e45804)

均方誤差（Negative MSE）: 皆為負數，表示模型的預測誤差小於預期。平均均方誤差為 -0.0011，標準差為 0.00066，表示模型在不同訓練集的表現穩定，沒有過度擬合。  

測試集MSE: 0.001。說明模型的預測誤差非常小，大多數預測值與實際值之間的差距都在可接受範圍內，表示模型具有良好的預測準確度。  

測試集RMSE: 0.0318。說明模型的預測與實際值之間的誤差並不大，預測相對準確。  

測試集MAE: 0.0245。說明預測的平均絕對誤差並不大，大多數情況下能夠較為準確地預測目標變數。

R²: 0.7051。說明能解釋 70.51% 的變異，表明模型有相對較好的預測能力。但仍有約 30% 的變異無法被模型解釋，表示可能還有些其他因素或特徵未被納入模型。

# 成果分析:  
###### <h4>邊際效應分析:  

目的是為了量化每個特徵對勝率預測的邊際影響，即當特徵的值改變時，勝率的預測值如何變化，從而實現球員篩選標準的設定。  

下圖是未發現顯著變化的特徵，依序為'FIP', 'ERA', 'BABIP', 'BB%', 'WHIP', 'OBP', 'K%', 'Defense%' :  

![FIP](https://github.com/user-attachments/assets/716c0370-35b0-474c-bbc4-8b61918533c3)  

![ERA](https://github.com/user-attachments/assets/a5cc8c93-b0bb-4d4d-91be-053cd0eb6a75)  

![BABIP](https://github.com/user-attachments/assets/4ab9067d-a565-4274-bc7b-15c7bce0467f)  

![BB%](https://github.com/user-attachments/assets/5ad9d839-44e8-462e-b038-1ff9377a6b12)  

![WHIP](https://github.com/user-attachments/assets/b31c4e20-870c-413b-8e23-f97f44015eb7)  

![OBP](https://github.com/user-attachments/assets/45b340c0-dc8e-4145-8c68-7af69a65e8e9)  

![K%](https://github.com/user-attachments/assets/9bb78c04-9cae-47ea-afdc-e3be1b2c71f3)  

![DEFENSE%](https://github.com/user-attachments/assets/2c805c10-f291-403c-99bf-54a284ecb8d7)  

下圖是發現有顯著變化的特徵，依序為'OPS', 'ISO', 'wOBA' :  

![OPS](https://github.com/user-attachments/assets/8c76624c-52b0-4ef2-8030-7b17ccbe56fe)  

![ISO](https://github.com/user-attachments/assets/ebc31039-6411-4828-be94-8b30027810d0)  

![wOBA](https://github.com/user-attachments/assets/9e6497ab-b9aa-4777-a5c3-9d6a130735d5)  


