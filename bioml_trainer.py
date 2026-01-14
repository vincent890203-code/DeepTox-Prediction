# 匯入工具箱
import pandas as pd # pandas 讀表格
from sklearn.model_selection import train_test_split # sklearn機器學習 # model_selection - train_test_split 資料分割
from sklearn.ensemble import RandomForestClassifier # ensemble - RandomForestClassifier 隨機森林載入
from sklearn.metrics import accuracy_score, classification_report # metrics - accuracy_score, classification_report 分析結果
from sklearn.datasets import load_breast_cancer # datasets - load_breast_cancer 載入乳癌資料庫
from imblearn.over_sampling import SMOTE


# 定義類別(Class)如設計一張實驗流程
class BioMLTrainer:
    '''

    初始化訓練環境
    :param X: 特徵數據 (Features)
    :param y: 目標標籤 (Labels, e.g., 0=良性, 1=惡性)

    '''
    
    def __init__(self, X, y, test_size=0.2): # __init__為初始化，self代表這個物件自己
        self.X = X  #把外部X存進來
        self.y = y  #把外部y存進來
        self.test_size = test_size # 規定測試data量
        self.model = None # 準備好空模組 for 接下來的RandomForest
        self.X_train, self.X_test, self.y_train, self.y_test, = (None,None,None,None) # 準備好空變數，之後資料切割會用到

    
    def split_data(self):
        """
        [模組 1] 資料分割與增強 (SMOTE)
        """
        print("⚖️ 正在處理資料不平衡 (SMOTE)...")
        
        # 1. 先切分訓練集與測試集 (這步不能變，一定要先切再增強，不然會作弊)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=42
        )
        
        # 2. 檢查訓練集中是否有少數類別
        # 如果全都是 0，SMOTE 會報錯，所以要加個檢查
        if sum(self.y_train) > 0:
            smote = SMOTE(random_state=42)
            # 只對「訓練集」進行增強，千萬不要動「測試集」
            X_resampled, y_resampled = smote.fit_resample(self.X_train, self.y_train)
            
            print(f"📈 SMOTE 增強前：{self.X_train.shape[0]} 筆 (有毒樣本: {sum(self.y_train)})")
            print(f"📊 SMOTE 增強後：{X_resampled.shape[0]} 筆 (有毒樣本: {sum(y_resampled)})")
            
            # 把增強後的數據塞回去
            self.X_train = X_resampled
            self.y_train = y_resampled
        else:
            print("⚠️ 警告：訓練集中沒有陽性樣本，跳過 SMOTE。")
            
        print("✅ 資料準備完成！")
    
    def train_model(self):
        """
        [模組 2] 模型訓練 (使用隨機森林)
        這個module未來可以替換成不同的演算法，如 XGBoost 以及 Pytorch

        """
        print("🚀 開始訓練模型 (Random Forest)...")
        # 初始化模型 (Model Initialization)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # n_estimators指定有多少顆決策樹，100是常見預設值。
        # 訓練模型 (Model Training)
        self.model.fit(self.X_train, self.y_train)
        ## sklearning中通用方法，讓模型根據提供的數據進行學習。
        print("✅ 模型訓練完畢")

    def evaluate(self):
        """
        [模組 3] 效能評估 (含閾值調整分析)
        """
        if self.model is None:
            print("❌ 錯誤：請先執行 train_model()")
            return

        print("📊 正在評估模型效能...")
        
        # 1. 取得「機率值」而不是直接的 0/1 預測
        # predict_proba 會回傳兩個數字：[是0的機率, 是1的機率]
        # 我們只關心「是1(有毒)的機率」，所以取第二個欄位 [:, 1]
        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        # 2. 測試不同的門檻 (Threshold)
        print("\n🔍 閾值敏感度分析 (Threshold Analysis):")
        print(f"{'Threshold':<10} {'Recall (抓到多少毒)':<20} {'Precision (抓得準不準)':<20}")
        print("-" * 60)

        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            # 如果機率 > threshold 就判斷為 1，否則為 0
            y_pred_adjusted = (y_prob >= threshold).astype(int)
            
            # 手動計算 Recall 和 Precision
            # 這裡我們只關心 Class 1 (有毒) 的表現
            from sklearn.metrics import recall_score, precision_score
            rec = recall_score(self.y_test, y_pred_adjusted)
            prec = precision_score(self.y_test, y_pred_adjusted)
            
            print(f"{threshold:<10} {rec:.4f}{' (🔥)' if rec > 0.6 else ''}           {prec:.4f}")
            
        print("-" * 60)
        print("💡 結論：通常我們會選 Recall > 0.6 且 Precision 不要太爛的門檻。")

# --- 模擬實戰區 (Main)/指揮中心 --- 
if __name__ == "__main__": # 這是 Python 的標準寫法。意思是：「如果我直接執行這個檔案，請從這裡開始跑。」
# 載入data (生醫領域的Hello world數據) 
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    print(f"📥 載入數據：{data.filename}")

    # 2. 實例化系統(Instantiation): 把機器造出來
    my_trainer = BioMLTrainer(X,y) 
    # BioMLTrainer 是你的設計圖（Class）。
    # my_trainer 是你根據設計圖，實際造出來的那台機器（Instance/Object）。

    # 3. 執行流水線
    my_trainer.split_data()
    my_trainer.train_model()
    my_trainer.evaluate()