import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from bioml_trainer import BioMLTrainer
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # 關閉所有 RDKit 的 C++ 層級警告
# import warnings 
# warnings.filterwarnings("ignore") 只能攔截python warning

# 1. 讀取數據   
file_path = 'tox21.csv'  # 確保檔名正確
print(f"正在讀取數據：{file_path} ...")
df = pd.read_csv(file_path)

# --- 新增：資料前處理模組 ---

def smile_to_fingerprint(smile, n_bits=2048):
    """
    將 SMILES 化學式轉換為 Morgan Fingerprint (數位指紋)
    :param smile: 化學式字串 (e.g., 'CCO')
    :param n_bits: 指紋長度 (通常用 2048)
    :return: Numpy 陣列 (一串 0 和 1)
    """
    try:
        # 1. 將文字轉成 RDKit 分子物件
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None # 如果化學式有錯，回傳空值
            
        # 2. 計算 Morgan Fingerprint (半徑=2, 類似 ECFP4)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        
        # 3. 轉成 Numpy 陣列讓 AI 看得懂
        return np.array(fp)
    except:
        return None

# --- 改變測試代碼 ---

print("🧪 正在進行全量特徵工程 (這會花一點時間)...")

# 1. 直接對「整張表」進行轉換，而不是只取 head(3)
# 我們新增一個欄位 'fingerprint' 來暫存，這樣數據跟標籤才會對應
df['fingerprint'] = df['smiles'].apply(smile_to_fingerprint)

# 2. 移除轉換失敗的資料
# 有些化學式可能格式錯誤導致產生 None，必須移除，否則無法堆疊
df_clean = df.dropna(subset=['fingerprint']).copy()

print(f"✅ 指紋轉換完成！有效資料：{len(df_clean)} 筆 (移除 {len(df) - len(df_clean)} 筆無效資料)")

# 3. 準備訓練數據
print("\n🚀 正在準備訓練矩陣...")

# X: 把指紋欄位堆疊成矩陣
# 注意：這裡要用 df_clean，長度才會對
X_data = np.stack(df_clean['fingerprint'].values)

# y: 拿出對應的標籤
target_col = 'NR-AR'
y_data = df_clean[target_col].values

# 4. 最後清洗：移除 Label 是 NaN (空值) 的數據
# 我們把 X 和 y 暫時綁在一起洗，確保對應關係不會亂掉
# 這裡使用一個技巧：建立暫存 DataFrame
model_df = pd.DataFrame(X_data)
model_df['Label'] = y_data

# 移除標籤是空值的列
model_df = model_df.dropna(subset=['Label'])

print(f"🧹 最終清洗完成！剩餘 {len(model_df)} 筆可訓練數據")

# 5. 分離 X 和 y 餵給模型
X_final = model_df.iloc[:, :-1].values
y_final = model_df['Label'].values

print(f"📦 最終訓練集維度：X={X_final.shape}, y={y_final.shape}")

# 6. 呼叫你的機器進行訓練
print("\n🔥 啟動 BioMLTrainer...")
trainer = BioMLTrainer(X_final, y_final)
trainer.split_data()
trainer.train_model()
trainer.evaluate()

# --- 請接在原本的程式碼最後面 ---
import joblib

print("\n💾 正在儲存模型...")
# 1. 儲存訓練好的模型
joblib.dump(trainer.model, 'tox_model.pkl')
print("✅ 模型已儲存為 'tox_model.pkl'")

# 2. 我們也需要知道指紋的長度，之後 App 轉換時才不會錯
# (雖然我們知道是 2048，但寫下來比較保險)
config = {'n_bits': 2048, 'threshold': 0.3} # 我們選定 0.3 作為產品的預設門檻
joblib.dump(config, 'model_config.pkl')
print("✅ 設定檔已儲存為 'model_config.pkl'")