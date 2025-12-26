import streamlit as st
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# 1. 載入模型與設定
@st.cache_resource # 快取機制，不用每次重新整理都重讀模型
def load_resources():
    model = joblib.load('tox_model.pkl')
    config = joblib.load('model_config.pkl')
    return model, config

model, config = load_resources()

# 2. 定義核心轉換函式 (這段跟你在 run_analysis 寫的一樣)
def smile_to_fingerprint(smile, n_bits):
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None: return None, None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        return np.array(fp).reshape(1, -1), mol # 回傳指紋和分子物件(畫圖用)
    except:
        return None, None

# --- 網頁介面設計 ---
st.title("💊 DeepTox: 藥物毒性預測系統")
st.markdown("輸入藥物化學式 (SMILES)，AI 將即時預測其 **NR-AR (雄激素受體)** 潛在毒性。")

# 左邊輸入，右邊顯示
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("請輸入 SMILES 化學結構:", "CC(=O)OC1=CC=CC=C1C(=O)O", height=100)
    
    # 加入一個「門檻調整」滑桿，讓這成為產品特色
    threshold = st.slider("判定門檻 (Risk Threshold)", 0.0, 1.0, 0.3, 0.05)
    st.caption("門檻越低，AI 越敏感 (寧可錯殺不放過)；門檻越高，AI 越保守。")

    if st.button("開始分析 🚀"):
        if not user_input:
            st.warning("請輸入化學式！")
        else:
            # 1. 轉換特徵
            X_input, mol = smile_to_fingerprint(user_input, config['n_bits'])
            
            if X_input is None:
                st.error("❌ 無法辨識此化學式，請檢查格式。")
            else:
                # 2. 模型預測
                # predict_proba 回傳 [[無毒機率, 有毒機率]]
                prob = model.predict_proba(X_input)[0][1] 
                
                # 3. 顯示結果
                st.divider()
                st.subheader("分析結果")
                
                # 動態顯示顏色
                if prob > threshold:
                    st.error(f"⚠️ **高風險 (TOXIC)**")
                    st.write(f"毒性機率: **{prob:.2%}** (超過設定門檻 {threshold})")
                else:
                    st.success(f"✅ **低風險 (SAFE)**")
                    st.write(f"毒性機率: **{prob:.2%}** (低於設定門檻 {threshold})")

with col2:
    st.write("### 分子結構預覽")
    if 'mol' in locals() and mol:
        # 畫出分子結構
        img = Draw.MolToImage(mol)
        st.image(img)
    else:
        st.info("輸入後顯示結構圖")

# 頁尾
st.divider()
st.caption("Model: Random Forest (Class Balanced + SMOTE) | Features: Morgan Fingerprints")