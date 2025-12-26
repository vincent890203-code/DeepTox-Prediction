# 💊 DeepTox: AI-Powered Drug Toxicity Prediction System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![RDKit](https://img.shields.io/badge/Chemoinformatics-RDKit-green)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Status](https://img.shields.io/badge/Status-MVP%20Completed-success)

## 📖 專案簡介 (Project Overview)
**DeepTox** 是一個端對端 (End-to-End) 的機器學習系統，旨在解決藥物研發早期的毒性篩選痛點。本專案整合了化學資訊學 (Chemoinformatics) 與 AI 演算法，針對 **Tox21 資料集** 中的 **NR-AR (雄激素受體)** 活性進行預測，協助研發人員在實驗室測試前識別高風險分子。

### 核心問題解決
* **特徵轉譯**：將化學式文字 (SMILES) 轉化為機器可讀的 **Morgan Fingerprints**。
* **資料不平衡**：針對極端不平衡數據 (1:25)，導入 **SMOTE** 與 **Class Weighting** 技術。
* **決策輔助**：開發互動式 Web App，提供動態 **閾值調整 (Threshold Moving)** 功能，平衡 Recall 與 Precision。

---

## 🚀 功能特色 (Key Features)

* **🧪 智慧特徵工程**：自動化學結構解析，生成 2048-bit ECFP4 分子指紋。
* **⚖️ 平衡訓練機制**：內建 SMOTE 演算法，合成少數類別樣本，大幅提升模型對有毒分子的敏感度。
* **📊 互動式儀表板**：基於 Streamlit 的視覺化介面，支援即時 SMILES 輸入與分子結構繪圖 (2D Visualization)。
* **🎚️ 動態風險評估**：使用者可自定義風險門檻 (Risk Threshold)，實現「寧可錯殺，不可放過」的篩選策略。

---

## 🛠️ 技術棧 (Tech Stack)

| 領域 | 技術/套件 | 用途 |
| :--- | :--- | :--- |
| **Language** | Python 3.x | 核心開發語言 |
| **Chemoinformatics** | **RDKit** | 分子物件生成、Morgan Fingerprint 計算、結構繪圖 |
| **Machine Learning** | **Scikit-learn** | Random Forest 模型訓練、評估 metrics |
| **Data Handling** | **Pandas, NumPy** | 數據清洗、矩陣運算 |
| **Imbalanced Data** | **Imbalanced-learn** | SMOTE 數據增強 (Synthetic Minority Over-sampling) |
| **Web App** | **Streamlit** | 前端介面開發、模型部署 |
| **Version Control** | Git / GitHub | 版本控制與協作 |

---

## 📂 專案結構 (Directory Structure)

```text
Bio-Project/
├── app.py                  # 🚀 產品入口：Streamlit 網頁主程式
├── run_analysis_3.py       # ⚙️ 訓練管線：負責數據清洗、特徵工程、模型訓練與存檔
├── my_practice.py          # 🧰 核心模組：封裝 BioMLTrainer 類別 (OOP 設計)
├── tox21.csv               # 📄 原始數據：tox21 Dataset (from)
├── tox_model.pkl           # 🧠 訓練好的模型 (Binary File)
├── model_config.pkl        # ⚙️ 模型設定檔 (Threshold, n_bits)
└── README.md               # 📖 專案說明文件

```

---

### ⚡ 快速開始 (Quick Start)
1. 安裝依賴套件
請確保已安裝 Python 環境，並執行以下指令安裝必要套件：

```Bash

pip install pandas numpy scikit-learn rdkit streamlit imbalanced-learn joblib

```
2. 訓練模型 (Model Training)
執行訓練腳本，這將會進行數據清洗、SMOTE 增強、訓練隨機森林，並產出 .pkl 模型檔。

```Bash

python run_analysis_3.py

```
預期輸出：您將看到終端機顯示準確率 (Accuracy) 與分類報告，並提示模型已儲存。

3. 啟動網頁應用 (Launch Web App)
啟動 Streamlit 伺服器，開啟瀏覽器介面。

```Bash

streamlit run app.py

```

---

### 📊 模型效能 (Performance)
整體準確率 (Accuracy): ~97% 
(Baseline)

優化策略: 由於原始數據中陽性樣本僅佔 4%，單看準確率容易產生誤導。本專案透過 閾值移動 (Threshold Moving) 分析，發現在門檻降至 0.3 時，能有效將 Recall 提升至 60%~80% 區間，滿足藥物篩選的高敏感度需求。

### 📝 關於作者 (Author)
[Yuan Chen Kuo/Vincent]

Bio-AI Developer | Full-Stack Algorithmic Engineer

專注於結合生物醫學領域知識與現代 AI 技術，解決複雜的生醫數據問題。