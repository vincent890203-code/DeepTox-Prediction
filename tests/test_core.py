# tests/test_core.py
import pytest
import numpy as np
import pandas as pd
# 注意：這裡假設你已經把 my_practice.py 改名為 bioml_trainer.py
from bioml_trainer import BioMLTrainer 

# 1. 製作假資料 (Mock Data)
@pytest.fixture
def dummy_data():
    n_samples = 100
    n_features = 10
    
    X = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f'feat_{i}' for i in range(n_features)])
    
    # 建立 80 個無毒(0) 和 20 個有毒(1)
    # 這樣切分後，訓練集大約會有 16 個有毒樣本，遠大於 SMOTE 需要的 5 個，就不會報錯了
    y = np.array([0]*80 + [1]*20)
    # --- 修改點結束 ---
    return X, y

# 2. 測試：類別能不能被初始化
def test_initialization(dummy_data):
    X, y = dummy_data
    trainer = BioMLTrainer(X, y)
    assert trainer.X is not None
    assert trainer.y is not None

# 3. 測試：不平衡處理 (SMOTE)
def test_smote_pipeline(dummy_data):
    X, y = dummy_data
    trainer = BioMLTrainer(X, y)
    
    # 執行分割與增強
    trainer.split_data()
    
    # 驗證：增強後的訓練集，陽性樣本(1)的數量應該變多了
    # 原始只有少數幾個 1，SMOTE 會把它變多
    assert sum(trainer.y_train) > 0
    assert len(trainer.X_train) > 0

# 4. 測試：模型訓練與存檔屬性
def test_model_training(dummy_data):
    X, y = dummy_data
    trainer = BioMLTrainer(X, y)
    trainer.split_data()
    trainer.train_model()
    
    # 確保模型真的產生了
    assert trainer.model is not None
    # 確保這是 Random Forest (檢查有無特徵重要性屬性)
    assert hasattr(trainer.model, "feature_importances_")

# 5. 測試：預測流程不報錯
def test_prediction_flow(dummy_data):
    X, y = dummy_data
    trainer = BioMLTrainer(X, y)
    trainer.split_data()
    trainer.train_model()
    
    # 測試能不能吐出機率值
    # 我們隨便拿第一筆測試資料來預測
    sample_input = trainer.X_test[:1]
    prob = trainer.model.predict_proba(sample_input)
    
    # 檢查輸出格式是否正確 (要是 [0的機率, 1的機率])
    assert prob.shape == (1, 2)