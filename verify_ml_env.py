import sys
import os

def verify_libraries():
    print("=" * 60)
    print("Python version:", sys.version)
    print("=" * 60)
    
    libraries = [
        ("numpy", "np"),
        ("pandas", "pd"),
        ("matplotlib", "plt"),
        ("seaborn", "sns"),
        ("scipy", "sp"),
        ("statsmodels", "sm"),
        ("sklearn", "sklearn"),
        ("xgboost", "xgb"),
        ("lightgbm", "lgb"),
        ("torch", "torch"),
        ("tensorflow", "tf")
    ]
    
    for lib, alias in libraries:
        try:
            module = __import__(lib)
            version = getattr(module, "__version__", "unknown")
            print(f"[SUCCESS] {lib} is installed (version: {version})")
        except ImportError as e:
            print(f"[FAILED] {lib} is NOT installed. Error: {e}")
            
    print("=" * 60)
    print("Running basic functionality tests...")
    print("=" * 60)
    
    # 1. NumPy & Scipy
    try:
        import numpy as np
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        c = np.dot(a, b)
        print("[NumPy] Matrix multiplication: SUCCESS")
    except Exception as e:
        print(f"[NumPy] Test FAILED: {e}")

    # 2. Pandas & openpyxl
    try:
        import pandas as pd
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        # Test Excel writing/reading (requires openpyxl)
        excel_path = 'temp_test.xlsx'
        df.to_excel(excel_path, index=False)
        df_read = pd.read_excel(excel_path)
        if os.path.exists(excel_path):
            os.remove(excel_path)
        print("[Pandas & openpyxl] DataFrame Excel round-trip: SUCCESS")
    except Exception as e:
        print(f"[Pandas & openpyxl] Test FAILED: {e}")

    # 3. Scikit-Learn
    try:
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf.fit(X, y)
        print("[Scikit-Learn] Model training: SUCCESS")
    except Exception as e:
        print(f"[Scikit-Learn] Test FAILED: {e}")

    # 4. XGBoost & LightGBM
    try:
        import xgboost as xgb
        import lightgbm as lgb
        
        xgb_model = xgb.XGBClassifier(n_estimators=2, max_depth=2, eval_metric='logloss')
        xgb_model.fit(X, y)
        print("[XGBoost] Model training: SUCCESS")
        
        lgb_model = lgb.LGBMClassifier(n_estimators=2, max_depth=2, verbosity=-1)
        lgb_model.fit(X, y)
        print("[LightGBM] Model training: SUCCESS")
    except Exception as e:
        print(f"[XGBoost/LightGBM] Test FAILED: {e}")

    # 5. PyTorch
    try:
        import torch
        x = torch.rand(5, 3)
        cuda_avail = torch.cuda.is_available()
        print(f"[PyTorch] Tensor creation: SUCCESS (CUDA available: {cuda_avail})")
    except Exception as e:
        print(f"[PyTorch] Test FAILED: {e}")

    # 6. TensorFlow
    try:
        import tensorflow as tf
        msg = tf.constant('Hello, TensorFlow!')
        gpu_avail = len(tf.config.list_physical_devices('GPU')) > 0
        print(f"[TensorFlow] Constant creation: SUCCESS (GPU available: {gpu_avail})")
    except Exception as e:
        print(f"[TensorFlow] Test FAILED: {e}")

if __name__ == "__main__":
    verify_libraries()
