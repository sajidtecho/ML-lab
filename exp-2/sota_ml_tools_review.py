"""
Experiment 2: State-of-the-Art (SOTA) Tools in Machine Learning Experiments and Project Development

This script implements:
1. Automated Tool Environment Diagnostics & Version Audit
2. High-Performance Data Processing Benchmark (NumPy vs Pandas)
3. Comparative Modeling Benchmark (Scikit-Learn Random Forest vs XGBoost vs LightGBM)
4. Lightweight MLOps Experiment Tracking Logger
"""

import os
import sys
import time
import json
import platform
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import xgboost as xgb
import lightgbm as lgb

# Seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Setup Figures Directory
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style="whitegrid")


def audit_installed_tools():
    """Perform a diagnostic check on installed ML tools and libraries."""
    print("\n" + "="*80)
    print("1. SYSTEM & SOTA ML TOOL ENVIRONMENT DIAGNOSTICS")
    print("="*80)
    print(f"OS Platform     : {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version   : {sys.version.split()[0]}")
    print("-" * 80)
    
    toolset = [
        ("Data Engineering", "numpy", "NumPy"),
        ("Data Engineering", "pandas", "Pandas"),
        ("Scientific Computing", "scipy", "SciPy"),
        ("Visualization", "matplotlib", "Matplotlib"),
        ("Visualization", "seaborn", "Seaborn"),
        ("Classical ML", "sklearn", "Scikit-Learn"),
        ("Gradient Boosting", "xgboost", "XGBoost"),
        ("Gradient Boosting", "lightgbm", "LightGBM"),
        ("Deep Learning", "torch", "PyTorch"),
        ("Statistical Modeling", "statsmodels", "Statsmodels"),
    ]
    
    audit_data = []
    for category, module_name, display_name in toolset:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "Available")
            status = "Installed"
            audit_data.append({"Category": category, "Tool": display_name, "Status": status, "Version": version})
            print(f"[{category:20s}] {display_name:15s} : {version}")
        except ImportError:
            audit_data.append({"Category": category, "Tool": display_name, "Status": "Not Installed", "Version": "N/A"})
            print(f"[{category:20s}] {display_name:15s} : NOT INSTALLED")
            
    return pd.DataFrame(audit_data)


def benchmark_data_processing():
    """Compare NumPy vectorization speed vs Pandas DataFrame element-wise operation speed."""
    print("\n" + "="*80)
    print("2. DATA PROCESSING BENCHMARK (NUMPY VS PANDAS - 1 MILLION ELEMENTS)")
    print("="*80)
    
    n_elements = 1_000_000
    np_array = np.random.randn(n_elements)
    pd_series = pd.Series(np_array)
    
    # NumPy Vectorized Operation
    start = time.perf_counter()
    np_result = np.sin(np_array) ** 2 + np.cos(np_array) ** 2
    numpy_time = (time.perf_counter() - start) * 1000  # ms
    
    # Pandas Series Vectorized Operation
    start = time.perf_counter()
    pd_result = np.sin(pd_series) ** 2 + np.cos(pd_series) ** 2
    pandas_time = (time.perf_counter() - start) * 1000  # ms
    
    speedup = pandas_time / numpy_time if numpy_time > 0 else 1.0
    print(f"NumPy Vectorized Execution Time  : {numpy_time:.3f} ms")
    print(f"Pandas Vectorized Execution Time : {pandas_time:.3f} ms")
    print(f"NumPy Acceleration Factor        : {speedup:.2f}x faster")
    
    # Plot Speed Comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(['NumPy (Arrays)', 'Pandas (Series)'], [numpy_time, pandas_time], color=['#1f77b4', '#ff7f0e'], width=0.5)
    ax.set_ylabel('Execution Time (milliseconds)', fontsize=11, fontweight='bold')
    ax.set_title('Data Processing Performance (1 Million Mathematical Operations)', fontsize=12, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f} ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "data_processing_speedup.png"), dpi=300)
    plt.close()
    
    return {"NumPy (ms)": numpy_time, "Pandas (ms)": pandas_time, "Speedup": speedup}


class LightweightExperimentTracker:
    """Simulates an MLOps experiment tracking system (similar to MLflow / W&B)."""
    def __init__(self, experiment_name="ML_Tools_Benchmark"):
        self.experiment_name = experiment_name
        self.runs = []
        
    def log_run(self, run_name, model_type, hyperparameters, metrics, duration_ms):
        run_entry = {
            "Run ID": f"run_{len(self.runs)+1:03d}",
            "Run Name": run_name,
            "Model Type": model_type,
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Duration (ms)": round(duration_ms, 2),
            **hyperparameters,
            **metrics
        }
        self.runs.append(run_entry)
        
    def get_leaderboard(self):
        return pd.DataFrame(self.runs)


def benchmark_sota_models():
    """Benchmark Scikit-Learn Random Forest, XGBoost, and LightGBM."""
    print("\n" + "="*80)
    print("3. STATE-OF-THE-ART MODELING BENCHMARK (SKLEARN VS XGBOOST VS LIGHTGBM)")
    print("="*80)
    
    # 1. Load Dataset
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    tracker = LightweightExperimentTracker(experiment_name="SOTA_Classifiers")
    
    models = {
        "Scikit-Learn Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED),
        "XGBoost Classifier": xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=SEED, eval_metric='logloss'),
        "LightGBM Classifier": lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=SEED, verbosity=-1)
    }
    
    importance_dict = {}
    
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train_scaled, y_train)
        duration_ms = (time.perf_counter() - start) * 1000
        
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        metrics = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        }
        
        hyperparams = {"n_estimators": 100, "max_depth": model.max_depth if hasattr(model, 'max_depth') else 4}
        tracker.log_run(run_name=name, model_type=name.split()[0], hyperparameters=hyperparams, metrics=metrics, duration_ms=duration_ms)
        
        if hasattr(model, 'feature_importances_'):
            importance_dict[name] = model.feature_importances_
            
    df_leaderboard = tracker.get_leaderboard()
    print("\n[MLOps Experiment Tracker Leaderboard]")
    display_cols = ["Run ID", "Run Name", "Duration (ms)", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    print(df_leaderboard[display_cols].to_string(index=False))
    
    # Visualizations
    # 1. Training Time Comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    bars = ax.bar(df_leaderboard["Run Name"], df_leaderboard["Duration (ms)"], color=colors, width=0.5)
    ax.set_ylabel('Training Runtime (milliseconds)', fontsize=11, fontweight='bold')
    ax.set_title('SOTA Classifier Training Speed Comparison', fontsize=13, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.2f} ms', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "tools_benchmark_runtime.png"), dpi=300)
    plt.close()
    
    # 2. Performance Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    df_plot = df_leaderboard.set_index("Run Name")[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].T
    df_plot.plot(kind='bar', ax=ax, width=0.7, colormap='tab10')
    ax.set_title('SOTA ML Tool Model Performance Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Metric Score', fontsize=11)
    ax.set_ylim(0.9, 1.01)
    ax.legend(loc='lower right')
    plt.xticks(rotation=0, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "tools_benchmark_accuracy.png"), dpi=300)
    plt.close()
    
    # 3. Feature Importance Top 8 across Models
    if importance_dict:
        df_imp = pd.DataFrame(importance_dict, index=feature_names)
        df_imp["Mean_Importance"] = df_imp.mean(axis=1)
        top8 = df_imp.sort_values(by="Mean_Importance", ascending=False).head(8)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        top8[["Scikit-Learn Random Forest", "XGBoost Classifier", "LightGBM Classifier"]].plot(kind='barh', ax=ax)
        ax.set_title('Top 8 Feature Importance Comparison across SOTA Tree Frameworks', fontsize=12, fontweight='bold')
        ax.set_xlabel('Relative Feature Importance Score')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "feature_importance_gbdt.png"), dpi=300)
        plt.close()
        
    return df_leaderboard


def main():
    print("Executing Experiment 2...")
    df_audit = audit_installed_tools()
    proc_metrics = benchmark_data_processing()
    df_leaderboard = benchmark_sota_models()
    print(f"\nExperiment 2 Execution Completed Successfully!")
    print(f"All figures saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    main()
