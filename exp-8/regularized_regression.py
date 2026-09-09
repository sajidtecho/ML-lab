"""
Experiment 8: Ridge and Lasso Regularized Linear Regression vs Standard Linear Regression

This script implements:
1. Standard Linear Regression (Ordinary Least Squares - OLS)
2. Ridge Regression (L2 Regularization)
3. Lasso Regression (L1 Regularization)

Tested on:
- California Housing Dataset (Continuous Regression)
- Breast Cancer Wisconsin Dataset (Binary Classification & Linear Probability Model)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Ensure reproducibility
SEED = 42
np.random.seed(SEED)

# Setup Output Directory for Figures
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set plotting theme
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style="whitegrid")


def evaluate_regression(y_true_train, y_pred_train, y_true_test, y_pred_test, model_name):
    """Compute and return regression metrics for train and test sets."""
    mse_train = mean_squared_error(y_true_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    mae_train = mean_absolute_error(y_true_train, y_pred_train)
    r2_train = r2_score(y_true_train, y_pred_train)
    
    mse_test = mean_squared_error(y_true_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_true_test, y_pred_test)
    r2_test = r2_score(y_true_test, y_pred_test)
    
    return {
        "Model": model_name,
        "Train MSE": mse_train,
        "Test MSE": mse_test,
        "Train RMSE": rmse_train,
        "Test RMSE": rmse_test,
        "Train MAE": mae_train,
        "Test MAE": mae_test,
        "Train R2": r2_train,
        "Test R2": r2_test
    }


def evaluate_classification(y_true, y_pred_prob, threshold=0.5):
    """Compute classification performance from probability/continuous predictions."""
    y_pred_binary = (y_pred_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred_binary)
    prec = precision_score(y_true, y_pred_binary, zero_division=0)
    rec = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_prob)
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc
    }


def run_california_housing_experiment():
    print("\n" + "="*80)
    print("PART 1: CALIFORNIA HOUSING DATASET (Continuous Regression)")
    print("="*80)
    
    # 1. Load Dataset
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    print(f"Dataset Shape: X = {X.shape}, y = {y.shape}")
    print(f"Features: {feature_names}")
    
    # 2. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    
    # 3. Standard Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Instantiate Models
    # a. Standard Linear Regression (OLS)
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train)
    
    # b. Ridge Regression with 5-Fold Cross-Validation for Optimal Alpha
    alphas_ridge = np.logspace(-3, 5, 200)
    ridge_cv = RidgeCV(alphas=alphas_ridge, cv=5, scoring='neg_mean_squared_error')
    ridge_cv.fit(X_train_scaled, y_train)
    best_alpha_ridge = ridge_cv.alpha_
    
    # c. Lasso Regression with 5-Fold Cross-Validation for Optimal Alpha
    alphas_lasso = np.logspace(-4, 2, 200)
    lasso_cv = LassoCV(alphas=alphas_lasso, cv=5, max_iter=20000, random_state=SEED)
    lasso_cv.fit(X_train_scaled, y_train)
    best_alpha_lasso = lasso_cv.alpha_
    
    print(f"\n[Hyperparameter Tuning Results]")
    print(f"Optimal Alpha for Ridge (L2): {best_alpha_ridge:.4f}")
    print(f"Optimal Alpha for Lasso (L1): {best_alpha_lasso:.6f}")
    
    # 5. Predict and Evaluate
    models = {
        "Standard Linear Regression (OLS)": ols,
        f"Ridge Regression (L2, alpha={best_alpha_ridge:.2f})": ridge_cv,
        f"Lasso Regression (L1, alpha={best_alpha_lasso:.4f})": lasso_cv
    }
    
    results = []
    coef_dict = {}
    
    for name, model in models.items():
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        metrics = evaluate_regression(y_train, y_pred_train, y_test, y_pred_test, name)
        results.append(metrics)
        coef_dict[name] = model.coef_
        
    df_results = pd.DataFrame(results)
    print("\n[Performance Metrics Comparison - California Housing]")
    print(df_results.to_string(index=False))
    
    # 6. Coefficient Analysis
    coef_df = pd.DataFrame(coef_dict, index=feature_names)
    print("\n[Learned Feature Coefficients (Standardized Scale)]")
    print(coef_df.to_string())
    
    print("\n[Feature Sparsity Analysis]")
    print(f"OLS Non-Zero Coefficients: {np.sum(ols.coef_ != 0)} / {len(feature_names)}")
    print(f"Ridge Non-Zero Coefficients: {np.sum(ridge_cv.coef_ != 0)} / {len(feature_names)}")
    print(f"Lasso Non-Zero Coefficients: {np.sum(lasso_cv.coef_ != 0)} / {len(feature_names)}")
    print(f"Lasso Zeroed-Out Features: {list(np.array(feature_names)[lasso_cv.coef_ == 0])}")
    
    # 7. Visualizations
    # Figure 1: Coefficients Bar Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x_indices = np.arange(len(feature_names))
    width = 0.25
    
    ax.bar(x_indices - width, ols.coef_, width, label='OLS Linear Regression', color='#1f77b4', alpha=0.9)
    ax.bar(x_indices, ridge_cv.coef_, width, label=f'Ridge (L2, alpha={best_alpha_ridge:.2f})', color='#2ca02c', alpha=0.9)
    ax.bar(x_indices + width, lasso_cv.coef_, width, label=f'Lasso (L1, alpha={best_alpha_lasso:.4f})', color='#d62728', alpha=0.9)
    
    ax.set_ylabel('Standardized Coefficient Magnitude', fontsize=12, fontweight='bold')
    ax.set_title('California Housing: Feature Coefficients Comparison across OLS, Ridge, & Lasso', fontsize=13, fontweight='bold')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(feature_names, rotation=30, ha='right')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "california_housing_coefficients.png"), dpi=300)
    plt.close()
    
    # Figure 2: Actual vs Predicted Scatter Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    model_names_short = ["OLS Linear Regression", "Ridge Regression", "Lasso Regression"]
    model_objs = [ols, ridge_cv, lasso_cv]
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    for ax, name, model, color in zip(axes, model_names_short, model_objs, colors):
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        ax.scatter(y_test, y_pred, alpha=0.3, color=color, edgecolors='none', s=15)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal Prediction')
        ax.set_xlabel('Actual MedHouseVal ($100k)', fontsize=11)
        ax.set_ylabel('Predicted MedHouseVal ($100k)', fontsize=11)
        ax.set_title(f"{name}\n$R^2$: {r2:.4f} | RMSE: {rmse:.4f}", fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        
    plt.suptitle("California Housing: Actual vs Predicted Values", fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "california_housing_actual_vs_pred.png"), dpi=300)
    plt.close()
    
    # Figure 3: Lasso Alpha Regularization Path
    fig, ax = plt.subplots(figsize=(9, 5))
    alphas_path = np.logspace(-4, 1, 100)
    coefs_lasso_path = []
    for a in alphas_path:
        l = Lasso(alpha=a, max_iter=20000, random_state=SEED)
        l.fit(X_train_scaled, y_train)
        coefs_lasso_path.append(l.coef_)
        
    ax.plot(alphas_path, coefs_lasso_path)
    ax.set_xscale('log')
    ax.set_xlabel('Regularization Strength alpha (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficients', fontsize=12, fontweight='bold')
    ax.set_title('Lasso Regularization Path (California Housing Features)', fontsize=13, fontweight='bold')
    ax.axvline(best_alpha_lasso, color='red', linestyle='--', label=f'Optimal Alpha ({best_alpha_lasso:.4f})')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "california_housing_lasso_path.png"), dpi=300)
    plt.close()

    return df_results, coef_df


def run_breast_cancer_experiment():
    print("\n" + "="*80)
    print("PART 2: BREAST CANCER WISCONSIN DATASET (Binary Classification)")
    print("="*80)
    
    # 1. Load Dataset
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names  # 0: malignant, 1: benign
    
    print(f"Dataset Shape: X = {X.shape}, y = {y.shape}")
    print(f"Target distribution: {np.bincount(y)} (0: {target_names[0]}, 1: {target_names[1]})")
    
    # 2. Train-Test Split (80/20 Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    # 3. Standard Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- APPROACH A: Regularized Linear Regression (Linear Probability Model) ---
    print("\n--- [Approach A: Regularized Linear Regression as Linear Probability Model] ---")
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train)
    
    alphas_ridge = np.logspace(-2, 4, 200)
    ridge_cv = RidgeCV(alphas=alphas_ridge, cv=5, scoring='neg_mean_squared_error')
    ridge_cv.fit(X_train_scaled, y_train)
    
    alphas_lasso = np.logspace(-4, 0, 200)
    lasso_cv = LassoCV(alphas=alphas_lasso, cv=5, max_iter=20000, random_state=SEED)
    lasso_cv.fit(X_train_scaled, y_train)
    
    reg_models = {
        "OLS Linear Regression": ols,
        f"Ridge Regression (alpha={ridge_cv.alpha_:.2f})": ridge_cv,
        f"Lasso Regression (alpha={lasso_cv.alpha_:.4f})": lasso_cv
    }
    
    lin_prob_results = []
    for name, model in reg_models.items():
        y_pred_prob_test = model.predict(X_test_scaled)
        metrics = evaluate_classification(y_test, y_pred_prob_test, threshold=0.5)
        metrics["Model"] = name
        metrics["Test MSE"] = mean_squared_error(y_test, y_pred_prob_test)
        metrics["Test R2"] = r2_score(y_test, y_pred_prob_test)
        metrics["Non-Zero Features"] = np.sum(model.coef_ != 0)
        lin_prob_results.append(metrics)
        
    df_lin_prob = pd.DataFrame(lin_prob_results)
    cols_order = ["Model", "Test MSE", "Test R2", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Non-Zero Features"]
    df_lin_prob = df_lin_prob[cols_order]
    print("\n[Linear Probability Model Performance - Breast Cancer]")
    print(df_lin_prob.to_string(index=False))
    
    # --- APPROACH B: Logistic Regression (Unregularized vs L1 vs L2) ---
    print("\n--- [Approach B: Logistic Regression (Unregularized vs L2 vs L1 Regularization)] ---")
    
    log_unreg = LogisticRegression(penalty=None, max_iter=10000, random_state=SEED)
    log_unreg.fit(X_train_scaled, y_train)
    
    log_l2 = LogisticRegression(penalty='l2', C=1.0, max_iter=10000, random_state=SEED)
    log_l2.fit(X_train_scaled, y_train)
    
    log_l1 = LogisticRegression(penalty='l1', solver='saga', C=0.5, max_iter=10000, random_state=SEED)
    log_l1.fit(X_train_scaled, y_train)
    
    clf_models = {
        "Unregularized Logistic Regression": log_unreg,
        "Ridge Logistic Regression (L2, C=1.0)": log_l2,
        "Lasso Logistic Regression (L1, C=0.5)": log_l1
    }
    
    log_results = []
    for name, model in clf_models.items():
        y_pred_prob_test = model.predict_proba(X_test_scaled)[:, 1]
        metrics = evaluate_classification(y_test, y_pred_prob_test, threshold=0.5)
        metrics["Model"] = name
        metrics["Non-Zero Features"] = np.sum(model.coef_[0] != 0)
        log_results.append(metrics)
        
    df_log = pd.DataFrame(log_results)
    cols_clf = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Non-Zero Features"]
    df_log = df_log[cols_clf]
    print("\n[Logistic Regression Regularization Comparison - Breast Cancer]")
    print(df_log.to_string(index=False))
    
    # 5. Visualizations
    # Figure 1: Coefficient Sparsity & Magnitudes for Breast Cancer Linear Models
    fig, ax = plt.subplots(figsize=(14, 7))
    x_indices = np.arange(len(feature_names))
    width = 0.25
    
    ax.bar(x_indices - width, ols.coef_, width, label='OLS Linear Regression', color='#1f77b4', alpha=0.85)
    ax.bar(x_indices, ridge_cv.coef_, width, label=f'Ridge Regression (alpha={ridge_cv.alpha_:.2f})', color='#2ca02c', alpha=0.85)
    ax.bar(x_indices + width, lasso_cv.coef_, width, label=f'Lasso Regression (alpha={lasso_cv.alpha_:.4f})', color='#d62728', alpha=0.85)
    
    ax.set_ylabel('Standardized Coefficient Weight', fontsize=12, fontweight='bold')
    ax.set_title('Breast Cancer Dataset: Feature Weights Across OLS, Ridge, & Lasso (Linear Regression)', fontsize=13, fontweight='bold')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(feature_names, rotation=90, ha='right', fontsize=9)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "breast_cancer_coefficients.png"), dpi=300)
    plt.close()

    # Figure 2: Metrics Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    df_plot = df_lin_prob.set_index('Model')[metrics_to_plot].T
    
    df_plot.plot(kind='bar', ax=ax, width=0.7, colormap='viridis')
    ax.set_title('Breast Cancer Classification Performance: Linear Regression (OLS) vs Ridge vs Lasso', fontsize=13, fontweight='bold')
    ax.set_ylabel('Metric Score', fontsize=12, fontweight='bold')
    ax.set_ylim(0.8, 1.02)
    ax.legend(loc='lower right', fontsize=10)
    plt.xticks(rotation=0, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "breast_cancer_performance.png"), dpi=300)
    plt.close()
    
    return df_lin_prob, df_log


def main():
    print("Starting Experiment 8 Execution...")
    housing_metrics, housing_coefs = run_california_housing_experiment()
    cancer_lin_metrics, cancer_log_metrics = run_breast_cancer_experiment()
    print("\nExperiment 8 Execution Completed Successfully!")
    print(f"All figures saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    main()
