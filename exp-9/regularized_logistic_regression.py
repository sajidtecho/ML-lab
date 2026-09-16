"""
Experiment 9: Regularized Logistic Regression (L1 Lasso vs L2 Ridge)

This script implements:
1. Custom Regularized Logistic Regression from Scratch (NumPy Gradient & Proximal Descent for L1 & L2 penalties)
2. Scikit-Learn Regularized Logistic Regression (L1, L2, ElasticNet, and Unregularized baselines)
3. Hyperparameter tuning over Regularization Strength C (C = 1/lambda)
4. Comprehensive evaluation & comparative analysis across:
   - California Housing Dataset (Converted to Binary Classification: High vs Low Price)
   - Breast Cancer Wisconsin Dataset (Binary Medical Classification)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, roc_curve
)

# Seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Output directory for figures
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style="whitegrid")


class RegularizedLogisticRegressionScratch:
    """
    Binary Logistic Regression from Scratch with L1 (Lasso) and L2 (Ridge) Regularization.
    
    Mathematical Formulation:
    - Sigmoid hypothesis: h_w,b(x) = 1 / (1 + exp(-(w^T * x + b)))
    - Unregularized Log-Loss: J(w,b) = - (1/n) * sum[ y*log(p) + (1-y)*log(1-p) ]
    - L2 Penalty: (lambda / (2*n)) * ||w||_2^2
      -> Gradient w.r.t w: (1/n) * X^T * (p - y) + (lambda / n) * w
    - L1 Penalty: (lambda / n) * ||w||_1
      -> Update via Proximal Soft-Thresholding: S_{eta*lambda}(w) = sign(w) * max(0, |w| - eta*lambda/n)
    """
    def __init__(self, penalty='l2', lmbda=0.1, learning_rate=0.05, num_iterations=1000, verbose=False):
        self.penalty = penalty.lower() if penalty else 'none'
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _soft_thresholding(self, w, threshold):
        """Soft-thresholding operator for L1 proximal gradient step."""
        return np.sign(w) * np.maximum(0.0, np.abs(w) - threshold)

    def _compute_loss(self, y_true, y_pred_prob, n_samples):
        epsilon = 1e-15
        p = np.clip(y_pred_prob, epsilon, 1 - epsilon)
        bce = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
        
        if self.penalty == 'l2':
            reg = (self.lmbda / (2.0 * n_samples)) * np.sum(self.weights ** 2)
        elif self.penalty == 'l1':
            reg = (self.lmbda / n_samples) * np.sum(np.abs(self.weights))
        else:
            reg = 0.0
            
        return bce + reg

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            loss = self._compute_loss(y, y_pred, n_samples)
            self.loss_history.append(loss)

            # Gradient of Log Loss
            dw_loss = (1.0 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1.0 / n_samples) * np.sum(y_pred - y)

            if self.penalty == 'l2':
                dw = dw_loss + (self.lmbda / n_samples) * self.weights
                self.weights -= self.learning_rate * dw
            elif self.penalty == 'l1':
                # Proximal gradient update for L1
                w_temp = self.weights - self.learning_rate * dw_loss
                threshold = self.learning_rate * (self.lmbda / n_samples)
                self.weights = self._soft_thresholding(w_temp, threshold)
            else:
                self.weights -= self.learning_rate * dw_loss

            self.bias -= self.learning_rate * db

            if self.verbose and (i % max(1, self.num_iterations // 10) == 0):
                print(f"Iter {i:4d}/{self.num_iterations} | Loss: {loss:.5f}")

        return self

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def evaluate_classification(y_true, y_pred_prob, model_name, threshold=0.5):
    """Compute and format classification metrics."""
    y_pred = (y_pred_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_prob)
    loss = log_loss(y_true, y_pred_prob)
    
    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "Log-Loss": loss
    }


def run_california_housing_classification_experiment():
    print("\n" + "="*80)
    print("PART 1: CALIFORNIA HOUSING DATASET (Converted to Binary Classification)")
    print("Task: Predict whether Median House Value is Above vs Below Median Price ($206.85k)")
    print("="*80)

    # 1. Load Data
    data = fetch_california_housing(as_frame=True)
    X, y_cont = data.data, data.target
    feature_names = data.feature_names

    # Convert continuous target into binary class (1 if >= median price, 0 otherwise)
    median_val = y_cont.median()
    y_binary = (y_cont >= median_val).astype(int)

    print(f"Dataset Shape: X = {X.shape}, y = {y_binary.shape}")
    print(f"Target Binarization Threshold: {median_val:.4f} ($206,850)")
    print(f"Class Distribution: High Price (1) = {np.sum(y_binary==1)}, Low Price (0) = {np.sum(y_binary==0)}")

    # 2. Train-Test Split (80/20, Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=SEED, stratify=y_binary
    )

    # 3. Standard Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Fit Baseline Models (Default C = 1.0)
    models = {
        "Unregularized (None)": LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000, random_state=SEED),
        "L2 Regularized (Ridge)": LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000, random_state=SEED),
        "L1 Regularized (Lasso)": LogisticRegression(penalty='l1', C=1.0, solver='saga', max_iter=2000, random_state=SEED),
        "ElasticNet (L1+L2 50/50)": LogisticRegression(penalty='elasticnet', C=1.0, l1_ratio=0.5, solver='saga', max_iter=2000, random_state=SEED)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
        metrics = evaluate_classification(y_test, y_prob_test, name)
        
        # Count non-zero coefficients
        coefs = model.coef_.flatten()
        metrics["Zero Coefs"] = np.sum(np.abs(coefs) < 1e-4)
        metrics["Non-Zero Coefs"] = np.sum(np.abs(coefs) >= 1e-4)
        
        results.append(metrics)
        trained_models[name] = model

    df_results = pd.DataFrame(results)
    print("\n--- California Housing Classification Results (Default C=1.0) ---")
    print(df_results.to_string(index=False))

    # 5. Hyperparameter Sweep over Regularization Strength C
    C_values = np.logspace(-4, 4, 15)
    l1_coefs, l2_coefs = [], []
    l1_acc, l2_acc = [], []
    l1_loss, l2_loss = [], []
    l1_zero_cnt, l2_zero_cnt = [], []

    for c in C_values:
        # L1 Logistic
        clf_l1 = LogisticRegression(penalty='l1', C=c, solver='saga', max_iter=2500, random_state=SEED)
        clf_l1.fit(X_train_scaled, y_train)
        p1 = clf_l1.predict_proba(X_test_scaled)[:, 1]
        l1_coefs.append(clf_l1.coef_.flatten())
        l1_acc.append(accuracy_score(y_test, (p1 >= 0.5).astype(int)))
        l1_loss.append(log_loss(y_test, p1))
        l1_zero_cnt.append(np.sum(np.abs(clf_l1.coef_.flatten()) < 1e-4))

        # L2 Logistic
        clf_l2 = LogisticRegression(penalty='l2', C=c, solver='lbfgs', max_iter=2500, random_state=SEED)
        clf_l2.fit(X_train_scaled, y_train)
        p2 = clf_l2.predict_proba(X_test_scaled)[:, 1]
        l2_coefs.append(clf_l2.coef_.flatten())
        l2_acc.append(accuracy_score(y_test, (p2 >= 0.5).astype(int)))
        l2_loss.append(log_loss(y_test, p2))
        l2_zero_cnt.append(np.sum(np.abs(clf_l2.coef_.flatten()) < 1e-4))

    l1_coefs = np.array(l1_coefs)
    l2_coefs = np.array(l2_coefs)

    # 6. Plotting - Figure 1: Coefficient Paths (L1 vs L2)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for j in range(X.shape[1]):
        plt.plot(np.log10(C_values), l1_coefs[:, j], label=feature_names[j], linewidth=2)
    plt.axvline(x=0, color='grey', linestyle='--', alpha=0.7, label='Default C=1.0')
    plt.title("L1 (Lasso) Regularized Logistic Regression: Coefficient Path", fontsize=12, fontweight='bold')
    plt.xlabel("log10(C) [Inverse Regularization Strength]", fontsize=10)
    plt.ylabel("Learned Feature Coefficients (w)", fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.subplot(1, 2, 2)
    for j in range(X.shape[1]):
        plt.plot(np.log10(C_values), l2_coefs[:, j], label=feature_names[j], linewidth=2)
    plt.axvline(x=0, color='grey', linestyle='--', alpha=0.7, label='Default C=1.0')
    plt.title("L2 (Ridge) Regularized Logistic Regression: Coefficient Path", fontsize=12, fontweight='bold')
    plt.xlabel("log10(C) [Inverse Regularization Strength]", fontsize=10)
    plt.ylabel("Learned Feature Coefficients (w)", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "california_logistic_coeff_paths.png"), dpi=300)
    plt.close()

    # Plotting - Figure 2: Accuracy, Log Loss & Zero Coefficients vs C
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(np.log10(C_values), l1_acc, 'o-', label="L1 (Lasso)", color='crimson', linewidth=2)
    plt.plot(np.log10(C_values), l2_acc, 's-', label="L2 (Ridge)", color='royalblue', linewidth=2)
    plt.title("Test Accuracy vs Regularization Strength (C)", fontsize=11, fontweight='bold')
    plt.xlabel("log10(C)", fontsize=10)
    plt.ylabel("Test Accuracy", fontsize=10)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(np.log10(C_values), l1_loss, 'o-', label="L1 (Lasso)", color='crimson', linewidth=2)
    plt.plot(np.log10(C_values), l2_loss, 's-', label="L2 (Ridge)", color='royalblue', linewidth=2)
    plt.title("Test Log-Loss vs Regularization Strength (C)", fontsize=11, fontweight='bold')
    plt.xlabel("log10(C)", fontsize=10)
    plt.ylabel("Log-Loss", fontsize=10)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(np.log10(C_values), l1_zero_cnt, 'o-', label="L1 Zero Coefficients", color='crimson', linewidth=2)
    plt.plot(np.log10(C_values), l2_zero_cnt, 's-', label="L2 Zero Coefficients", color='royalblue', linewidth=2)
    plt.title("Feature Sparsity (Zeroed Features) vs C", fontsize=11, fontweight='bold')
    plt.xlabel("log10(C)", fontsize=10)
    plt.ylabel("Count of Zero Coefficients (|w| < 1e-4)", fontsize=10)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "california_logistic_metrics_vs_c.png"), dpi=300)
    plt.close()

    # Plotting - Figure 3: ROC Curves & Confusion Matrix (California Housing)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for name, model in trained_models.items():
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label="Random Baseline (AUC = 0.50)")
    plt.title("ROC Curves - California Housing Classification", fontsize=11, fontweight='bold')
    plt.xlabel("False Positive Rate", fontsize=10)
    plt.ylabel("True Positive Rate", fontsize=10)
    plt.legend(fontsize=8)

    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y_test, trained_models["L1 Regularized (Lasso)"].predict(X_test_scaled))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Low Price', 'High Price'], yticklabels=['Low Price', 'High Price'])
    plt.title("Confusion Matrix - L1 Logistic Regression", fontsize=11, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=10)
    plt.ylabel("True Label", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "california_logistic_roc_cm.png"), dpi=300)
    plt.close()

    return df_results


def run_breast_cancer_experiment():
    print("\n" + "="*80)
    print("PART 2: BREAST CANCER WISCONSIN DATASET (Binary Medical Classification)")
    print("Task: Predict Malignant (0) vs Benign (1) Tumor Diagnosis")
    print("="*80)

    # 1. Load Data
    cancer = load_breast_cancer(as_frame=True)
    X, y = cancer.data, cancer.target
    feature_names = cancer.feature_names

    print(f"Dataset Shape: X = {X.shape}, y = {y.shape}")
    print(f"Class Distribution: Benign (1) = {np.sum(y==1)}, Malignant (0) = {np.sum(y==0)}")

    # 2. Train-Test Split (80/20, Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # 3. Standard Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Custom Scratch Implementation vs Scikit-Learn Comparison
    print("\n--- Training Custom Scratch Regularized Logistic Regression Models ---")
    scratch_l2 = RegularizedLogisticRegressionScratch(penalty='l2', lmbda=1.0, learning_rate=0.1, num_iterations=1500)
    scratch_l2.fit(X_train_scaled, y_train.values)
    p_scratch_l2 = scratch_l2.predict_proba(X_test_scaled)
    m_scratch_l2 = evaluate_classification(y_test, p_scratch_l2, "Scratch Regularized (L2, lambda=1.0)")

    scratch_l1 = RegularizedLogisticRegressionScratch(penalty='l1', lmbda=1.0, learning_rate=0.1, num_iterations=1500)
    scratch_l1.fit(X_train_scaled, y_train.values)
    p_scratch_l1 = scratch_l1.predict_proba(X_test_scaled)
    m_scratch_l1 = evaluate_classification(y_test, p_scratch_l1, "Scratch Regularized (L1, lambda=1.0)")

    # Scikit-Learn Models
    models = {
        "Unregularized (None)": LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000, random_state=SEED),
        "L2 Regularized (Ridge)": LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000, random_state=SEED),
        "L1 Regularized (Lasso)": LogisticRegression(penalty='l1', C=1.0, solver='saga', max_iter=2000, random_state=SEED),
        "ElasticNet (L1+L2 50/50)": LogisticRegression(penalty='elasticnet', C=1.0, l1_ratio=0.5, solver='saga', max_iter=2000, random_state=SEED)
    }

    results = [m_scratch_l2, m_scratch_l1]
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
        metrics = evaluate_classification(y_test, y_prob_test, f"Scikit-Learn {name}")
        coefs = model.coef_.flatten()
        metrics["Zero Coefs"] = np.sum(np.abs(coefs) < 1e-4)
        metrics["Non-Zero Coefs"] = np.sum(np.abs(coefs) >= 1e-4)
        results.append(metrics)
        trained_models[name] = model

    m_scratch_l2["Zero Coefs"] = np.sum(np.abs(scratch_l2.weights) < 1e-4)
    m_scratch_l2["Non-Zero Coefs"] = np.sum(np.abs(scratch_l2.weights) >= 1e-4)
    m_scratch_l1["Zero Coefs"] = np.sum(np.abs(scratch_l1.weights) < 1e-4)
    m_scratch_l1["Non-Zero Coefs"] = np.sum(np.abs(scratch_l1.weights) >= 1e-4)

    df_results = pd.DataFrame(results)
    print("\n--- Breast Cancer Classification Performance Comparison ---")
    print(df_results.to_string(index=False))

    # 5. Hyperparameter Sweep over C (Breast Cancer)
    C_values = np.logspace(-4, 4, 15)
    l1_coefs, l2_coefs = [], []
    l1_acc, l2_acc = [], []
    l1_loss, l2_loss = [], []
    l1_zero_cnt, l2_zero_cnt = [], []

    for c in C_values:
        clf_l1 = LogisticRegression(penalty='l1', C=c, solver='saga', max_iter=3000, random_state=SEED)
        clf_l1.fit(X_train_scaled, y_train)
        p1 = clf_l1.predict_proba(X_test_scaled)[:, 1]
        l1_coefs.append(clf_l1.coef_.flatten())
        l1_acc.append(accuracy_score(y_test, (p1 >= 0.5).astype(int)))
        l1_loss.append(log_loss(y_test, p1))
        l1_zero_cnt.append(np.sum(np.abs(clf_l1.coef_.flatten()) < 1e-4))

        clf_l2 = LogisticRegression(penalty='l2', C=c, solver='lbfgs', max_iter=3000, random_state=SEED)
        clf_l2.fit(X_train_scaled, y_train)
        p2 = clf_l2.predict_proba(X_test_scaled)[:, 1]
        l2_coefs.append(clf_l2.coef_.flatten())
        l2_acc.append(accuracy_score(y_test, (p2 >= 0.5).astype(int)))
        l2_loss.append(log_loss(y_test, p2))
        l2_zero_cnt.append(np.sum(np.abs(clf_l2.coef_.flatten()) < 1e-4))

    l1_coefs = np.array(l1_coefs)
    l2_coefs = np.array(l2_coefs)

    # 6. Plotting - Figure 4: Coefficient Paths for 30 Features (Breast Cancer)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for j in range(X.shape[1]):
        plt.plot(np.log10(C_values), l1_coefs[:, j], alpha=0.7, linewidth=1.5)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Default C=1.0')
    plt.title("Breast Cancer L1 (Lasso) Logistic: Coefficient Path (30 Features)", fontsize=11, fontweight='bold')
    plt.xlabel("log10(C)", fontsize=10)
    plt.ylabel("Feature Weight (w)", fontsize=10)
    plt.legend()

    plt.subplot(1, 2, 2)
    for j in range(X.shape[1]):
        plt.plot(np.log10(C_values), l2_coefs[:, j], alpha=0.7, linewidth=1.5)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Default C=1.0')
    plt.title("Breast Cancer L2 (Ridge) Logistic: Coefficient Path (30 Features)", fontsize=11, fontweight='bold')
    plt.xlabel("log10(C)", fontsize=10)
    plt.ylabel("Feature Weight (w)", fontsize=10)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "breast_cancer_logistic_coeff_paths.png"), dpi=300)
    plt.close()

    # Plotting - Figure 5: Metrics vs C (Breast Cancer)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(np.log10(C_values), l1_acc, 'o-', label="L1 (Lasso)", color='crimson', linewidth=2)
    plt.plot(np.log10(C_values), l2_acc, 's-', label="L2 (Ridge)", color='royalblue', linewidth=2)
    plt.title("Breast Cancer Test Accuracy vs log10(C)", fontsize=11, fontweight='bold')
    plt.xlabel("log10(C)", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(np.log10(C_values), l1_loss, 'o-', label="L1 (Lasso)", color='crimson', linewidth=2)
    plt.plot(np.log10(C_values), l2_loss, 's-', label="L2 (Ridge)", color='royalblue', linewidth=2)
    plt.title("Breast Cancer Test Log-Loss vs log10(C)", fontsize=11, fontweight='bold')
    plt.xlabel("log10(C)", fontsize=10)
    plt.ylabel("Log-Loss", fontsize=10)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(np.log10(C_values), l1_zero_cnt, 'o-', label="L1 Zero Coefs (Sparsity)", color='crimson', linewidth=2)
    plt.plot(np.log10(C_values), l2_zero_cnt, 's-', label="L2 Zero Coefs", color='royalblue', linewidth=2)
    plt.title("Breast Cancer Zero Coefficients out of 30 vs C", fontsize=11, fontweight='bold')
    plt.xlabel("log10(C)", fontsize=10)
    plt.ylabel("Zeroed Feature Count", fontsize=10)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "breast_cancer_logistic_metrics_vs_c.png"), dpi=300)
    plt.close()

    # Plotting - Figure 6: Custom Scratch vs Scikit-Learn Loss Convergence
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(scratch_l2.loss_history, label="Scratch L2 (Ridge) GD Loss", color='royalblue', linewidth=2)
    plt.plot(scratch_l1.loss_history, label="Scratch L1 (Lasso) Proximal Loss", color='crimson', linewidth=2)
    plt.title("Scratch Gradient Descent Optimization Loss Curves", fontsize=11, fontweight='bold')
    plt.xlabel("Iteration", fontsize=10)
    plt.ylabel("Regularized Log-Loss", fontsize=10)
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, model in trained_models.items():
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label="Random Baseline")
    plt.title("ROC Curves - Breast Cancer Classification", fontsize=11, fontweight='bold')
    plt.xlabel("False Positive Rate", fontsize=10)
    plt.ylabel("True Positive Rate", fontsize=10)
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "breast_cancer_scratch_vs_sklearn.png"), dpi=300)
    plt.close()

    return df_results


def main():
    print("="*80)
    print("EXPERIMENT 9: REGULARIZED LOGISTIC REGRESSION (L1 LASSO & L2 RIDGE)")
    print("="*80)
    
    df_california = run_california_housing_classification_experiment()
    df_cancer = run_breast_cancer_experiment()

    print("\n" + "="*80)
    print("SUMMARY COMPARISON & CONCLUSION")
    print("="*80)
    print("1. L1 Regularization (Lasso) successfully eliminates irrelevant features by shrinking their coefficients to exactly zero.")
    print("2. L2 Regularization (Ridge) maintains all feature coefficients while shrinking their magnitudes smoothly, preventing extreme parameter values.")
    print("3. High Regularization (Low C / High lambda) reduces variance and prevents overfitting, but over-regularization damages model recall and log-loss.")
    print("4. ElasticNet offers a hybrid compromise combining both L1 feature selection and L2 weight stability.")
    print("="*80)


if __name__ == "__main__":
    main()
