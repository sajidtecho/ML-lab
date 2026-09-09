"""
Experiment 5: Binary Logistic Regression for Cancer Identification
Dataset: Breast Cancer Wisconsin Dataset

Implementations:
1. Binary Logistic Regression from Scratch (NumPy Gradient Descent)
2. Scikit-Learn LogisticRegression (Production Baseline)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, roc_curve,
    precision_recall_curve, ClassificationReportVisualizer if False else None
)

# Seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Setup Figures Directory
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style="whitegrid")


class BinaryLogisticRegressionScratch:
    """
    Binary Logistic Regression implementation from scratch using Gradient Descent.
    
    Mathematical Formulation:
    - Hypothesis: h(x) = sigmoid(w^T * x + b)
    - Loss: Binary Cross-Entropy (Log Loss)
    - Gradients: dw = (1/n) * X^T * (p - y), db = (1/n) * sum(p - y)
    """
    def __init__(self, learning_rate=0.01, num_iterations=1000, verbose=False):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    @staticmethod
    def _sigmoid(z):
        # Clip z to prevent overflow in exp
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))
    
    @staticmethod
    def _compute_loss(y_true, y_pred_prob):
        epsilon = 1e-15
        y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []
        
        for i in range(self.num_iterations):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            # Loss computation
            loss = self._compute_loss(y, y_predicted)
            self.loss_history.append(loss)
            
            # Gradient computation
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Parameter update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if self.verbose and i % (self.num_iterations // 10) == 0:
                print(f"Iteration {i:4d}/{self.num_iterations} | Loss: {loss:.5f}")
                
        return self
        
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
        
    def predict(self, X, threshold=0.5):
        y_predicted_cls = self.predict_proba(X)
        return (y_predicted_cls >= threshold).astype(int)


def calculate_metrics(y_true, y_pred_prob, threshold=0.5):
    """Compute detailed evaluation metrics including sensitivity and specificity."""
    y_pred = (y_pred_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity / True Positive Rate
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0     # Specificity / True Negative Rate
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_prob)
    loss = log_loss(y_true, y_pred_prob)
    
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall (Sensitivity)": rec,
        "Specificity": spec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "Log Loss": loss,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn
    }


def main():
    print("="*80)
    print("EXPERIMENT 5: BINARY LOGISTIC REGRESSION (BREAST CANCER IDENTIFICATION)")
    print("="*80)
    
    # 1. Load Dataset
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names  # 0: malignant, 1: benign
    
    print(f"Dataset Dimensions: X = {X.shape}, y = {y.shape}")
    print(f"Class Counts: {np.bincount(y)} (0: {target_names[0]} [Malignant], 1: {target_names[1]} [Benign])")
    
    # 2. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    # 3. Standard Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train Scratch Logistic Regression Model
    print("\n--- Training Custom Logistic Regression (Scratch - Gradient Descent) ---")
    scratch_model = BinaryLogisticRegressionScratch(learning_rate=0.1, num_iterations=2000, verbose=True)
    scratch_model.fit(X_train_scaled, y_train.values)
    
    # 5. Train Scikit-Learn Logistic Regression Model
    print("\n--- Training Scikit-Learn LogisticRegression ---")
    sklearn_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000, random_state=SEED)
    sklearn_model.fit(X_train_scaled, y_train.values)
    
    # 6. Model Evaluation
    y_prob_scratch = scratch_model.predict_proba(X_test_scaled)
    y_prob_sklearn = sklearn_model.predict_proba(X_test_scaled)[:, 1]
    
    metrics_scratch = calculate_metrics(y_test, y_prob_scratch, threshold=0.5)
    metrics_sklearn = calculate_metrics(y_test, y_prob_sklearn, threshold=0.5)
    
    metrics_scratch["Model"] = "Scratch Logistic Regression"
    metrics_sklearn["Model"] = "Scikit-Learn LogisticRegression"
    
    df_comparison = pd.DataFrame([metrics_scratch, metrics_sklearn])
    cols_display = ["Model", "Accuracy", "Precision", "Recall (Sensitivity)", "Specificity", "F1-Score", "ROC-AUC", "Log Loss", "TP", "TN", "FP", "FN"]
    df_comparison = df_comparison[cols_display]
    
    print("\n[Performance Metrics Comparison - Scratch vs Scikit-Learn]")
    print(df_comparison.to_string(index=False))
    
    # 7. Cross-Validation Stability Check
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(sklearn_model, scaler.fit_transform(X), y, cv=skf, scoring='accuracy')
    print(f"\n5-Fold Stratified Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # 8. Decision Threshold Sensitivity Analysis
    thresholds = np.linspace(0.1, 0.9, 9)
    thresh_results = []
    for t in thresholds:
        m = calculate_metrics(y_test, y_prob_scratch, threshold=t)
        m["Threshold"] = round(t, 2)
        thresh_results.append(m)
    df_thresh = pd.DataFrame(thresh_results)[["Threshold", "Accuracy", "Precision", "Recall (Sensitivity)", "Specificity", "F1-Score", "FP", "FN"]]
    print("\n[Decision Threshold Analysis (Scratch Model)]")
    print(df_thresh.to_string(index=False))
    
    # --- VISUALIZATIONS ---
    # Fig 1: Loss Convergence Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scratch_model.loss_history, color='#1f77b4', lw=2, label='Binary Cross-Entropy Loss')
    ax.set_title('Gradient Descent Loss Convergence (Scratch Model)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Iteration / Epoch', fontsize=11)
    ax.set_ylabel('Log Loss', fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "loss_convergence.png"), dpi=300)
    plt.close()
    
    # Fig 2: Confusion Matrices Side-by-Side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_scratch = confusion_matrix(y_test, (y_prob_scratch >= 0.5).astype(int))
    cm_sklearn = confusion_matrix(y_test, (y_prob_sklearn >= 0.5).astype(int))
    
    sns.heatmap(cm_scratch, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Malignant (0)', 'Benign (1)'],
                yticklabels=['Malignant (0)', 'Benign (1)'])
    axes[0].set_title(f"Scratch Logistic Regression\nAccuracy: {metrics_scratch['Accuracy']:.4f}", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    
    sns.heatmap(cm_sklearn, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Malignant (0)', 'Benign (1)'],
                yticklabels=['Malignant (0)', 'Benign (1)'])
    axes[1].set_title(f"Scikit-Learn Logistic Regression\nAccuracy: {metrics_sklearn['Accuracy']:.4f}", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    
    plt.suptitle("Breast Cancer Identification: Confusion Matrices", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrices.png"), dpi=300)
    plt.close()
    
    # Fig 3: ROC and Precision-Recall Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    fpr_sc, tpr_sc, _ = roc_curve(y_test, y_prob_scratch)
    fpr_sk, tpr_sk, _ = roc_curve(y_test, y_prob_sklearn)
    
    axes[0].plot(fpr_sc, tpr_sc, color='#1f77b4', lw=2, label=f"Scratch (AUC = {metrics_scratch['ROC-AUC']:.4f})")
    axes[0].plot(fpr_sk, tpr_sk, color='#2ca02c', lw=2, linestyle='--', label=f"Scikit-Learn (AUC = {metrics_sklearn['ROC-AUC']:.4f})")
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random Chance')
    axes[0].set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('False Positive Rate (1 - Specificity)')
    axes[0].set_ylabel('True Positive Rate (Recall / Sensitivity)')
    axes[0].legend(loc='lower right')
    
    prec_sc, rec_sc, _ = precision_recall_curve(y_test, y_prob_scratch)
    prec_sk, rec_sk, _ = precision_recall_curve(y_test, y_prob_sklearn)
    
    axes[1].plot(rec_sc, prec_sc, color='#1f77b4', lw=2, label='Scratch Model')
    axes[1].plot(rec_sk, prec_sk, color='#2ca02c', lw=2, linestyle='--', label='Scikit-Learn Model')
    axes[1].set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Recall (Sensitivity)')
    axes[1].set_ylabel('Precision')
    axes[1].legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "roc_and_pr_curves.png"), dpi=300)
    plt.close()
    
    # Fig 4: Decision Threshold Trade-off
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_thresh["Threshold"], df_thresh["Accuracy"], marker='o', label='Accuracy', color='#1f77b4')
    ax.plot(df_thresh["Threshold"], df_thresh["Precision"], marker='s', label='Precision', color='#2ca02c')
    ax.plot(df_thresh["Threshold"], df_thresh["Recall (Sensitivity)"], marker='^', label='Recall (Sensitivity)', color='#d62728')
    ax.plot(df_thresh["Threshold"], df_thresh["F1-Score"], marker='d', label='F1-Score', color='#9467bd')
    
    ax.set_title('Effect of Decision Threshold on Classification Metrics', fontsize=13, fontweight='bold')
    ax.set_xlabel('Decision Probability Threshold', fontsize=11)
    ax.set_ylabel('Metric Score', fontsize=11)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1, label='Default Threshold (0.5)')
    ax.legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "threshold_analysis.png"), dpi=300)
    plt.close()
    
    # Fig 5: Learned Feature Weights (Top 10 Positive & Top 10 Negative)
    fig, ax = plt.subplots(figsize=(10, 6))
    weights = pd.Series(scratch_model.weights, index=feature_names).sort_values()
    top_bottom_weights = pd.concat([weights.head(8), weights.tail(8)])
    
    colors = ['#d62728' if w < 0 else '#1f77b4' for w in top_bottom_weights]
    top_bottom_weights.plot(kind='barh', ax=ax, color=colors)
    ax.set_title('Top Feature Importance (Learned Weights in Scratch Model)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Standardized Weight Coefficient', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=300)
    plt.close()
    
    print(f"\nExperiment 5 completed successfully! All figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
