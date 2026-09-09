# Machine Learning Lab (7th Semester)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the practical experiments, code implementations, Jupyter Notebooks, and documentation for the **Machine Learning Laboratory Course (7th Semester)**.

---

## 📂 Repository Structure

```text
ML-lab/
├── Lab1.ipynb              # Lab 1: Environment Setup & Introduction
├── ep-3/
│   └── lab-3.ipynb         # Lab 3: Exploratory Data Analysis (EDA) on Iris Dataset
├── exp-4/
│   └── lab-4.ipynb         # Lab 4: Simple Linear Regression on California Housing Dataset
├── exp-8/
│   ├── exp-8.ipynb         # Lab 8: Ridge & Lasso Regularized Linear Regression Notebook
│   └── regularized_regression.py # Executable Python benchmark script
├── verify_ml_env.py        # Automated environment & library verification script
├── .gitignore              # Git ignore rules for virtual environments & caches
└── README.md               # Repository documentation
```

---

## 🧪 Experiments Overview

| Experiment | Title | Dataset / Tool | Description & Key Focus |
| :--- | :--- | :--- | :--- |
| **Lab 1** | Machine Learning Setup | Python / Jupyter | Environment verification, library imports, and baseline sanity checks. |
| **Lab 3** | Exploratory Data Analysis | Iris Dataset | Data cleaning, statistical summary, pairplots, boxplots, and feature distribution analysis. |
| **Lab 4** | Simple Linear Regression | California Housing Dataset | Predicting house values using single feature OLS regression (`MedInc`), residual diagnostics, and evaluation. |
| **Lab 8** | Regularized Linear Regression | California Housing & Breast Cancer Wisconsin | Implementation & comparative evaluation of Ridge ($L_2$) and Lasso ($L_1$) regularized regression vs Standard Linear Regression (OLS). |

---

## 🔬 Lab 4 Detail: Simple Linear Regression

### **Objective**
Implement a **Simple Linear Regression** model to predict median house prices (`MedHouseVal`) in California districts using a single predictor feature, median income (`MedInc`).

### **Mathematical Model**
$$\text{MedHouseVal} = m \cdot \text{MedInc} + c$$

Parameters estimated via **Ordinary Least Squares (OLS)**:
- **Learned Slope ($m$)**: `0.4194`
- **Learned Intercept ($c$)**: `0.4501`

### **Performance Metrics**
- **Mean Absolute Error (MAE)**: `0.5332` (~$\$53,320$)
- **Root Mean Squared Error (RMSE)**: `0.8421` (~$\$84,210$)
- **$R^2$ Score (Test Set)**: `0.4735` (Explains **47.35%** of variance using single feature `MedInc`)

---

## 🔬 Lab 8 Detail: Ridge and Lasso Regularized Linear Regression

### **Objective**
Implement and compare **Ordinary Least Squares (OLS)**, **Ridge Regression ($L_2$)**, and **Lasso Regression ($L_1$)** across continuous (`California Housing`) and binary classification (`Breast Cancer Wisconsin`) datasets.

### **Mathematical Formulation**
- **OLS**: $\min_{\mathbf{w}} \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2$
- **Ridge ($L_2$)**: $\min_{\mathbf{w}} \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_2^2$ *(Smooth weight shrinkage)*
- **Lasso ($L_1$)**: $\min_{\mathbf{w}} \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_1$ *(Automatic feature selection / Sparsity)*

### **Summary of Results**
1. **California Housing Dataset ($R^2 \approx 0.5758$, Test MSE $\approx 0.5558$)**:
   - **Ridge ($\alpha = 24.38$)**: Smoothly shrinks collinear house room features without dropping predictors.
   - **Lasso ($\alpha = 0.0016$)**: Zeroes out redundant features while retaining top predictors (`MedInc`, `Latitude`, `Longitude`).
2. **Breast Cancer Wisconsin Dataset (Accuracy $\approx 97.37\%$, ROC-AUC $\approx 0.9934$)**:
   - **Lasso Regularization**: Eliminates 20 non-informative measurement features out of 30, building a sparse, highly interpretable 10-feature classifier with 97.37% accuracy.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https.github.com/sajidtecho/ML-lab.git
cd ML-lab
```

### 2. Set Up Virtual Environment (Optional but Recommended)
```bash
python -m venv .venv
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn jupyter notebook
```

### 4. Verify Machine Learning Environment
Run the included verification script to check your local installation:
```bash
python verify_ml_env.py
```

### 5. Launch Jupyter Notebooks
```bash
jupyter notebook
```
Navigate to `exp-4/lab-4.ipynb` to view or rerun the Simple Linear Regression experiment.

---

## 📊 Key Technologies & Libraries
- **Language**: Python 3.11+
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`
- **Visualization**: `matplotlib`, `seaborn`
- **Interactive Development**: Jupyter Notebook (`.ipynb`)

---

## 📜 License
This repository is maintained for academic and educational purposes.
