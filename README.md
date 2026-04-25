# Financial Risk Prediction using Machine Learning

## Overview

This project focuses on predicting **financial risk (IsUnderRisk)** using machine learning models. The goal is to identify high-risk entities based on audit, financial, and historical indicators.

The project follows a complete data science workflow:

* Exploratory Data Analysis (EDA)
* Feature Engineering (validated via ablation study)
* Model Benchmarking (multiple algorithms with cross-validation)
* Performance Evaluation using classification metrics and confusion matrix

---

## Problem Statement

Financial institutions must identify risky cases early to prevent losses. This project aims to build a model that can:

* Detect **high-risk cases accurately**
* Minimize **false negatives (missed risk)**
* Maintain **high precision to avoid false alarms**

---

## Dataset

* Source: https://www.kaggle.com/datasets/manukulamkombil/machinehack-financial-risk-prediction
* Total records: ~543
* Target variable: **IsUnderRisk (0 = No Risk, 1 = Risk)**

### Features

* Location_Score
* Internal_Audit_Score
* External_Audit_Score
* Fin_Score
* Loss_score
* Past_Results
* City

---

## Exploratory Data Analysis (EDA)

Key findings:

* **Location_Score** shows strong negative correlation with risk
* **Audit scores (Internal & External)** are strong positive indicators
* **Past_Results contains extreme values**, which act as **risk signals (not noise)**
* The dataset exhibits **non-linear relationships**, making tree-based models suitable

---

## Feature Engineering

Feature engineering was performed and **validated using ablation study**.

Tested features include:

* audit_total
* loss_to_fin
* interaction features
* binary threshold flags

### Result:

* Most engineered features **did not improve performance**
* Only **loss_to_fin** provided a small improvement
* Tree-based models already capture:

  * Non-linear relationships
  * Feature interactions

### Conclusion:

Minimal feature engineering is optimal.

---

## Modeling Approach

### Models Evaluated:

* Logistic Regression
* Random Forest
* Gradient Boosting
* AdaBoost
* Decision Tree
* SVM
* KNN
* XGBoost

### Evaluation Method:

* **Stratified 5-Fold Cross Validation**
* Metrics:

  * Accuracy
  * Precision
  * Recall
  * F1 Score
  * ROC AUC

---

## Model Performance

| Model             | Accuracy | Precision | Recall | F1         | AUC        |
| ----------------- | -------- | --------- | ------ | ---------- | ---------- |
| Gradient Boosting | 0.8747   | 0.9257    | 0.8706 | **0.8964** | **0.9355** |
| AdaBoost          | 0.8582   | 0.9153    | 0.8529 | 0.8826     | 0.9319     |
| Random Forest     | 0.8527   | 0.9210    | 0.8382 | 0.8761     | 0.9312     |
| XGBoost           | 0.8471   | 0.8890    | 0.8647 | 0.8755     | 0.9245     |

### Best Model: Gradient Boosting

---

## Confusion Matrix Analysis

|              | Predicted 0 | Predicted 1 |
| ------------ | ----------- | ----------- |
| **Actual 0** | 179         | 24          |
| **Actual 1** | 44          | 296         |

### Insights:

* High **precision (0.925)** → low false positives
* Good **recall (0.87)** → most risk cases detected
* **44 false negatives** → missed risk cases (critical)

---

## Key Insights

* Ensemble models outperform linear models
* Feature engineering had **limited impact**
* The problem is **non-linear**
* The model is strong but can be improved via:

  * Hyperparameter tuning
  * Threshold optimization

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Matplotlib, Seaborn

---

## Project Structure

```
.
├── banking-fraud-detection-complete.ipynb
├── README.md
```

---

## How to Run

1. Clone repository:

```
git clone https://github.com/kbdp1305/your-repo-name.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run notebook:

```
jupyter notebook banking-fraud-detection-complete.ipynb
```

---

## Future Improvements

* Hyperparameter tuning (Optuna / GridSearch)
* Threshold optimization for better recall
* SHAP explainability for model interpretation
* Deploy as API or dashboard

---

## Author

**Krisna Bayu Dharma Putra**
Email: [dharma.work.dev@gmail.com](mailto:dharma.work.dev@gmail.com)
GitHub: https://github.com/kbdp1305

---

## Acknowledgment

Dataset provided by Manu Mathew via Kaggle:
https://www.kaggle.com/datasets/manukulamkombil/machinehack-financial-risk-prediction

---
