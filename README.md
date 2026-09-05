# Health Risk Prediction — Heart Disease & Breast Cancer (Machine Learning)

A machine learning project that predicts patient risk for **heart disease** and **breast cancer** using classification models trained on clinical datasets. This repository contains the full data analysis and modelling pipeline, from exploration through to exported, deployment-ready models.

> **My role:** I built the complete data analysis and machine learning pipeline (exploration, preprocessing, model training, evaluation, and model export). The trained models were later integrated into a web application built with Flask by a teammate.

## Overview

The goal was to build and compare classification models that flag patients at risk, using two separate medical datasets. For each dataset, two models were trained and evaluated head-to-head: **Logistic Regression** and **Random Forest**.

## Datasets

| Dataset | Records | Target | Source |
|---|---|---|---|
| Heart Disease | 303 | `target` (0 = no disease, 1 = disease) | UCI Heart Disease |
| Breast Cancer | 569 | `diagnosis` (B = benign, M = malignant) | Wisconsin Breast Cancer |

## Methodology

1. **Exploratory Data Analysis** — class balance, missing-value checks, distribution plots, and feature correlation heatmaps.
2. **Feature selection** — narrowed each dataset to the most relevant clinical attributes.
3. **Preprocessing** — feature scaling with `StandardScaler`; 80/20 train/test split.
4. **Model training** — Logistic Regression and Random Forest on each dataset.
5. **Evaluation** — accuracy, precision, recall, F1-score, and AUC-ROC, plus confusion matrices and ROC curves.
6. **Model export** — best models saved as `.pkl` files with `joblib` for downstream integration.

## Results

**Heart Disease**

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | **0.787** | 0.738 |
| Precision | 0.833 | 0.774 |
| Recall | 0.758 | 0.727 |
| F1-Score | 0.794 | 0.750 |
| AUC-ROC | **0.854** | 0.844 |

**Breast Cancer**

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | **0.930** | 0.912 |
| Precision | 0.886 | 0.900 |
| Recall | 0.929 | 0.857 |
| F1-Score | 0.907 | 0.878 |
| AUC-ROC | **0.985** | 0.985 |

Logistic Regression was the stronger performer on both datasets, with especially strong separation on the breast cancer data (AUC 0.985).

## Tech Stack

- **Python** — pandas, NumPy
- **scikit-learn** — LogisticRegression, RandomForestClassifier, model selection & metrics
- **Visualisation** — matplotlib, seaborn
- **Model persistence** — joblib
- **Environment** — Google Colab

## Repository Contents

- `01_heart_exploration.ipynb` — the full analysis and modelling notebook
- `requirements.txt` — Python dependencies

## How to Run

1. Open the notebook in Google Colab or Jupyter.
2. Install dependencies: `pip install -r requirements.txt`
3. Update the dataset paths to point to your copies of `heart.csv` and `cancer.csv`.
4. Run the cells top to bottom.

---

*Built as my final-year Master's project.*
