# 💳 Credit Card Fraud Detection with Streamlit Dashboard

This project is a machine learning pipeline to detect credit card fraud using a variety of models. It features a **Streamlit dashboard** that allows:

- 🚀 Live prediction on single transactions
- 📊 Evaluation of 4 ML models on both imbalanced and balanced datasets
- 🔍 Visual comparison via confusion matrices

---

## 📁 Dataset

We use the well-known [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), containing anonymized features `V1`–`V28` from PCA, as well as `Time` and `Amount`.

---

## ⚙️ Features

- **ML Models Trained**:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Neural Network (Keras)

- **Data Handling**:
  - Imbalanced vs Balanced sampling
  - Feature scaling (`Time`: `MinMaxScaler`, `Amount`: `RobustScaler`)
  - Custom feature engineering pipeline

- **UI via Streamlit**:
  - Interactive form for entering transactions
  - Option to paste full feature array
  - Model evaluation and comparison dashboard

---

## 🖼️ Screenshots

### 🌐 Full App View
![Full App Screenshot](<img width="1440" alt="Screenshot 2025-07-07 at 2 49 34 PM" src="https://github.com/user-attachments/assets/16bc9b7c-1eef-41a5-940d-88bc7dade9fa" />)

### 🔮 Prediction Output
![Prediction Output](screenshots/prediction_view.png)

> Make sure to place your screenshots in a `screenshots/` folder in your repo.

---

## 🧠 Model Selection Logic

The best-performing model (by F1-score) is automatically chosen for predicting the fraud status of a single transaction. You can change the sampling strategy (`imbalanced` or `balanced`) to see how it affects performance.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
