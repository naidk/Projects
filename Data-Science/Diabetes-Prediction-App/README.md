# Data Science Projects

This folder contains projects related to **Data Science**.


# 🩺 Diabetes Risk Prediction

This project is focused on predicting diabetes risk using various Machine Learning models and a tuned Artificial Neural Network (ANN). The models are evaluated and compared using multiple performance metrics, and the best-performing model is saved for deployment.

## 📌 Problem Statement

Predict whether a person is likely to be diabetic based on medical attributes like Glucose, Blood Pressure, BMI, etc. This binary classification task uses the Pima Indians Diabetes dataset.

---

## 📊 Dataset

- **Source**: Pima Indians Diabetes Dataset
- **Target Variable**: `Outcome` (0 = Non-Diabetic, 1 = Diabetic)
- **Features**:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age

---

## ⚙️ Project Pipeline

1. **Data Cleaning & EDA**
   - Handled missing values
   - Outlier detection and treatment
   - Visualization of distributions

2. **Feature Engineering**
   - Binary & One-Hot Encoding
   - Feature Scaling using StandardScaler
   - Saved scaler as `scaler.pkl`

3. **Model Building & Tuning**
   - ✅ Models Tested:
     - LightGBM (Tuned)
     - KNN
     - Soft Voting Ensemble (LGBM + KNN)
     - ANN (Tuned with Optuna)
     - LazyPredict Benchmarking
   - ✅ Best Models:
     - ANN (F1: 0.9831, ROC AUC: 0.9996)
     - LightGBM + KNN Ensemble (F1: 0.8338, ROC AUC: 0.9305)

4. **Hyperparameter Tuning**
   - Optuna used for ANN tuning
   - RandomizedSearchCV used for LightGBM and VotingClassifier

5. **Model Evaluation**
   - Accuracy, Precision, Recall, F1 Score, ROC AUC
   - Visualizations: Confusion Matrix, ROC Curve, Discrimination Threshold

6. **Model Saving**
   - Saved best ANN model (`ann_model.h5`)
   - Saved scaler (`scaler.pkl`)

---

## ✅ Best Model Summary

| Model                  | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-----------------------|----------|-----------|--------|----------|----------|
| ANN (Tuned)           | 0.9883   | 0.9887    | 0.9776 | 0.9831   | 0.9996   |
| LightGBM + KNN Voting | 0.8867   | 0.8423    | 0.8283 | 0.8338   | 0.9305   |
| LightGBM (Tuned)      | 0.8711   | 0.8155    | 0.8075 | 0.8104   | 0.9431   |

---

## 📦 Deployment Ready

- ✅ ANN model saved: `ann_model.h5`
- ✅ Scaler saved: `scaler.pkl`
- FastAPI / Streamlit deployment support ready (optional)

---

## 🛠️ Requirements

```bash
pip install -r requirements.txt
