# 🛠️ Predictive Maintenance for Industrial Equipment using SVM & SMOTE

A machine learning project designed to predict industrial equipment failure prior to occurrence. Using the **AI4I 2020 Predictive Maintenance Dataset**, this repository demonstrates an end-to-end data pipeline utilizing **Support Vector Machines (SVM)**, physical sensor feature evaluation, automated feature scaling, and synthetic over-sampling (**SMOTE**) wrapped inside an **Imbalanced-Learn Pipeline (`ImbPipeline`)** to prevent data leakage.

---

## 📌 Executive Summary

Unexpected failures in manufacturing lines lead to costly downtime and maintenance expenses. This project builds a binary classification engine to forecast machine failures based on operating parameters such as temperatures, torque, tool wear, and rotational speed. 

Due to the extreme class imbalance inherent in equipment failure datasets (~3.4% failure rate), standard classification models tend to overfit the majority class (operational status). By combining **`StandardScaler`**, **`OneHotEncoder`**, **`SMOTE`**, and **`SVC(kernel='rbf')`** in a leak-proof pipeline, we significantly improve the model's sensitivity (**Recall**) to critical failure events.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.10+
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning & Pipeline:** `scikit-learn`
* **Class Imbalance Management:** `imbalanced-learn` (`imblearn`)

---

## 📊 Dataset Overview & Sensor Features

The project uses the **AI4I 2020 Predictive Maintenance Dataset** (synthetic dataset reflecting real-world industrial tool operations).

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| **`UDI`** | Identifier | Unique Device Identifier (Dropped during preprocessing) |
| **`Product ID`** | Identifier | Quality variant ID combining type and serial number (Dropped) |
| **`Type`** | Categorical | Product quality variant: **L** (Low - 50%), **M** (Medium - 30%), **H** (High - 20%) |
| **`Air temperature [K]`** | Continuous | Ambient environment temperature in Kelvin |
| **`Process temperature [K]`** | Continuous | Internal process operational temperature in Kelvin |
| **`Rotational speed [rpm]`** | Continuous | Spindle rotational speed calculated in RPM |
| **`Torque [Nm]`** | Continuous | Torque values measured in Newton-meters |
| **`Tool wear [min]`** | Continuous | Accumulated tool wear time in minutes |
| **`Machine failure`** | **Target** | Binary indicator: `0` = No Failure, `1` = Equipment Failure |

---

## ⚙️ Machine Learning Pipeline Architecture

The workflow implements strict pipeline isolation to prevent **Data Leakage** during feature transformation and over-sampling:

1. **Exploratory Data Analysis (EDA):**
   * Missing value check & Duplicate analysis (`0` NaNs, `0` Duplicates).
   * Feature correlation analysis via heatmaps.
   * Target class distribution assessment (Pie chart & value counts).
   * Multi-failure occurrence investigation.

2. **Data Preprocessing & Splitting:**
   * Exclusion of non-predictive identifiers (`UDI`, `Product ID`).
   * Stratified Train/Test split (`80/20` ratio, `random_state=42`).

3. **Pipeline Stages (`ImbPipeline`):**
   * **Numerical Scaling:** Applying `StandardScaler` to continuous physical variables (`Air temp`, `Process temp`, `Speed`, `Torque`, `Tool wear`).
   * **Categorical Encoding:** Applying `OneHotEncoder(handle_unknown='ignore')` to the `Type` feature.
   * **Class Resampling:** Executing Synthetic Minority Over-sampling Technique (**`SMOTE`**) strictly on `X_train` during model fitting.
   * **Classification:** Training a Support Vector Classifier (**`SVC`**) with radial basis function (`rbf` kernel), `C=1.0`, `gamma='scale'`, and `class_weight='balanced'`.

---

## 📈 Model Performance & Results

### Core Metrics

* **Accuracy:** `97.40%`
* **Precision (Class 1 - Failure):** `57.63%`
* **Recall (Class 1 - Failure):** `55.74%` *(Detected 34 out of 61 actual failures)*
* **F1-Score (Class 1 - Failure):** `56.67%`

### Confusion Matrix

$$\begin{bmatrix} 1914 & 25 \\ 27 & 34 \end{bmatrix}$$

* **True Negatives (TN):** 1914 (Correctly identified operational state)
* **False Positives (FP):** 25 (False alarms)
* **False Negatives (FN):** 27 (Missed failures)
* **True Positives (TP):** 34 (Successfully predicted failures)

### Classification Report

```text
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      1939
           1       0.58      0.56      0.57        61

    accuracy                           0.97      2000
   macro avg       0.78      0.77      0.78      2000
weighted avg       0.97      0.97      0.97      2000
```

---


## 🔮 Future Enhancements & Optimization

1. **Physical Feature Engineering:**
   * **`Temp Difference`**: Compute $T_{process} - T_{air}$ to explicitly capture heat dissipation capacity.
   * **`Power Requirement`**: Compute $Torque \times Rotational\_speed$ to model mechanical load stress.
   * **`Strain Index`**: Combine $Tool\_wear \times Torque$ to monitor tool strain accumulation.

2. **Hyperparameter Tuning (`GridSearchCV`):**
   * Systematic grid search over $C \in [0.1, 1, 10, 100]$ and $gamma \in [0.001, 0.01, 0.1, 1, 'scale']$.

3. **Decision Threshold Tuning:**
   * Adjust the classification decision threshold from the default `0.5` down to `0.2 - 0.3` to maximize **Recall** according to industrial safety tolerances.

---

## 📜 License & Acknowledgments

* **Dataset Source:** Stephan Matzka, *AI4I 2020 Predictive Maintenance Dataset*, [Kaggle Dataset Link](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020).
* Distributed under the **MIT License**.