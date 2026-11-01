# Employee Attrition & Workplace Culture Analytics 🏢📊

An end-to-end Machine Learning project to predict and analyze employee attrition using the **IBM HR Analytics Employee Attrition & Performance** dataset. This project implements a robust preprocessing and modeling pipeline using Python and Scikit-Learn.

---

## 🚀 Project Overview
Employee turnover is a critical challenge for organizations. This project aims to identify the key factors driving employees to leave the company and build a predictive classification model using a **Random Forest Classifier** to flag high-risk employees early.

---

## 🛠️ Project Workflow & Pipeline

1. **Data Loading & Inspection:**
   - Loads the dataset and inspects shapes, data types, missing values, and duplicates.
2. **Exploratory Data Analysis (EDA):**
   - Visualizes the class distribution of the target variable (`Attrition`) using a pie chart.
3. **Data Preprocessing & Feature Engineering:**
   - Drops redundant columns (`EmployeeCount`, `EmployeeNumber`, `Over18`, `StandardHours`).
   - Encodes the target variable (`Attrition`: Yes/No $\rightarrow$ 1/0).
   - Splits data into training and testing sets (80/20 split).
   - Separates numerical and categorical features automatically.
4. **Scikit-Learn Pipelines (`ColumnTransformer`):**
   - **Numerical Pipeline:** Imputes missing values with the `median` and scales features using `StandardScaler`.
   - **Categorical Pipeline:** Imputes missing values with the `most_frequent` value and encodes categories using `OneHotEncoder` (ignoring unknown categories).
5. **Model Training:**
   - Trains a `RandomForestClassifier` (with 300 estimators and balanced class weights to handle class imbalance).
6. **Model Evaluation:**
   - Evaluates performance using **Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Classification Report**.

---

## 📦 Requirements & Dependencies

Make sure you have the following libraries installed before running the code:

```bash
pip install pandas matplotlib scikit-learn