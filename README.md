# 💳 Credit Card Fraud Detection System

An end-to-end Machine Learning project to detect fraudulent credit card transactions using an highly imbalanced dataset. This project covers comprehensive Exploratory Data Analysis (EDA), handling class imbalance using SMOTE, rigorous data scaling to prevent data leakage, model training, threshold tuning, and advanced evaluation metrics.

---

## 📊 Dataset Overview
- **Source:** Kaggle (Credit Card Fraud Detection Dataset)
- **Features:** 
  - `V1` to `V28`: Principal Component Analysis (PCA) transformed numerical features for user privacy.
  - `Amount`: Transaction amount.
  - `Class`: The target variable (`0` for normal transactions, `1` for fraudulent transactions).
- **Challenge:** The dataset is extremely imbalanced, where fraudulent transactions account for less than $0.2\%$ of the total data.

---

## 🛠️ Project Pipeline & Methodology

1. **Exploratory Data Analysis (EDA):**
   - Checked for missing values and duplicates.
   - Visualized class distribution showing severe class imbalance.
   - Analyzed transaction amounts across classes using log-scale boxplots.

2. **Data Preprocessing & Splitting:**
   - Dropped the `Time` column and separated features ($X$) from the target ($y$).
   - Performed an 80-20 train-test split using `stratify=y` to preserve class proportions.

3. **Feature Scaling:**
   - Applied **`RobustScaler`** specifically to the `Amount` feature to handle extreme outliers safely without dropping valuable transactions. 
   - *Note: `fit_transform` was strictly used on the training set only, and `transform` on the test set to prevent Data Leakage.*

4. **Handling Imbalanced Data:**
   - Used **Synthetic Minority Over-sampling Technique (SMOTE)** exclusively on the training set (`X_train`) to balance the class distribution.

5. **Model Training:**
   - Trained a **Logistic Regression** classifier with L2 regularization ($C = 10.0$).

6. **Evaluation & Threshold Tuning:**
   - Evaluated the model using **Accuracy, Precision, Recall, ROC-AUC, PR-AUC**, and **Confusion Matrix**.
   - Implemented custom **Threshold Tuning** (testing thresholds like $0.5$ and $0.9$) to optimize the trade-off between minimizing False Positives (customer friction) and maximizing Recall (security).

---

## 📈 Key Results & Performance

* **ROC-AUC Score:** $\approx 0.9778$
* **PR-AUC (Average Precision) Score:** $\approx 0.7457$
* **At Default Threshold ($0.5$):** Achieved a high **Recall of $91.84\%$** to capture the vast majority of frauds.
* **At Optimized Threshold ($0.9$):** 
  - **Precision** jumped significantly to **$46.41\%$** (drastically reducing false alarms/customer blockages).
  - **Recall** remained strong at **$85.71\%$**.
  - **False Positives** dropped from $543$ down to just $97$.

---

## 📂 Code Structure

```text
├── creditcard.csv                 # Dataset (downloaded from Kaggle)
├── credit_card_fraud.ipynb          # Main script containing EDA, preprocessing, and modeling
├── README.md                      # Project documentation