# House Price Prediction & Neighborhood Economic Impact Analysis 🏡📊

An end-to-end Machine Learning project to predict house prices in King County, USA, with a heavy focus on exploratory data analysis (EDA), geographic feature engineering, and regularized linear regression models (Ridge and Lasso).

## 🚀 Project Overview
The goal of this project is not just to predict house prices accurately, but also to analyze how neighborhood characteristics (such as location, geographical coordinates, and economic factors represented by zipcodes) impact overall property values.

## 🗂️ Dataset
- **Source:** King County House Sales Dataset (Kaggle).
- **Features:** Includes 21 variables such as living area (`sqft_living`), grade, year built, geographical coordinates (`lat`, `long`), and zipcodes.

## 🛠️ Project Workflow & Steps
1. **Exploratory Data Analysis (EDA):**
   - Examined the data distribution and handled right-skewness in prices using **Log Transformation (`np.log1p`)**.
   - Created **Correlation Heatmaps** to identify key relationships between features (e.g., `sqft_living`, `grade`, `lat`) and the target variable.
   - Removed extreme **Outliers** using the IQR (Interquartile Range) method to stabilize model training.
2. **Feature Engineering & Preprocessing:**
   - Extracted features and handled dates.
   - Applied **Target Encoding** on the `zipcode` column to capture the economic impact of different neighborhoods without blowing up dimensionality.
   - Performed **Feature Scaling (Standardization)** to ensure all features contribute equally to regularized models.
3. **Model Building & Regularization:**
   - **Linear Regression (Baseline):** Trained a standard linear model to establish baseline metrics.
   - **Ridge Regression ($L2$):** Used `RidgeCV` to handle multicollinearity and optimize the regularization strength ($\alpha$).
   - **Lasso Regression ($L1$):** Used `LassoCV` for automatic feature selection by shrinking coefficients of less important features to zero.
4. **Model Evaluation:**
   - Compared models using metrics such as **$R^2$ Score** and **RMSE**.

## 📊 Results & Model Comparison
*(You can add a brief summary or table of your R² and RMSE results here after running the final code!)*

## ⚙️ Technologies Used
- Python 🐍
- Pandas & NumPy (Data Manipulation)
- Scikit-Learn (Machine Learning Models & Preprocessing)
- Matplotlib & Seaborn (Data Visualization)
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
