# 🚲 Bike Sharing Time-Series Forecasting 

A robust, end-to-end machine learning model built to forecast hourly bike rental demand using the **UCI Bike Sharing Dataset**. This project leverages advanced feature engineering techniques (time-based extraction, lag features, rolling statistics) and state-of-the-art gradient boosting frameworks (**XGBoost** and **LightGBM**) to achieve high predictive accuracy.

---

## 📊 Project Workflow & Architecture

1. **Exploratory Data Analysis (EDA):** 
   - Visualized seasonal, monthly, hourly, and weather-based trends impacting bike rentals.
   - Identified peak commuting hours and environmental dependencies (temperature, humidity, and wind speed).

2. **Feature Engineering & Preprocessing:**
   - **Time Features:** Extracted day of week indicators and created specialized weekend flags (`is_weekend`).
   - **Lag Features:** Engineered historical demand markers (`cnt_lag_1` and `cnt_lag_24`) to capture immediate past behavior and daily seasonality.
   - **Rolling Statistics:** Computed moving averages (`cnt_rolling_mean_3h` and `cnt_rolling_mean_24h`) to smooth out short- and long-term fluctuations.
   - **Target Transformation:** Applied log transformation (`np.log1p`) to the target variable to stabilize variance, while ensuring strict chronological sorting to prevent data leakage.

3. **Modeling & Evaluation:**
   - Trained `XGBoostRegressor` and `LGBMRegressor` on the processed dataset.
   - Evaluated models on inverse-transformed predictions (`np.expm1`) using rigorous metrics: **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)**.

---

## 📈 Model Performance Results

Both models delivered exceptional predictive performance on the test dataset after reversing the logarithmic transformation:

| Model | RMSE (Root Mean Squared Error) | MAE (Mean Absolute Error) | Performance Summary |
| :--- | :---: | :---: | :--- |
| **XGBoost** | **33.47** | **20.87** | Outperformed LightGBM slightly, achieving a lower error margin in predicting hourly rental counts. |
| **LightGBM** | **35.70** | **22.05** | Delivered strong baseline performance with highly efficient training times. |

---

## 🔍 Feature Importance & Key Insights

Analyzing the feature importance via **Gain** highlighted the core drivers behind the models' predictions:

1. **`cnt_lag_1` (Previous Hour Demand):** Stood out as by far the most dominant predictor (~72.5 Gain score), confirming that immediate past demand is the strongest indicator of future volume.
2. **`hr` (Hour of the Day) & `workingday`:** Acted as crucial temporal indicators reflecting commuter rush hours and work-week routines.
3. **`cnt_lag_24` (Same Hour Previous Day):** Effectively captured daily cyclical patterns.
4. **Weather & Environmental Factors:** Provided secondary adjustments based on humidity, temperature, and weather conditions.

---

## 🚀 Conclusion
This pipeline successfully demonstrates a production-ready workflow for time-series forecasting, highlighting how combining lagged historical data with gradient boosting models yields reliable, high-precision results for urban mobility analytics.