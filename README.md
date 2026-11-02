# Real Estate Location-Based Valuation & Similarity Search 🏠📍

A Machine Learning project using **K-Nearest Neighbors (KNN)** to predict California housing prices and analyze geospatial similarity patterns based on location and property attributes.

---

## 📌 Project Overview
Location is one of the most critical factors influencing real estate prices. This project aims to:
1. Conduct Spatial Exploratory Data Analysis (EDA) on California housing data.
2. Build an end-to-end Machine Learning pipeline using `KNeighborsRegressor`.
3. Account for geographical coordinates (`latitude`, `longitude`) and demographic features to accurately estimate home values.

---

## 📊 Dataset
The project utilizes the famous **California Housing Prices** dataset from Kaggle.
* **Target Variable:** `median_house_value`
* **Numerical Features:** `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`.
* **Categorical Feature:** `ocean_proximity`

---

## 🛠️ Data Preprocessing & Pipeline Construction
1. **Handling Missing Values:** Missing entries in `total_bedrooms` were imputed using the median value.
2. **Feature Engineering & Transformation:**
   * **StandardScaler:** Applied to numerical features to handle distance sensitivity in KNN.
   * **OneHotEncoder:** Applied to categorical features (`ocean_proximity`).
3. **Pipeline Integration:** Built a robust `Scikit-Learn Pipeline` chaining the preprocessing (`ColumnTransformer`) with a weighted KNN Regressor (`weights='distance'`).

---

## 📈 Model Performance & Results
The KNN Regressor with distance weighting achieved strong performance metrics on the test dataset:

| Metric | Result |
| :--- | :--- |
| **$R^2$ Score** | **0.7173** (71.73% of variance explained) |
| **RMSE** | **\$60,860.28** |
| **MSE** | **3,703,974,292.83** |

---

## 💻 How to Run the Code

### 1. Prerequisites
Make sure you have the following packages installed:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 2. Execution
Place the `housing.csv` dataset in the project directory and run the main python script or Jupyter notebook.

```bash
python main.py
```

---

## 🖼️ Visualizations Highlighted
* **Correlation Heatmap:** Identifies relationships between numeric attributes and price (`median_income` showing the highest correlation).
* **Geospatial Price Distribution:** A scatter plot mapping `longitude` and `latitude` against prices and population density to illustrate coastal price spikes.

---

## 🚀 Next Improvements
* Apply Hyperparameter Tuning (`GridSearchCV`) for optimal $K$ values and distance metrics.
* Extend the pipeline to include a **Similarity Search Engine** using `sklearn.neighbors.NearestNeighbors` to recommend similar properties.