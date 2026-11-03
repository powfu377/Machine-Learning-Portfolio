# 📰 Fake News Detector Pipeline

An end-to-end Natural Language Processing (NLP) pipeline that classifies news articles into **Real** or **Fake** using **Multinomial Naive Bayes** and **TF-IDF Vectorization**.

---

## 📌 Project Overview

This project builds a Machine Learning workflow designed to identify misinformation in news articles. By combining title and text data, preprocessing the content, and leveraging a Scikit-Learn `Pipeline` with `TfidfVectorizer` and `MultinomialNB`, the model achieves **95% overall accuracy**.

---

## 📊 Dataset

The model utilizes the Kaggle **Fake and Real News Dataset**:
* **`True.csv`**: Contains verified authentic news stories.
* **`Fake.csv`**: Contains fabricated/fake news articles.

### Features Used
* **`title`**: Article title.
* **`text`**: Main body text of the article.
* **`content`**: Feature engineering combining `title + " " + text`.
* **`label`**: Target variable (`1` for Real News, `0` for Fake News).

---

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.x
* **Data Processing:** `pandas`, `numpy`
* **Text Processing:** `string`, `re`
* **Machine Learning:** `scikit-learn`
  * `train_test_split`
  * `TfidfVectorizer`
  * `MultinomialNB`
  * `Pipeline`
  * `classification_report`, `confusion_matrix`

---

## ⚙️ Methodology & Pipeline Steps

1. **Data Concatenation & Labeling:**
   * Loaded `True.csv` (labeled as `1`) and `Fake.csv` (labeled as `0`).
   * Concatenated datasets into a unified pandas DataFrame.
2. **Feature Engineering:**
   * Merged `title` and `text` into a single `content` feature to maximize text signal.
3. **Train-Test Split:**
   * 80% Training Data / 20% Testing Data.
4. **Text Preprocessing:**
   * Converted text to lowercase.
   * Stripped all punctuation marks.
5. **Vectorization & Classification Pipeline:**
   * **`TfidfVectorizer`**: Extracted top 5,000 features with N-Gram range `(1, 2)` (Unigrams + Bigrams).
   * **`MultinomialNB`**: Trained Multinomial Naive Bayes classifier with smoothing parameter `alpha=0.1`.

---

## 📈 Model Performance & Evaluation

The model achieved high precision, recall, and F1-scores across both classes on the holdout test set (8,980 samples).

### Classification Report

```text
              precision    recall  f1-score   support

    Fake (0)       0.96      0.95      0.95      4650
    Real (1)       0.94      0.95      0.95      4330

    accuracy                           0.95      8980
   macro avg       0.95      0.95      0.95      8980
weighted avg       0.95      0.95      0.95      8980
```

### Confusion Matrix

$$\begin{pmatrix} 4408 & 242 \\ 197 & 4133 \end{pmatrix}$$

* **True Negatives (Correctly predicted Fake):** 4,408
* **False Positives (Fake misclassified as Real):** 242
* **False Negatives (Real misclassified as Fake):** 197
* **True Positives (Correctly predicted Real):** 4,133

---

## 💻 Sample Predictions

Running custom unseen news headlines through the trained pipeline:

```text
The news: Breaking news: scientists discover a brand new planet in our... 
 --> Predicted: fake

The news: SHOCKING: Aliens landed in downtown Washington and took over... 
 --> Predicted: fake
```

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/fake-news-detector.git
   cd fake-news-detector
   ```

2. **Install required dependencies:**
   ```bash
   pip install pandas scikit-learn
   ```

3. **Prepare Datasets:**
   Ensure `True.csv` and `Fake.csv` are placed in the root directory.

4. **Execute the script:**
   ```bash
   python fake_news_detector.py
   ```

---

## 🎯 Key Takeaways & Future Enhancements

* Combining title and body text significantly improved class distinguishability.
* Bigram extraction `(1, 2)` helped capture contextual phrase patterns common in sensationalized fake news.
* **Future Work:** Integrate NLTK Stemming/Lemmatization, explore Linear SVM / Logistic Regression for comparison, and build a Streamlit web app interface.