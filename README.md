# 🗂️ Complaint Ticket Classification System

## 📌 Project Overview
This project builds a machine learning model to classify consumer complaints into product categories (e.g., Credit Card, Mortgage) using a JSON dataset of 78,313 complaint records. The goal is to automate complaint categorization for efficient customer service. A tuned Random Forest classifier achieves high performance.

## 💡 Problem Statement
Manual complaint categorization is time-consuming. This project automates the process by predicting product categories from text and categorical features.

## 🧠 Methodology

### 1. Data Overview
- **Rows**: 78,313
- **Features**: 22 columns (21 categorical, 1 numerical, mostly text-based)
- **Target**: `_source.product`

### 2. Data Preprocessing
- **Text Processing**: Cleaned text fields (`_source.complaint_what_happened`, `_source.issue`, `_source.sub_issue`) using NLTK and SpaCy, applied TF-IDF vectorization.
- **Categorical Encoding**: Used `LabelEncoder` for `_source.state`, `_source.sub_product`; `OneHotEncoder` for `_source.company_response`, `_source.submitted_via`, `_source.timely`.
- **Missing Values**: Imputed `_source.state`, `_source.sub_product` with mode; dropped columns with excessive missing data.
- **Feature Selection**: Removed non-predictive columns (e.g., `_index`, `_score`) and unique identifiers.

### 3. Model Building
- Evaluated classifiers (Logistic Regression, Naive Bayes, SVM, Decision Tree, Random Forest, XGBoost, LightGBM) with 5-fold cross-validation.
- Tuned Random Forest using `RandomizedSearchCV` (e.g., `n_estimators`, `max_depth`).

### 4. Evaluation
- NOTE: Results are based on a subset of the first 500 rows of the dataset due to computational constraints.
- **Best Model**: Random Forest
- **Results**:
  - Test Accuracy: 93.55%
  - Test Precision: 95.16%
  - Test Recall: 93.55%
  - Test F1-Score: 93.77%
- Saved model: `rf_complaint_ticket_classification.joblib`

## 📈 Results
The Random Forest model achieves 93.55% accuracy, effectively classifying complaints into product categories.

## 📦 Future Improvements
- Address potential class imbalance with undersampling/oversampling.
- Enhance feature engineering (e.g., sentiment analysis).
- Perform more rigorous hyperparameter tuning.
