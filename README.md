# Cardiovascular Disease Prediction Using Machine Learning 🩺

This project focuses on predicting the presence of cardiovascular disease (CVD) using clinical and demographic data. Cardiovascular disease remains a leading cause of death globally, and early detection is vital in improving patient outcomes. By building predictive models using structured health data, this project demonstrates the practical application of data science in the healthcare sector. The final model achieves strong classification performance, showcasing the potential for aiding medical decision-making.

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset Overview](#-dataset-overview)
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [Data Preprocessing](#-data-preprocessing)
- [Modeling Approach](#-modeling-approach)
- [Results and Evaluation](#-results-and-evaluation)
- [Interpretation and Feature Importance](#-interpretation-and-feature-importance)
- [Conclusion and Next Steps](#-conclusion-and-next-steps)
- [Skills Demonstrated](#️-skills-demonstrated)

## ❓ Problem Statement
Cardiovascular diseases are a group of disorders of the heart and blood vessels, often resulting in heart attacks or strokes. Early identification of at-risk individuals can significantly reduce mortality and morbidity. However, manual screening is time-consuming and costly. This project addresses the problem of efficiently predicting the presence of cardiovascular disease based on patient data, thereby supporting preventative healthcare efforts.

## 📊 Dataset Overview
*   **Source:** [Kaggle – Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
*   **Rows:** 70,000 records
*   **Columns:** 13 features (plus target)
*   **Target Variable:** `cardio` (1 = presence of cardiovascular disease, 0 = absence)
*   **Features:**
    *   Demographic: `age`, `gender`, `height`, `weight`
    *   Lifestyle: `smoking`, `alcohol intake`, `physical activity`
    *   Clinical: `blood pressure (systolic/diastolic)`, `cholesterol`, `glucose`
*   **Issues Identified:**
    *   Outliers in height and weight
    *   Imbalanced class distribution (~65% positive class)

## 🔎 Exploratory Data Analysis (EDA)
Key findings from the exploratory analysis:
*   **Correlation:** Strong positive correlation between systolic/diastolic BP and CVD.
*   **Distribution:** Cholesterol and glucose levels skewed toward unhealthy levels in CVD-positive patients.
*   **Class Balance:** The dataset is moderately imbalanced; more positive cases than negative.
*   **Visuals Used:**
    *   Histograms for feature distributions (e.g., age, BMI)
    *   Boxplots to visualize outliers in numerical features
    *   Heatmaps for correlation between features

## 🧹 Data Preprocessing
Several preprocessing steps were applied:
*   **Feature Engineering:**
    *   Created a BMI feature from `weight` and `height`.
    *   Categorized `age` into bins.
*   **Outlier Removal:**
    *   Removed implausible `height`/`weight` records.
*   **Encoding:**
    *   One-hot encoding for categorical features (e.g., cholesterol, glucose levels).
*   **Scaling:**
    *   `StandardScaler` applied to continuous variables.
*   **Imbalance Handling:**
    *   Utilized `class_weight='balanced'` in models to address class imbalance.

## 🤖 Modeling Approach
Multiple classification models were trained and compared:
*   **Logistic Regression:** Simple, interpretable baseline.
*   **Random Forest:** Handles non-linear relationships well.
*   **XGBoost:** Advanced gradient boosting classifier.

**Model Selection Criteria:**
*   Balance between precision and recall.
*   Generalization performance via cross-validation.

**Hyperparameter Tuning:**
*   `GridSearchCV` used for optimizing Random Forest and XGBoost.
*   5-fold stratified cross-validation applied to prevent overfitting.

## 📈 Results and Evaluation

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.72     | 0.69      | 0.75   | 0.72     | 0.78    |
| Random Forest       | 0.74     | 0.72      | 0.76   | 0.74     | 0.81    |
| XGBoost             | 0.75     | 0.74      | 0.77   | 0.75     | 0.83    |

*   **Best Model:** XGBoost, with the highest AUC and F1 score.
*   Confusion Matrix and ROC Curve were plotted to visualize performance.

## 🧠 Interpretation and Feature Importance
Feature Importance from XGBoost highlights:
*   Age
*   Systolic blood pressure
*   BMI
*   Cholesterol level

These features align with known clinical risk factors for CVD. Although SHAP values were not applied, tree-based feature importances offer valuable insights.

## ✅ Conclusion and Next Steps
This project successfully built a machine learning model to predict cardiovascular disease with high accuracy and interpretability. The findings affirm the value of machine learning in clinical risk stratification.

**Next Steps:**
*   Apply SHAP for model explainability.
*   Evaluate model on unseen external data for generalizability.
*   Consider deployment using a web app (e.g., with Streamlit or Flask).

## 🛠️ Skills Demonstrated
*   **Exploratory Data Analysis (EDA):** Matplotlib, Seaborn
*   **Data Preprocessing:** Encoding, scaling, feature engineering
*   **Modeling:** Logistic Regression, Random Forest, XGBoost
*   **Evaluation:** Confusion matrix, ROC-AUC, precision, recall
*   **Tools:** Python, Pandas, NumPy, scikit-learn, XGBoost, Jupyter