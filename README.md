# 🏠 House Prices

This repository contains my Machine Learning project developed for the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition on Kaggle. The goal is to predict house sale prices using a comprehensive dataset with numerous features.

> 📅 Project developed in June 2025 as part of my Artificial Intelligence journey.

---

## 🎯 Objective

The goal is build a regression model capable of predicting the sale price of residential homes. The dataset includes 79 explanatory variables describing different aspects of the properties. This challenge requires extensive data cleaning, feature engineering, model tuning and validation.

---

## 🧹 1. Data Preprocessing

All data preparation steps were performed in the `1_preprocessing` notebook. This stage focused on cleaning and transforming the raw dataset, while building a customized preprocessing pipeline. The process includes the steps:

1. **Load and Split the Training Data**
   - The original training dataset was loaded and split into training and test (validation) subsets.
2. **Check General Information**
   - Basic dataset inspection, including data types, missing values and statistical summaries.
3. **Handle Missing Values**
   - This section has two subsections to handle "false" and "true" missing values.
4. **Handle Outliers**
   - Identification and removal of extreme outliers that could negatively impact model performance.
5. **Feature Engineering**
   - Creation of new columns from existing ones.
6. **Columns Log Transformation**
   - Log transformation was applied to skewed numerical features to reduce their impact on the model.
7. **Columns Type Conversion**
   - Proper conversion of features to the appropriate data types.
8. **Columns Significance**
   - Analysis of features relevance using correlation and statistical technique.
9. **Encoder and Scaler**
   - Applied appropriate encoding for categorical variables and scaling of numerical data.
10. **Export Preprocessing Pipeline**
    - The final pipeline was exported for reuse in model training and prediction.

---

## 🤖 2. Machine Learning and Model Selection

This stage, detailed in the `2_machine_learning` notebook, focused on selecting and evaluating the best predictive models. The process involved:

1. **Import the DFs and Preprocess**  
    - Load the split datasets and the preprocessing pipeline.
2. **Feature Selection**  
    - `Recursive Feature Elimination with Cross-Validation` (RFECV) was used to reduce dimensionality and select the most significant features.
3. **Model Selection with Grid Search** 
    - Several regression models were tested, including:
        - Ridge
        - Lasso
        - ElasticNet
        - Random Forest
        - Gradient Boosting
        - XGBoost
        - Support Vector Regressor
        - CatBoost
        - Multi-Layer Perceptron
4. **Ensemble**  
   - The best models were combined using Voting and Stacking Regressor to improve predictive performance and robustness.
5. **Final Model Validation with Cross-Validation**
    - The final model was evaluated using cross-validation to ensure generalization and stability.

---

## 📤 3. Final Prediction and Submission

The `3_prediction` notebook is responsible for producing the final predictions and generating the CSV file for submission to Kaggle.

1. **Training the Final Model**  
    - The full training dataset was used to retrain the final model.
2. **Predicting**  
    - The retrained final model produced the predictions.
3. **Exporting the Results**
    - The predictions were exported to a CSV file.

---

## 📈 Results

The final model achieved the following performance:

- **Cross-validation RMSLE (on training set):** `0.11678 ± 0.02441`  
- **Kaggle public leaderboard score:** `0.12030`

---

## 🧠 Key Learnings

Throughout this project, I deepened my understanding of:

- Handling real-world datasets with many missing values and categorical variables
- Engineering meaningful features to improve model performance
- Choosing appropriate preprocessing strategies for different models
- Using GridSearchCV effectively for hyperparameter optimization
- Building and evaluating ensemble models using voting and stacking
- Ensuring model robustness through cross-validation

---

I'm very happy to apply what I learned about Machine Learning and Statistics in the first semester of 2025 in [Alura courses](https://www.alura.com.br/escola-data-science) and for having learned even more during the development of the project.

Thanks!