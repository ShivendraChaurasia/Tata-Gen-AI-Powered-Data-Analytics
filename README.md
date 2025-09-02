# AI-Powered Credit Card Delinquency Prediction for Geldium Finance
This repository documents the process of developing an AI-powered solution to reduce credit card delinquency rates for Geldium Finance. This project was undertaken as part of a Tata iQ job simulation for an AI transformation consultant role.

# Project Overview
As an AI transformation consultant at Tata iQ, the primary objective is to leverage data-driven decision-making to solve challenges in the financial services sector. This project focuses on helping Geldium Finance, a financial services company, address its high credit card delinquency rate.

## Goal

The main goal is to analyze customer data to predict delinquency risks using advanced analytics. This involves:

*Performing exploratory data analysis (EDA) to uncover insights from the data.

* Building predictive models to identify at-risk customers.

* Developing a recommendation framework to guide intervention strategies for the Head of Collections.

* Ensuring all AI-driven solutions are ethical, explainable, and effective.

## Project Tasks

The project is broken down into the following key tasks:
1. EDA using GenAI: Conduct an exploratory analysis of the provided datasets using GenAI tools to summarize data, handle missing values, and identify risk profiles.

2. Predictive Modeling using GenAI: Apply common credit risk modeling techniques (e.g., decision trees, logistic regression, neural networks) and evaluate model performance with a focus on ethical considerations like bias and fairness.

3. Additional Refinement: Develop a structured recommendation framework with next steps, key takeaways, and business justifications.

4. Final Presentation: Design a presentation outlining the framework for an AI-powered collections system, including guardrails to ensure fair decision-making.

Task 1: EDA using GenAI
Exploratory data analysis (EDA) is the foundational step in understanding a dataset before developing predictive models. It helps uncover patterns, inconsistencies, and missing values to ensure data quality and reliability.

Why EDA Matters in Predicting Delinquency:
Ensures data integrity: Identifies missing values, duplicates, and inconsistencies.

Highlights patterns and anomalies: Helps detect trends in customer behavior.

Prevents biased models: Reduces the risk of unfair treatment by ensuring diverse data representation.

Supports better decision-making: Provides clear insights for proactive customer engagement.

Key Steps in Conducting EDA
Understanding the dataset: Identifying key variables, data types, and missing values.

Identifying missing values and outliers: Deciding whether to remove or impute missing data.

Understanding relationships between variables: Exploring correlations, such as between credit utilization and delinquency risk.

Detecting patterns and risk factors: Identifying behaviors associated with delinquency.

Review the Dataset using GEN -AI

The dataset was uploaded to a GenAI tool (e.g., Gemini, ChatGPT) to perform an initial analysis.
<img width="1296" height="591" alt="Screenshot 2025-08-29 183912" src="https://github.com/user-attachments/assets/cfc13e1e-f37b-40b5-bdd2-37a62f5817b4" />

The following prompts were used:
“Summarize key patterns, outliers, and missing values in this dataset. Highlight any fields that might present problems for modeling delinquency.”

“Identify the top 3 variables most likely to predict delinquency based on this dataset. Provide brief reasoning.”


After uploading use this or similar type of prompt,

“Summarize key patterns, outliers, and missing values in this dataset. Highlight any fields that might present problems for modeling delinquency.”
“Identify the top 3 variables most likely to predict delinquency based on this dataset. Provide brief reasoning.”

Result
<img width="1307" height="564" alt="Screenshot 2025-08-29 183803" src="https://github.com/user-attachments/assets/7afd0403-bf9e-47ba-a780-a37e7c782c26" />

<img width="1300" height="568" alt="Screenshot 2025-08-29 184102" src="https://github.com/user-attachments/assets/655d998e-f88b-4dab-984a-c68077b9d88c" />


Data Analysis Findings Report
The initial analysis produced the following key findings:

Notable Missing or Inconsistent Data
Missing Values:

Income: 7.8% missing (39 of 500 records).

Loan_Balance: 5.8% missing (29 of 500 records).

Credit_Score: 0.4% missing (2 of 500 records).

Inconsistent Data:

Employment_Status: This categorical feature had inconsistent entries (e.g., 'Employed', 'employed', 'EMP') that required standardization.

Key Anomalies
Credit_Utilization: Some records showed utilization slightly above 1.0, which may be data entry errors or extreme cases.

Skewed Distributions: Income, Credit_Utilization, and Loan_Balance were all right-skewed.

Delinquent_Account Imbalance: The target variable is severely imbalanced, with only 16% of accounts being delinquent. This requires special handling (e.g., oversampling, undersampling) to prevent model bias.

Early Indicators of Delinquency Risk:

Missed_Payments: This feature directly quantifies past payment behavior and is a strong early indicator. Higher values of missed payments are likely to correlate with increased delinquency risk.

Credit_Utilization: High credit utilization rates (closer to 1.0) generally indicate financial strain and are often strong predictors of future delinquency.

Credit_Score: Lower credit scores are typically associated with higher credit risk and thus serve as a key indicator of potential delinquency.

Debt_to_Income_Ratio: A higher debt-to-income ratio suggests a greater financial burden relative to income, which can be an early warning sign of difficulty in managing finances and potential delinquency.

Payment status trend from Month_1 to Month_6: The sequence of payment statuses ('On-time', 'Late', 'Missed') across these six months provides a temporal pattern of payment behavior. A trend of increasing 'Late' or 'Missed' payments over these months would be a strong early indicator of escalating delinquency risk.


Initial Data Quality Observations Summary

The initial assessment of the dataset reveals generally good data quality, with most features being complete and within expected ranges. However, the presence of missing values in Income and Loan_Balance requires careful handling, and the inconsistent capitalization in Employment_Status necessitates data cleaning. The most significant data quality challenge for modeling delinquency is the substantial class imbalance in the Delinquent_Account target variable, which must be addressed to build an effective predictive model. Overall, the dataset provides a solid foundation for building a delinquency prediction model, provided these identified issues are appropriately managed.

Address Missing Data and Data Quality Issues
The following prompts were used to determine the best approach for handling missing data:

“Suggest an imputation strategy for missing values in this dataset based on industry best practices.”

“Propose best-practice methods to handle missing credit utilization data for predictive modeling.”

Recommended Imputation Strategies:
For Credit_Score (0.4% missing): Median imputation is sufficient due to the low percentage of missing data.

For Income (7.8% missing) and Loan_Balance (5.8% missing):

Regression Imputation: Predicts missing values based on other features.

K-Nearest Neighbors (KNN) Imputation: Fills values based on the k most similar data points.

Multiple Imputation by Chained Equations (MICE): Considered the gold standard, this method creates multiple imputed datasets to account for uncertainty.

For a balance of effectiveness and simplicity, KNN Imputation or Regression Imputation are good starting points. For maximum statistical rigor, MICE is recommended.

Python Implementation

The data cleaning and imputation steps were implemented in a Python notebook.

My Notebook: [Uploading Geldium Data Cleaning.ipynb…]()


Detect Patterns and Risk Factors
The analysis identified several high-risk indicators for delinquency.

High-Risk Indicators for Delinquency
Missed_Payments: A higher number of missed payments directly increases delinquency risk.

Credit_Utilization: High utilization suggests financial strain and a greater likelihood of default.

Credit_Score: A lower score reflects a history of poor credit management.

Debt_to_Income_Ratio: A higher ratio indicates a greater financial burden relative to income.

"Late" or "Missed" Payment Status (Month_1 to Month_6): A trend of deteriorating payment behavior is a strong precursor to delinquency.

Insights Impacting Delinquency Prediction
The significant class imbalance in the target variable must be addressed to build an accurate model.

Skewed numerical distributions may require transformation (e.g., log transformation) for certain models.

The time-series potential of the Month_x variables can be leveraged by creating new features (e.g., number of late payments in the last 3 months).


Exploratory Data Analysis (EDA) Summary Report
1. Introduction
This report summarizes the EDA performed on the delinquency prediction dataset. The goal was to understand the data's structure, identify patterns, detect anomalies, and pinpoint potential risk indicators for predicting customer delinquency.

2. Dataset Overview
Number of records: 500

Key variables: Customer_ID, Age, Income, Credit_Score, Credit_Utilization, Missed_Payments, Delinquent_Account, Loan_Balance, Debt_to_Income_Ratio, Employment_Status, Account_Tenure, Credit_Card_Type, Location, and Month_1 - Month_6.

Data types: A mix of numerical (float64, int64) and categorical (object) data.

Anomalies:

Class Imbalance: 84% non-delinquent vs. 16% delinquent.

Inconsistent Categorical Entries: Employment_Status required standardization.

Potential Outliers: Credit_Utilization values slightly above 1.0.

Skewed Distributions: Income, Loan_Balance, and Credit_Utilization.

3. Missing Data Analysis
Variables with missing values:

Income: 7.8%

Loan_Balance: 5.8%

Credit_Score: 0.4%

Missing data treatment:

Credit_Score: Imputed with the median.

Income and Loan_Balance: Imputed with the median for simplicity during this initial phase.

4. Key Findings and Risk Indicators
The analysis confirmed that the class imbalance is a major challenge. The skewed numerical features may benefit from transformation. The Month_1 through Month_6 variables provide a crucial historical sequence of payment behavior. Strong correlations are expected between Delinquent_Account and variables like Credit_Score, Missed_Payments, and Credit_Utilization.

5. AI & GenAI Usage
Generative AI tools were used to accelerate the EDA process. Key prompts included:

"Summarize key patterns, outliers, and missing values in this dataset..."
"Suggest an imputation strategy for missing values..."
"List high-risk indicators..."
"Provide all the steps as of now from the beginning to last in a ipynb file."

6. Conclusion & Next Steps
This EDA provided valuable insights into the dataset's quality and characteristics. The key challenges are missing values, class imbalance, and the need for feature engineering.

Recommended Next Steps:
Address Class Imbalance: Apply techniques like SMOTE (oversampling) or use algorithms robust to imbalance.

Feature Engineering:

Derive new features from Month_1 to Month_6 (e.g., payment trends, total late payments).

Encode categorical variables like Credit_Card_Type and Location using one-hot encoding.

Model Selection and Training: Explore various classification models and evaluate them using appropriate metrics for imbalanced datasets (e.g., Precision, Recall, F1-score, AUC-ROC).

Model Evaluation: Rigorously test the final model on unseen data to ensure it generalizes well.


