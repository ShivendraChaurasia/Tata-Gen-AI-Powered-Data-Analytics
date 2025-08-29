# DATA ANALYTICS USING AI
# ( JOB SIMULATION BY TATA & FORAGE )
## My Role

I am an AI transformation consultant at Tata iQ, specializing in data-driven decision-making for financial services.

My work within a larger data analytics team, collaborating with business leaders to develop AI-powered solutions.

My team has been tasked with helping Geldium Finance, a financial services company, reduce its high credit card delinquency rate by performing advanced analytics and building AI/ML models with the assistance of GenAI.

Goal

My primary objective is to analyze customer data and predict delinquency risks using AI-driven techniques.

Develop a recommendation framework to help the Head of Collections at Geldium determine the best intervention strategies for at-risk customers.

Throughout the project, I have to ensure that AI-driven solutions are ethical, explainable, and effective in supporting responsible financial decision-making.

TASKS
EDA using GEN-AI

Exploratory analysis on provided datasets, using GenAI tools to assist with summarization, treatment of missing data, risk profiling, and synthetic data creation.

Predictive Modeling using GEN-AI

Common techniques for credit risk modeling (e.g., decision trees, logistic regression, neural networks).
Evaluating model performance and ethical considerations (bias, explainability, fairness).
Additional Refinement
Develop a structured recommendation framework that outlines the next steps, key takeaways, and business justifications for proposed actions.
Discuss ethical considerations, such as responsible AI for financial decision-making and customer fairness.
Final Presentation
Design a presentation outlining a framework for an AI-powered collections system, showing how AI can automate and optimize outreach efforts.
Identify guardrails to prevent unfair or biased decision-making.
TASK - 1
EDA USING GEN-AI

What is exploratory data analysis?

Exploratory data analysis (EDA) is the first step in understanding a dataset before applying any predictive models or making business decisions. EDA helps analysts uncover patterns, trends, inconsistencies, and missing values to ensure data quality and reliability. In the context of financial services, EDA plays a crucial role in risk assessment, allowing teams to identify key factors contributing to credit card delinquency and build stronger prediction models. Here is why EDA matters in predicting delinquency:

Ensures data integrity – Identifies missing values, duplicates, and inconsistencies before analysis.
Highlights patterns and anomalies – Helps detect trends in customer behavior, such as spending patterns before delinquency.
Prevents biased models – Reduces the risk of unfair treatment by ensuring diverse data representation.
Supports better decision-making – Provides Geldium’s Collections and Risk teams with clear insights for proactive customer engagement.
Without a thorough EDA process, predictive models can be built on flawed data, leading to inaccurate insights, poor risk management, and potentially unfair decision-making. Later in this task, you will examine a dataset related to delinquency risk, identify missing or inconsistent data points, and generate initial insights using GenAI tools.

Key steps in conducting EDA

Understanding the dataset
What are the key variables (e.g., payment history, income levels, credit utilization)?
Are there categorical or numerical data points?
Are there missing or inconsistent values?
Identifying missing values and outliers
Statistical imputation (industry standard): Replace missing values using well-established techniques such as mean, median, or regression-based imputation.

Removing irrelevant data: If a feature has excessive missing values and cannot be meaningfully imputed without introducing bias, it may be best to exclude it—but only after assessing its impact on model accuracy and fairness..

Understanding relationships between variables
Do customers with high credit utilization rates have a higher risk of delinquency?

Is there a correlation between income levels and late payments?

Detecting patterns and risk factors
Customers who miss one payment often miss multiple.
Younger customers or those with recently opened accounts may have different risk profiles.
Review the Dataset using GEN -AI

Here I am using Gemini. You can use chatgpt ,deepseek as well.

First, Download the dataset
Then upload the data in the GenAI

Screenshot 2025-06-14 172730

After uploading use this or similar type of prompt,

“Summarize key patterns, outliers, and missing values in this dataset. Highlight any fields that might present problems for modeling delinquency.”
“Identify the top 3 variables most likely to predict delinquency based on this dataset. Provide brief reasoning.”

Result,

Screenshot 2025-06-14 1759032

Screenshot 2025-06-14 180118

Data Analysis Findings Report

Here are the key findings from the initial analysis of the delinquency prediction dataset:

Notable Missing or Inconsistent Data
Missing Values:
Income: Approximately 7.8% of the data is missing (39 out of 500 records). This is a significant amount and requires a robust imputation strategy or careful handling during modeling.
Loan_Balance: Approximately 5.8% of the data is missing (29 out of 500 records), also requiring attention for imputation.
Credit_Score: Only 0.4% of the data is missing (2 out of 500 records), which can be easily managed through imputation (e.g., mean/median) or row removal.
Inconsistent Data:
Employment_Status: This categorical feature contains inconsistent capitalization and variations (e.g., 'Employed', 'employed', 'EMP', 'Self-employed', 'Unemployed', 'retired'). These need to be standardized to a consistent set of categories to ensure accurate analysis and proper feature encoding for modeling.
Key Anomalies (Outliers and Distributional Characteristics)

Credit_Utilization: While most values are within a typical range, some records show credit utilization slightly above 1.0 (maximum value of 1.025843), which could represent data entry errors or extreme cases of over-utilization, warranting further investigation.
Skewed Distributions:
Income: The distribution is right-skewed, indicating a higher frequency of lower incomes.
Credit_Utilization: Also right-skewed, with more customers having lower utilization rates.
Loan_Balance: Exhibits a right-skewed distribution, suggesting more lower loan balances. These skewed distributions might require transformation (e.g., log transformation) for certain modeling algorithms that assume normally distributed data.
Delinquent_Account Imbalance: The target variable Delinquent_Account is severely imbalanced, with 84% of accounts being non-delinquent (0) and only 16% being delinquent (1). This is a critical anomaly that will require specific techniques (e.g., oversampling, undersampling, or cost-sensitive learning) to prevent models from being biased towards the majority class.
Early Indicators of Delinquency Risk

Missed_Payments: This feature directly quantifies past payment behavior and is a strong early indicator. Higher values of missed payments are likely to correlate with increased delinquency risk.
Credit_Utilization: High credit utilization rates (closer to 1.0) generally indicate financial strain and are often strong predictors of future delinquency.
Credit_Score: Lower credit scores are typically associated with higher credit risk and thus serve as a key indicator of potential delinquency.
Debt_to_Income_Ratio: A higher debt-to-income ratio suggests a greater financial burden relative to income, which can be an early warning sign of difficulty in managing finances and potential delinquency.
Month_1 to Month_6: The sequence of payment statuses ('On-time', 'Late', 'Missed') across these six months provides a temporal pattern of payment behavior. A trend of increasing 'Late' or 'Missed' payments over these months would be a strong early indicator of escalating delinquency risk.
Initial Data Quality Observations Summary

The initial assessment of the dataset reveals generally good data quality, with most features being complete and within expected ranges. However, the presence of missing values in Income and Loan_Balance requires careful handling, and the inconsistent capitalization in Employment_Status necessitates data cleaning. The most significant data quality challenge for modeling delinquency is the substantial class imbalance in the Delinquent_Account target variable, which must be addressed to build an effective predictive model. Overall, the dataset provides a solid foundation for building a delinquency prediction model, provided these identified issues are appropriately managed.

Address missing data and data quality issues
Determine the best treatment approach for each case:
Remove: Drop columns with excessive missing data.
Impute: Fill in missing values using mean, median, or predictive modeling.
Generate synthetic data: Use AI tools to create realistic values while maintaining fairness and distribution patterns.
Use this Prompts,

“Suggest an imputation strategy for missing values in this dataset based on industry best practices.”

“Propose best-practice methods to handle missing credit utilization data for predictive modeling.”

“Generate realistic synthetic income values for missing entries using normal distribution assumptions.”

Output,

Imputation Strategies for Missing Values

For the Delinquency_prediction_dataset, the following imputation strategies are suggested based on industry best practices, considering the nature and extent of missing values in Income, Loan_Balance, and Credit_Score:

For Credit_Score (0.4% missing):

Median Imputation: Given the very small percentage of missing values, imputing with the median is a practical and robust approach. The median is less sensitive to outliers than the mean, which is beneficial if the Credit_Score distribution contains any extreme values. This method is quick to implement and has minimal impact on the overall data distribution due to the low number of missing entries.
For Income (7.8% missing) and Loan_Balance (5.8% missing): These variables are numerical and exhibit skewed distributions. For features with a moderate percentage of missing data, more sophisticated imputation methods are generally preferred over simple mean/median imputation, which can distort relationships and reduce variance.

Regression Imputation: This method involves building a predictive model (e.g., linear regression, Random Forest Regressor) to estimate missing values based on other relevant features in the dataset. This approach preserves relationships between variables better than simple imputation. For skewed data like Income and Loan_Balance, consider transforming the variable (e.g., log transform) before applying regression, and then inverse transforming the imputed values, or using a robust regression model.
K-Nearest Neighbors (KNN) Imputation: KNN imputation fills missing values by considering the values of the k nearest data points. The "nearest" points are determined by a distance metric (e.g., Euclidean distance) applied to other features. This method is effective for handling non-linear relationships and can provide more accurate imputations than simple statistical methods. It's particularly useful when the data has complex patterns.
Multiple Imputation by Chained Equations (MICE): Often considered the gold standard for imputation, MICE creates multiple complete datasets by imputing missing values sequentially based on predictive models that use all other variables. The analysis is then performed on each imputed dataset, and the results are combined using specific rules (Rubin's Rules). MICE accounts for the uncertainty introduced by imputation and provides more accurate standard errors and confidence intervals, leading to more reliable model inferences. This method is recommended for high-stakes modeling where imputation accuracy and statistical rigor are paramount.
Recommendation: For Credit_Score, median imputation is sufficient. For Income and Loan_Balance, consider starting with KNN Imputation or Regression Imputation due to their balance of effectiveness and implementation complexity. For the most robust and statistically sound results, especially if the downstream analysis is sensitive to imputation, Multiple Imputation by Chained Equations (MICE) is the best practice. The choice of strategy should also consider the downstream modeling algorithm's sensitivity to imputation methods.

NOW, Implementing all the necessary steps into Python notebook ,

In the Gemini after you use your first prompt you can convert it into a Python notebook for further adjustment and analysis.

Screenshot 2025-06-14 225222

Now, Apply further cleaning after this code or you can start from scratch as well.

But First you have to import the dataset to the Python notebook.

My Notebook - link .

3) Detect patterns and risk factors

Here's a list of high-risk indicators for delinquency based on the dataset analysis, along with insights that could impact prediction:

High-Risk Indicators for Delinquency

Missed_Payments: A higher number of missed payments is a direct and strong indicator of a customer's inability or unwillingness to meet their financial obligations, directly increasing delinquency risk.
Credit_Utilization: A higher credit utilization ratio signifies that a customer is using a large portion of their available credit, indicating potential financial strain and a greater likelihood of defaulting on payments.
Credit_Score: A lower credit score is a widely recognized measure of creditworthiness, directly correlating with a higher risk of delinquency as it reflects a history of poor credit management.
Debt_to_Income_Ratio: A higher debt-to-income ratio suggests that a significant portion of a customer's income is already allocated to debt payments, leaving less disposable income and increasing the potential for financial distress and delinquency.
"Late" or "Missed" Payment Status (Month_1 to Month_6): A trend of consistently 'Late' or 'Missed' payment statuses over recent months signals a deteriorating payment behavior pattern and serves as a strong temporal precursor to delinquency.
Insights Impacting Delinquency Prediction

Class Imbalance in Target Variable (Delinquent_Account): The significant imbalance between delinquent and non-delinquent accounts (16% vs. 84%) is crucial to address, as models trained on such imbalanced data tend to be biased towards the majority class and perform poorly in identifying high-risk individuals.
Skewed Numerical Distributions: Features like Income, Credit_Utilization, and Loan_Balance exhibit skewed distributions, which may require transformation (e.g., log transformation) to improve the performance of models sensitive to feature distribution assumptions.
Inconsistent Categorical Data (Employment_Status): The initial inconsistencies in the Employment_Status column highlight the importance of thorough data cleaning and standardization, as uncleaned categorical features can lead to misleading insights and reduced model accuracy.
Time-Series Potential of Month_x Variables: The Month_x variables represent a sequence of payment behaviors over six months, offering a rich source of information for capturing trends and dynamic risk patterns that could be leveraged with time-series modeling techniques or by creating aggregated features (e.g., number of late payments in last 3 months).
4) Final EDA report

Exploratory Data Analysis (EDA) Summary Report
1. Introduction
This report summarizes the exploratory data analysis performed on the delinquency prediction dataset. The primary goal of this analysis was to understand the dataset's structure, identify key patterns, detect anomalies and missing values, and pinpoint potential risk indicators for predicting customer delinquency.

2. Dataset Overview
This section summarizes the dataset, including the number of records, key variables, and data types. It also highlights any anomalies or inconsistencies observed during the initial review.

Number of records: 500
Key variables:
Customer_ID: Unique identifier for each customer.
Age: Customer's age.
Income: Customer's annual income.
Credit_Score: Customer's credit score.
Credit_Utilization: Ratio of credit used to available credit.
Missed_Payments: Number of missed payments.
Delinquent_Account: Target variable (0 = Not Delinquent, 1 = Delinquent).
Loan_Balance: Outstanding loan balance.
Debt_to_Income_Ratio: Ratio of debt to income.
Employment_Status: Customer's employment status.
Account_Tenure: Duration of the account.
Credit_Card_Type: Type of credit card held.
Location: Customer's geographic location.
Month_1 - Month_6: Payment status for the past six months ('On-time', 'Late', 'Missed').
Data types: The dataset contains a mix of numerical (float64, int64) and categorical/object data types.
Anomalies, duplicates, or inconsistencies observed during the initial review:
Class Imbalance: The Delinquent_Account target variable is highly imbalanced, with 84% non-delinquent accounts and 16% delinquent accounts.
Inconsistent Categorical Entries: The Employment_Status column had inconsistent capitalization and variations (e.g., 'Employed', 'employed', 'EMP', 'retired'), which required standardization.
Potential Outliers in Credit_Utilization: A few Credit_Utilization values were slightly above 1.0, which might indicate extreme utilization or data entry anomalies.
Skewed Distributions: Income, Loan_Balance, and Credit_Utilization showed right-skewed distributions.
3. Missing Data Analysis
Identifying and addressing missing data is critical to ensuring model accuracy. This section outlines missing values in the dataset, the approach taken to handle them, and justifications for the chosen method.

Variables with missing values:
Income: 7.8% missing.
Loan_Balance: 5.8% missing.
Credit_Score: 0.4% missing.
Missing data treatment: Imputation was chosen as the treatment method to retain as much data as possible, given the moderate percentage of missing values.
Credit_Score: Imputed with the median due to the very small percentage of missing values, which minimizes impact and robustness to potential outliers.
Income and Loan_Balance: Imputed with the median values. While more advanced methods like regression or KNN imputation were considered, median imputation was chosen for its simplicity and effectiveness given the dataset size and initial exploratory phase, particularly for the skewed distributions.
4. Key Findings and Risk Indicators
This section identifies trends and patterns that may indicate risk factors for delinquency. Feature relationships and statistical correlations are explored to uncover insights relevant to predictive modeling.

Key findings (Patterns and Anomalies):
The Delinquent_Account distribution highlights a significant class imbalance, which is a major challenge for predictive modeling.
Numerical features like Income, Loan_Balance, and Credit_Utilization are skewed, suggesting that transformations might be beneficial for certain models.
The Employment_Status column required standardization due to inconsistent entries, revealing common employment categories: 'Unemployed', 'Retired', 'Employed', and 'Self-employed'.
Payment statuses in Month_1 through Month_6 show a mix of 'On-time', 'Missed', and 'Late' payments, providing a historical sequence of behavior crucial for trend analysis.
Correlations observed between key variables: While explicit correlation calculations were not detailed, it is expected that Credit_Score would have a strong inverse correlation with Delinquent_Account, and Missed_Payments, Credit_Utilization, and Debt_to_Income_Ratio would show positive correlations with delinquency.
Unexpected anomalies: No highly unexpected or severe anomalies were found that would suggest significant data corruption, beyond the minor Credit_Utilization values slightly above 1.0.
5. AI & GenAI Usage
Generative AI tools were used to summarize the dataset, impute missing data, and detect patterns. This section documents AI-generated insights and the prompts used to obtain results.

Example AI prompts used:
"Summarize key patterns, outliers, and missing values in this dataset. Highlight any fields that might present problems for modeling delinquency."
"Document your findings in bullet points for your report. Focus on: Notable missing or inconsistent data, Key anomalies, Early indicators of delinquency risk. Then, write a short paragraph (3–5 sentences) summarizing your initial data quality observations."
"Suggest an imputation strategy for missing values in this dataset based on industry best practices."
"Implement and provide a cleaned dataset."
"Can you provide it in excel format?"
"Provide all the steps as of now form the beginning to last in a ipynb file."
"List high-risk indicators, each with a one-sentence explanation of why it’s important, as well as any insights that could impact delinquency prediction."
6. Conclusion & Next Steps
This initial EDA provided valuable insights into the dataset's quality and characteristics. The key challenges identified are the missing values, the class imbalance in the target variable, and the need for feature engineering from categorical and time-series-like variables.

Recommended Next Steps:

Address Class Imbalance: Apply techniques such as oversampling (e.g., SMOTE), undersampling, or using algorithms robust to imbalance to ensure the model can effectively identify delinquent accounts.
Feature Engineering:
Derive new features from Month_1 to Month_6 (e.g., total late payments, consecutive missed payments, payment trends).
Encode categorical variables like Credit_Card_Type and Location using appropriate methods (e.g., one-hot encoding).
Model Selection and Training: Explore various machine learning models suitable for binary classification, evaluating their performance using appropriate metrics for imbalanced datasets (e.g., Precision, Recall, F1-score, AUC-ROC).
Model Evaluation: Rigorously evaluate the chosen model's performance on unseen data to ensure its generalization capability.
