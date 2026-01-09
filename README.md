##Credit Risk Modeling (Loan Default Prediction)

##Overview

This project is about predicting whether a loan applicant will default or not. The idea is to help banks identify high-risk customers before approving loans. I built an end-to-end ML pipeline with explainability so it's not just about accuracy but also about understanding why the model predicts default.

##Dataset

I used the Home Credit Default Risk dataset from Kaggle.
Link to dataset: https://www.kaggle.com/c/home-credit-default-risk/data

Target column:
 1. TARGET = 1 → Customer defaulted
 2. TARGET = 0 → Customer paid the loan

Dataset contains financial + demographic info of customers.

##Steps I Followed
1. EDA (Exploratory Data Analysis)
 * Checked class imbalance
 * Looked at income, credit amount, annuity, etc.
 * Found missing values and outliers
 * Basic visualization and interpretation
2. Data Cleaning
 * Removed columns with extremely high missing values
 * Filled missing numeric data with median and categorical with mode
 * Capped extreme outliers using percentiles
3. Feature Engineering
* Created features that make financial sense like:
  * Loan-to-Income ratio
  * Annuity-to-Income ratio
  * Age (converted from days)
  * Employment duration
*Replaced infinite values created during division
4. Baseline Model
 * Logistic Regression
 * Used class_weight='balanced' because of imbalance
 * Evaluated using ROC-AUC (better than accuracy for imbalance)
5. Advanced Model
 * XGBoost Classifier
 * Used scale_pos_weight to handle imbalance
 * Improved ROC-AUC compared to baseline
6. Explainability
 * Used SHAP values to understand which features influence default
 * Looked at both global and individual explanations
 * Useful for banking domain because transparency matters
7. Business Impact

*The model can help banks:
   * Filter out risky loan applications
   * Reduce losses due to defaults
   * Take decisions backed by data instead of guesswork

##Important Features Observed

Some of the important features according to SHAP:
 * Loan-to-Income ratio
 * Annuity-to-Income ratio
 * Total income
 * Employment stability
 * Age

These make sense because affordability and income stability affect repayment ability.

##Business Insights

During this project, several meaningful patterns emerged from a business perspective:

 * Applicants with a high loan-to-income ratio showed a higher likelihood of default, indicating affordability challenges.
 * Lower-income customers defaulted more frequently, suggesting that income stability affects repayment capability.
 * Customers with longer employment duration demonstrated lower default risk, likely due to more stable and predictable income.
 * Younger applicants showed slightly higher default tendencies, potentially due to limited credit experience.
 * Tree-based models such as XGBoost captured combinations like high loan amount + low income + short employment history, which aligned with high-risk behavior.
 * SHAP analysis highlighted loan-to-income, annuity-to-income, income, and employment stability as the most influential features.

##Models Used
 * Logistic Regression
 * XGBoost

Metric used:

 * ROC-AUC (because dataset is imbalanced)

##Tools & Libraries

 * Python
 * Pandas, NumPy
 * Scikit-learn
 * XGBoost
 * SHAP
 * Matplotlib / Seaborn

credit-risk-model/
 │
 ├── notebooks/
 │   ├── 01_business_understanding_eda.ipynb
 │   ├── 02_data_cleaning.ipynb
 │   ├── 03_feature_engineering.ipynb
 │   ├── 04_logistic_regression.ipynb
 │   ├── 05_xgboost.ipynb
 │   └── 06_shap_explainability.ipynb

##(Note: I did not upload the raw dataset because it’s large. You can download it from Kaggle.)## 

##Conclusion

This project helped me understand how real loan risk problems work.
Key takeaways for me:
 * Imbalanced data handling is important
 * Ratios/features matter more than raw values
 * Explainability is very important for finance domain
