# DSCI-2025-Team-C

Capstone Project

<h3>Project Overview</h3>

The main goal is to identify whether each transaction is fraudulent and calculate their fraud probability. The dataset is about 590,000 rows and 434 columns.

The data is mainly divided into 2 categories, which are joined by TransactionID. Not all transactions have corresponding identity information.

- Transaction data
- Identity data.

<h3>Data Description</h3>

Transaction Features:

- TransactionDT: Timedelta from a given reference datetime (not an actual timestamp), but the time difference in seconds from a certain time.
- TransactionAMT: Transaction payment amount in USD, the decimal part is worth paying attention to.
- ProductCD: Product code, the product for each transaction. It may not necessarily be an actual product, but may also refer to a service.
- card1-card6: Payment card information, such as card type, card category, issue bank, country, etc.
- addr1-addr2: Address, billing region, and billing country
- dist: Distances between (not limited) billing address, mailing address, zip code, IP address, phone area, etc.
- P_ and (R__) emaildomain: Purchaser and recipient email domain, some transactions do not require the recipient, and the corresponding Remaildomain is empty
- C1-C14: Counting, such as how many addresses are found to be associated with the payment card, etc.
- D1-D15: Timedelta, such as days between previous transaction, etc.
- M1-M9: Match, such as names on the card and address, etc.
- Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations. Some V features are missing in different proportions.

Identity Features:

- id_01-id_11: Numerical features for identity, which are collected by Vesta and security partners such as device rating, IP domain rating, proxy rating, etc. Also, it recorded behavioral fingerprints like account login times/failed login times, how long an account stayed on the page, etc.
- DeviceType, DeviceInfo, and id_12-id_38: Categorical Features

<h3>Problem Type</h3>
Since we need to classify a transaction as fraudulent or non-fraudulent, this is a binary classification problem.

<h4>Performance Metric</h4>
In fraud detection, both false positives and false negatives have serious consequences, but false negatives, missed fraud cases can result in substantial financial losses. While precision and recall each highlight a specific type of error, the F1-score provides a balanced measure between the two, yet lacks flexibility in emphasizing one over the other. In contrast, the ROC-AUC metric evaluates a model’s ability to distinguish between classes across all possible thresholds. This makes ROC-AUC especially valuable, as it offers a comprehensive assessment of model performance while supporting threshold adjustments to align sensitivity and specificity with real-world risk priorities.

<h3>Exploratoy Data Analysis (EDA)</h3>

One of the initial observations during exploratory data analysis was the sparsity and imbalance of the dataset. 
- The target variable isFraud showed a significant class imbalance, with approximately 96.5% non-fraud and 3.5% fraud transactions.
- Fraudulent activity was found to be more prevalent in transactions conducted via mobile devices (DeviceType) and among users flagged as 'IP_PROXY:ANONYMOUS' based on the id_31 feature.
- Additionally, the TransactionDT field, rather than being a true timestamp, represented a timedelta, highlighting the need for temporal feature engineering.
- A large portion of the dataset—particularly the anonymized V-series features—contained substantial missing values and showed non-normal distributions, complicating imputation and model interpretability.

These findings informed key preprocessing steps, including imputation strategies, feature transformation, and sampling design to address imbalance and sparsity.

The libraries used are:

- numpy
- pandas
- matplotlib,
- seaborn
- sklearn
- lightgbm
- xgboost
- catboost
- Random Forest
- streamlit

<h3>Our Contribution</h3>

- We develop a framework to evaluate how imbalanced data influences model bias and how sampling strategies can mitigate this issue.
- This research compares various classifiers, including Random Forest, Logistic Regression, XGBoost, and CatBoost, and evaluates them using both accuracy and F1-score.
- We demonstrate that raw accuracy may be misleading on imbalanced datasets and advocate for F1-score as a more appropriate metric.
- XGBoost emerged as the most effective model when combined with sampling methods. Among the three data balancing strategies tested—random oversampling, random undersampling, and SMOTE, random oversampling yielded the best results.
- Utilized XGBoost using undersampling for deployment due to its strong performance using Streamlit. 


<h3>Applied 3 Modeling Approach</h3>

![image](https://github.com/user-attachments/assets/0bba2749-4d88-41f4-9be9-b65ccfbdc68f)

Streamlit UI: https://nbakhati-dsci-2025-team-c-streamlitapp-div3dy.streamlit.app/
![image](https://github.com/user-attachments/assets/ec43a2e1-9993-4fc6-85c4-30e0d3589662)




