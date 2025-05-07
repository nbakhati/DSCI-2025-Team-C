# DSCI-2025-Team-C

Capstone Project

<h3>Competition Overview</h3>

The main goal is to identify whether each transaction is fraudulent. Among them, the dataset is about 590,000 rows and 434 columns.

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
In fraud detection, both false positives and false negatives carry significant implications, but false negatives—failing to detect fraud—can lead to major financial losses. Metrics like precision and recall address one type of error each, while the F1-score balances them without offering flexibility in prioritization. ROC-AUC, however, evaluates a model’s ability to distinguish between classes across all thresholds and allows for customizable decision boundaries. This makes it an ideal performance metric, providing a comprehensive view of model effectiveness while enabling users to balance sensitivity and specificity based on real-world priorities.

<h3>Exploratoy Data Analysis (EDA)</h3>

- One of the first things we noticed when conducting EDA was the sparsity of the dataset.
- The distribution of the target variable 'isFraud' has a class imbalance problem, where it shows that 96.5% of the data contains non-fraud transactions, whereas only 3.5% are fraud.
- Target variable 'isFraud' is more prevalent in the mobile 'DeviceType' as well as more prevalent in the 'IP_PROXY:ANONYMOUS' based on 'id_31'.
- Another observation that was immediately apparent is the imbalanced nature of the data. This shows that 'TransactionDT' is a timedelta gap, not a timestamp.
- Dataset has a very high percentage of missing values, especially the V columns.
- Anonymized columns not only had a high amount of missing data, but their distributions were also not normally distributed.

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

<h3>Our Contribution</h3>

- We develop a framework to evaluate how imbalanced data influences model bias and how sampling strategies can mitigate this issue.
- This research compares various classifiers, including Random Forest, Logistic Regression, XGBoost, and CatBoost, and evaluates them using both accuracy and F1-score.
- We demonstrate that raw accuracy may be misleading on imbalanced datasets and advocate for F1-score as a more appropriate metric.
- XGBoost emerged as the most effective model when combined with sampling methods. Among the three data balancing strategies tested—random oversampling, random undersampling, and SMOTE, random oversampling yielded the best results.
- Utilized XGBoost using undersampling for deployment due to its strong performance using Streamlit. 


<h3>Applied 3 Modeling Approach</h3>

![image](https://github.com/user-attachments/assets/0bba2749-4d88-41f4-9be9-b65ccfbdc68f)

