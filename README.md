# Project1
Abstract
This project focuses on developing a credit card fraud detection system using machine learning techniques. The analysis begins with loading and pre - processing a dataset containing credit card transaction information from both training and testing files. Exploratory data analysis (EDA) is conducted to understand the distribution of various features such as gender, transaction category, and transaction hour. After pre - processing the data, multiple machine learning models including Logistic Regression and Support Vector Machine (SVM) are trained. The performance of these models is evaluated using common classification metrics such as accuracy, precision, recall, and F1 - score. The results show the effectiveness of these models in detecting credit card fraud, and suggestions for further improvement are provided.
Rationale
Credit card fraud is a major concern for financial institutions and consumers alike. Detecting fraudulent transactions in a timely and accurate manner can help prevent financial losses, protect the privacy of customers, and maintain the integrity of the financial system. By leveraging machine learning algorithms on transaction data, we can build a model that can automatically identify potentially fraudulent activities, reducing the need for manual review and improving the overall efficiency of fraud detection.
Research Question
Can we develop an accurate and reliable machine learning model to detect credit card fraud based on the given transaction data? What are the key features that contribute to the detection of fraud? And how can we optimize the model's performance?
Data Sources
The data used in this project is sourced from two CSV files: fraudTrain.csv and fraudTest.csv located in the Credit Card Transactions Fraud Detection Dataset directory. These files contain a comprehensive set of credit card transaction information, including the transaction date and time, transaction amount, merchant details, transaction category, customer demographic information, and a label indicating whether the transaction is fraudulent or not.
Methodology
1. Data Loading and Initial Exploration
The first step is to load the training and testing datasets using the pandas library and concatenate them into a single DataFrame. Basic information about the dataset such as the number of rows and columns, data types, and the first few rows are then examined to get an initial understanding of the data.

python
运行
import pandas as pd
df_train = pd.read_csv('Credit Card Transactions Fraud Detection Dataset/fraudTrain.csv')
df_test = pd.read_csv('Credit Card Transactions Fraud Detection Dataset/fraudTest.csv')
df = pd.concat([df_train, df_test], ignore_index=True)
print('Data shape:', df.shape)
print('First 10 rows:', df.head(10))
print('Column names:', df.columns)
print('Data information:', df.info())
2. Data Pre - processing
Date Conversion: The trans_date_trans_time and dob columns are converted to the datetime type to facilitate further analysis related to time and age.

python
运行
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["dob"] = pd.to_datetime(df["dob"])
print('Updated data information:', df.info())

Data Cleaning: Unnecessary columns such as Unnamed: 0, cc_num, first, etc. are dropped from the dataset. Missing values are also removed to ensure data quality.

python
运行
def clean_data(clean):
    clean.drop(["Unnamed: 0", 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num', 'trans_date_trans_time'], axis = 1, inplace = True)
    clean.dropna()
    return clean
df = clean_data(df)

Encoding Categorical Variables: Categorical variables like merchant, category, job, and gender are encoded using the LabelEncoder from the sklearn library to convert them into numerical values suitable for machine learning algorithms.

python
运行
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for column in ['merchant', 'category', 'job', 'gender']:
    df[column] = label_encoder.fit_transform(df[column])
print('First few rows after encoding:', df.head())
3. Exploratory Data Analysis (EDA)
Gender Distribution: A pie chart is used to visualize the distribution of genders in the dataset.

python
运行
import matplotlib.pyplot as plt
gender_counts = df['gender'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Genders')
plt.axis('equal')
plt.show()

Transaction Category Distribution: Both bar charts and pie charts are used to show the distribution of transaction categories.

python
运行
# Bar chart
category_counts = df['category'].value_counts()
plt.figure(figsize=(14, 6))
bars = plt.bar(category_counts.index, category_counts.values)
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("Distribution of Categories")
plt.xticks(rotation=90, ha='center')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')
plt.tight_layout()
plt.show()

# Pie chart
plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Categories')
plt.axis('equal')
plt.show()

Hourly Transaction Distribution: The hourly distribution of total transactions and fraudulent transactions is analyzed and visualized. The highest and lowest bars are highlighted for better understanding.

python
运行
df['transaction_hour'] = df['trans_date_trans_time'].dt.hour
# Total transactions
hourly_transactions = df.groupby('transaction_hour')['cc_num'].count()
plt.figure(figsize=(18, 6))
bars = plt.bar(hourly_transactions.index, hourly_transactions.values)
sorted_bars = sorted(bars, key=lambda bar: bar.get_height())
sorted_bars[-1].set_color('red')  # Highest bar
sorted_bars[-2].set_color('red')  # Second highest bar
sorted_bars[0].set_color('green')  # Lowest bar
sorted_bars[1].set_color('green')  # Second lowest bar
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Transactions")
plt.title("Hourly Transaction Distribution")
plt.xticks(range(24))
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')
plt.show()

# Fraudulent transactions
fraudulent_transactions = df[df['is_fraud'] == 1]
hourly_fraud_transactions = fraudulent_transactions.groupby('transaction_hour')['cc_num'].count()
plt.figure(figsize=(16, 6))
bars = plt.bar(hourly_fraud_transactions.index, hourly_fraud_transactions.values)
sorted_bars = sorted(bars, key=lambda bar: bar.get_height())
sorted_bars[-1].set_color('red')
sorted_bars[-2].set_color('red')
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Fraudulent Transactions")
plt.title("Hourly Fraud Transaction Distribution")
plt.xticks(range(24))
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')
plt.show()

# Combined plot
plt.figure(figsize=(16, 6))
plt.bar(hourly_transactions.index, hourly_transactions.values, label='Total Transactions')
plt.plot(hourly_fraud_transactions.index, hourly_fraud_transactions.values, marker='o', color='red', label='Fraudulent Transactions')
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Transactions")
plt.title("Hourly Transaction and Fraud Distribution")
plt.xticks(range(24))
plt.legend()
plt.show()

Fraud vs. Non - Fraud Transaction Counts: A bar chart is used to compare the number of fraudulent and non - fraudulent transactions.

python
运行
fraud_counts = df['is_fraud'].value_counts()
plt.figure(figsize=(12, 6))
plt.bar(fraud_counts.index, fraud_counts.values, color=['blue', 'orange'])
plt.xticks(fraud_counts.index, ['Not Fraud', 'Fraud'])
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.title("Fraud vs. Non - Fraud Transaction Counts")
for i, v in enumerate(fraud_counts.values):
    plt.text(i, v + 5000, str(v), ha='center', va='bottom', fontweight='bold')
plt.show()

Correlation Analysis: A heatmap is created to visualize the correlation between different features in the dataset.

python
运行
import seaborn as sns
correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()
4. Model Training
The dataset is split into training and testing sets using the train_test_split function from sklearn. Two machine learning models, Logistic Regression and Support Vector Machine (using SGDClassifier for faster training), are trained on the training data.

python
运行
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
X = df.drop('is_fraud', axis = 1)
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Support Vector Machine
svm_model = SGDClassifier(loss='hinge', max_iter = 1000, tol = 1e - 3)
svm_model.fit(X_train, y_train)
print("All models trained successfully!")
5. Model Evaluation
A function is defined to calculate and print performance metrics such as accuracy, precision, recall, and F1 - score for each model.

python
运行
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def print_performance_metrics(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"Performance Metrics for {model_name}:")
    print("-" * 30)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 - Score: {f1_score(y_test, y_pred):.4f}")
    print("-" * 30 + "\n")

models = {
    "Logistic Regression": lr_model,
    "SVM": svm_model
}
for model_name, model in models.items():
    print_performance_metrics(model, X_test, y_test, model_name)
Results
The performance of the Logistic Regression and SVM models is evaluated based on the accuracy, precision, recall, and F1 - score. The specific values of these metrics are printed during the model evaluation step. A high accuracy indicates that the model can correctly classify a large proportion of transactions. Precision measures the proportion of correctly predicted positive cases (fraudulent transactions) out of all predicted positive cases. Recall measures the proportion of actual positive cases that are correctly predicted. The F1 - score is a weighted average of precision and recall, providing a balanced measure of the model's performance.
Next steps
Try More Algorithms: Explore other machine learning algorithms such as Decision Tree, Random Forest, XGBoost, and LightGBM to see if they can achieve better performance in detecting credit card fraud.
Hyperparameter Tuning: Use techniques like grid search or random search to optimize the hyperparameters of the existing models and improve their performance.
Feature Engineering: Look for additional features or transform existing features to extract more useful information from the data and enhance the model's ability to detect fraud.
Handling Imbalanced Data: Since the number of fraudulent transactions is usually much smaller than non - fraudulent transactions, use techniques such as oversampling, undersampling, or SMOTE to handle the class imbalance and improve the model's performance on detecting fraud.
Conclusion
The results of this project show that both Logistic Regression and SVM models can be used to detect credit card fraud to a certain extent. However, there is still room for improvement. Based on the analysis, we recommend further exploring different machine learning algorithms and conducting hyperparameter tuning to optimize the model's performance. It should be noted that the performance of the models may be affected by the quality and quantity of the data, and continuous data updates and model optimization are needed to adapt to the changing patterns of credit card fraud.
