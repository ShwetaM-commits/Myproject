import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt 
import seaborn as sns


df = pd.read_csv(r"C:\Users\manko\OneDrive\Desktop\Project\bank_transactions_data.csv")
print(df.head())
print(df.info())

df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], format="%d-%m-%Y %H:%M")
df["PreviousTransactionDate"] = pd.to_datetime(df["PreviousTransactionDate"], format="%d-%m-%Y %H:%M")

df = df.rename(columns={'TransactionDate':'PreviousTransactionDate', 'PreviousTransactionDate': 'Latest_TransationDate'})
df = df.drop(['gender', 'CustomerAge'], axis=1)

# Compute time difference (days since last transaction)
df["DaysSinceLastTransaction"] = (df["Latest_TransationDate"] - df["PreviousTransactionDate"]).dt.days
df.columns

# Aggregate user transaction history
user_features = df.groupby("AccountID").agg(
    MostFrequentCategory=("transaction_category_last", lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
    MostFrequentMerchant=("MerchantID", lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
    AvgDaysBetweenTransactions=("DaysSinceLastTransaction", "mean")
).reset_index()

# Merge back with original dataset
df = df.merge(user_features, on="AccountID", how="left")
df.head()
df.columns

# Aggregate user spending behavior
spending_features = df.groupby("AccountID").agg(
    TotalSpend=("TransactionAmount", "sum"),
    AvgTransactionAmount=("TransactionAmount", "mean"),
    TransactionCount=("TransactionID", "count"),
).reset_index()

# Define spending categories based on quartiles
spending_features["SpendingCategory"] = pd.qcut(
    spending_features["TotalSpend"], q=4, labels=["Low", "Medium", "High", "Premium"]
)
# Merge back with main dataframe
df = df.merge(spending_features, on="AccountID", how="left")


# Encode categorical features
categorical_cols = ["MostFrequentCategory", "MostFrequentMerchant", "offer_type_last_given","SpendingCategory"]
label_encoders = {col: LabelEncoder() for col in categorical_cols}

for col in categorical_cols:
    df[col] = label_encoders[col].fit_transform(df[col])

# Select features and target variable
features = ["TransactionCount", "AvgTransactionAmount",
            "AvgDaysBetweenTransactions", "MostFrequentCategory",
            "MostFrequentMerchant", "offer_type_last_given","SpendingCategory"]

X = df[features]
y = df["offer_accepted"]

# Standardize numerical features
scaler = StandardScaler()
X[["TransactionCount", "AvgTransactionAmount",
   "AvgDaysBetweenTransactions"]] = scaler.fit_transform(
    X[["TransactionCount", "AvgTransactionAmount",
       "AvgDaysBetweenTransactions"]])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Display dataset shape
X_train.shape, X_test.shape

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# from sklearn.model_selection import RandomizedSearchCV
# from xgboost import XGBClassifier

# # Define the parameter grid
# param_grid = {
#     'n_estimators': [100, 200, 300, 400],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'max_depth': [3, 5, 7, 10],
#     'min_child_weight': [1, 3, 5],
#     'gamma': [0, 0.1, 0.2, 0.3],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0]
# }

# # Initialize the XGBoost model
# xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# # Set up RandomizedSearchCV
# random_search = RandomizedSearchCV(
#     estimator=xgb, 
#     param_distributions=param_grid,
#     n_iter=20,  # Number of different combinations to try
#     scoring='accuracy',
#     cv=3,  # 3-fold cross-validation
#     verbose=1,
#     random_state=42,
#     n_jobs=-1
# )

# # Fit the model
# random_search.fit(X_train, y_train)

# # Get the best parameters and best model
# best_params = random_search.best_params_
# best_model = random_search.best_estimator_

# print("Best Parameters:", best_params)

# # Evaluate the tuned model
# y_pred_best = best_model.predict(X_test)
# accuracy_best = accuracy_score(y_test, y_pred_best)
# print(f"Tuned Model Accuracy: {accuracy_best}")
# print(classification_report(y_test, y_pred_best))

from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 500],  # Number of trees
    'max_depth': [3, 5, 7, 10],  # Maximum depth of each tree
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Step size shrinkage
    'subsample': [0.6, 0.8, 1.0],  # Percentage of samples used per tree
    'colsample_bytree': [0.6, 0.8, 1.0],  # Percentage of features used per tree
    'gamma': [0, 0.1, 0.2, 0.3],  # Minimum loss reduction for split
    'reg_alpha': [0, 0.01, 0.1, 1],  # L1 regularization
    'reg_lambda': [0, 0.01, 0.1, 1]  # L2 regularization
}

# Initialize XGBoost model
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

# RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    estimator=xgb_model, 
    param_distributions=param_grid, 
    n_iter=20,  # Number of combinations to try
    scoring='accuracy', 
    cv=5,  # 5-fold cross-validation
    verbose=2, 
    n_jobs=-1, 
    random_state=42
)

# Fit the model
random_search.fit(X_train, y_train)

# Best parameters & best score
print("Best Hyperparameters:", random_search.best_params_)
print("Best Accuracy Score:", random_search.best_score_)

# Train final model with best parameters
best_xgb_model = random_search.best_estimator_

# Predict on test data
y_pred = best_xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Model Accuracy: {accuracy}")
print("Classification Report:\n", classification_report(y_test, y_pred))

#1. User Spending Distribution

fig, ax = plt.subplots()
sns.histplot(df["TransactionAmount"], bins=30, kde=True, color="blue") 
ax.set_title("User Spending Distribution (Transaction Amounts)")

plt.figure(figsize=(6, 4))
sns.countplot(x=df["offer_accepted"], palette="coolwarm")
plt.title("Offer Acceptance Rate")
plt.xlabel("Offer Accepted (0=No, 1=Yes)")
plt.ylabel("Count")
plt.show()   

df.columns                 
df['MostFrequentCategory'].count()
df["MostFrequentCategory"].value_counts()



def recommend_offer(row):
 if row['SpendingCategory']==3 and row['MostFrequentCategory']==1:
     return "Loyalty Points"
 elif row['SpendingCategory']>=2 and row['MostFrequentCategory']==0:
     return "Cashback 10%"
 elif row['SpendingCategory']>=1 and row['MostFrequentCategory']==2:
     return "Discount 10%"
 elif row['SpendingCategory']==0 and row['MostFrequentCategory']==1:
     return "Freebie"
 else :
     return "BOGO"
     
# Generate target variable
df["predicted_offer"] = df.apply(recommend_offer, axis=1)

category_counts = df["transaction_category_last"].value_counts() 
sns.barplot(x=category_counts.index, y=category_counts.values, palette="coolwarm") 
plt.tight_layout() 
plt.show()

# Convert TransactionDate to extract time-based features
df["TransactionMonth"] = df["PreviousTransactionDate"].dt.to_period("M")


plt.figure(figsize=(12, 5))
df.groupby("TransactionMonth").size().plot(kind="line", marker="o", color="blue")
plt.title("Total Transactions Per Month")
plt.xlabel("Month")
plt.ylabel("Transaction Count")
plt.xticks(rotation=45)
plt.show()                        



# Comparison
# Set plot style
sns.set_style("whitegrid")

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Spending Category Distribution
sns.countplot(x=df["SpendingCategory"], palette="coolwarm", ax=axes[0, 0])
axes[0, 0].set_title("Spending Category Distribution")
axes[0, 0].set_xlabel("Spending Category")
axes[0, 0].set_ylabel("Count")


# 2. Most Frequent Transaction Categories
sns.countplot(y=df["MostFrequentCategory"], palette="Blues_r", ax=axes[0, 1])
axes[0, 1].set_title("Most Frequent Transaction Categories")
axes[0, 1].set_xlabel("Count")
axes[0, 1].set_ylabel("Transaction Category")

# 3. Offer Acceptance Rate
sns.countplot(x=df["offer_accepted"], palette="viridis", ax=axes[1, 0])
axes[1, 0].set_title("Offer Acceptance Rate")
axes[1, 0].set_xlabel("Offer Accepted (0 = No, 1 = Yes)")
axes[1, 0].set_ylabel("Count")


# 4. Personalized Offer Recommendations
df["RecommendedOffer"] = df.apply(recommend_offer, axis=1)
sns.countplot(y=df["RecommendedOffer"], palette="Set2", ax=axes[1, 1])
axes[1, 1].set_title("Personalized Offer Recommendations")
axes[1, 1].set_xlabel("Count")
axes[1, 1].set_ylabel("Recommended Offer")

plt.tight_layout()
plt.show()

#evaluation interpret 
import matplotlib.pyplot as plt
import xgboost as xgb

xgb.plot_importance(best_model)
plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()

#precision recall
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))



