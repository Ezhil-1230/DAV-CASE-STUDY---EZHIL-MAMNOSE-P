# ============================================================
# E-COMMERCE PRODUCT RETURN ANALYSIS
# ============================================================

# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import site
site.addsitedir(site.getusersitepackages())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("returns_sustainability_dataset.csv")

print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nFirst 5 records:")
print(df.head())


# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

# ------------------------------------------------------------
# 3.1 Create a working copy
# ------------------------------------------------------------

df_clean = df.copy()

print("\n====================================")
print("DATA PREPROCESSING")
print("====================================")

print("\nOriginal dataset shape:", df.shape)
print("Working dataset shape:", df_clean.shape)


# ------------------------------------------------------------
# 3.2 Check Missing Values
# ------------------------------------------------------------

missing_values = df_clean.isnull().sum()

print("\n------------------------------------")
print("MISSING VALUE CHECK")
print("------------------------------------")

print("\nMissing values in each column:")
print(missing_values)

print("\nTotal missing values:",
      df_clean.isnull().sum().sum())


# ------------------------------------------------------------
# 3.3 Check Duplicate Records
# ------------------------------------------------------------

duplicate_count = df_clean.duplicated().sum()

print("\n------------------------------------")
print("DUPLICATE RECORD CHECK")
print("------------------------------------")

print("\nNumber of duplicate records:",
      duplicate_count)


# ------------------------------------------------------------
# 3.4 Check Data Types
# ------------------------------------------------------------

print("\n------------------------------------")
print("DATA TYPES")
print("------------------------------------")

print(df_clean.dtypes)


# ------------------------------------------------------------
# 3.5 Convert Order_Date to Datetime
# ------------------------------------------------------------

df_clean["Order_Date"] = pd.to_datetime(
    df_clean["Order_Date"]
)

print("\n------------------------------------")
print("DATE CONVERSION")
print("------------------------------------")

print("Order_Date data type after conversion:")
print(df_clean["Order_Date"].dtype)


# ------------------------------------------------------------
# 3.6 Create Date-Based Features
# ------------------------------------------------------------

df_clean["Order_Year"] = df_clean["Order_Date"].dt.year

df_clean["Order_Month"] = df_clean["Order_Date"].dt.month

df_clean["Order_Month_Name"] = (
    df_clean["Order_Date"].dt.strftime("%b")
)

df_clean["Order_Year_Month"] = (
    df_clean["Order_Date"]
    .dt.to_period("M")
    .astype(str)
)

print("\nDate-based features created:")
print("- Order_Year")
print("- Order_Month")
print("- Order_Month_Name")
print("- Order_Year_Month")


# ------------------------------------------------------------
# 3.7 Validate Categorical Variables
# ------------------------------------------------------------

categorical_columns = [
    "Product_Category",
    "Shipping_Method",
    "Payment_Method",
    "User_Gender",
    "Return_Status",
    "Return_Reason"
]

print("\n------------------------------------")
print("CATEGORICAL DATA VALIDATION")
print("------------------------------------")

for column in categorical_columns:

    print("\n", column)
    print("--------------------------------")

    print(df_clean[column].value_counts())


# ------------------------------------------------------------
# 3.8 Validate Numerical Variables
# ------------------------------------------------------------

numeric_columns = [
    "Product_Price",
    "Order_Quantity",
    "Discount_Applied",
    "User_Age",
    "Days_to_Return",
    "Order_Value",
    "Return_Cost",
    "Profit_Loss",
    "CO2_Emissions",
    "Packaging_Waste",
    "CO2_Saved",
    "Waste_Avoided"
]

print("\n------------------------------------")
print("NUMERICAL DATA VALIDATION")
print("------------------------------------")

print(
    df_clean[numeric_columns].describe().T
)


# ------------------------------------------------------------
# 3.9 Outlier Inspection
# ------------------------------------------------------------

print("\n------------------------------------")
print("OUTLIER INSPECTION")
print("------------------------------------")


# Product Price

plt.figure(figsize=(8, 5))

plt.boxplot(df_clean["Product_Price"])

plt.title("Box Plot of Product Price")
plt.ylabel("Product Price")

plt.tight_layout()
plt.show()


# Order Value

plt.figure(figsize=(8, 5))

plt.boxplot(df_clean["Order_Value"])

plt.title("Box Plot of Order Value")
plt.ylabel("Order Value")

plt.tight_layout()
plt.show()


# User Age

plt.figure(figsize=(8, 5))

plt.boxplot(df_clean["User_Age"])

plt.title("Box Plot of User Age")
plt.ylabel("User Age")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 3.10 Check Target Variable Distribution
# ------------------------------------------------------------

print("\n------------------------------------")
print("RETURN STATUS DISTRIBUTION")
print("------------------------------------")

print("\nReturn Status Counts:")

print(
    df_clean["Return_Status"].value_counts()
)

print("\nReturn Status Percentages:")

print(
    df_clean["Return_Status"]
    .value_counts(normalize=True)
    .mul(100)
)


# ------------------------------------------------------------
# 3.11 Define Machine Learning Features
# ------------------------------------------------------------

ml_features = [
    "Product_Category",
    "Product_Price",
    "Order_Quantity",
    "Discount_Applied",
    "Shipping_Method",
    "Payment_Method",
    "User_Age",
    "User_Gender",
    "User_Location",
    "Order_Value"
]

target = "Return_Status"


# ------------------------------------------------------------
# 3.12 Separate Input Features and Target
# ------------------------------------------------------------

X = df_clean[ml_features]

y = df_clean[target]


# ------------------------------------------------------------
# 3.13 Identify Excluded Features
# ------------------------------------------------------------

excluded_features = [
    "Return_Reason",
    "Days_to_Return",
    "Return_Cost",
    "Profit_Loss"
]

print("\n------------------------------------")
print("FEATURE SELECTION")
print("------------------------------------")

print("\nFeatures used for ML prediction:")

for feature in ml_features:
    print("-", feature)


print("\nFeatures excluded from ML prediction:")

for feature in excluded_features:
    print("-", feature)


# ------------------------------------------------------------
# 3.14 Final Preprocessing Verification
# ------------------------------------------------------------

print("\n====================================")
print("FINAL PREPROCESSING VERIFICATION")
print("====================================")

print("\nDataset shape:")
print(df_clean.shape)

print("\nTotal missing values:")
print(df_clean.isnull().sum().sum())

print("\nTotal duplicate rows:")
print(df_clean.duplicated().sum())

print("\nTarget distribution:")
print(df_clean["Return_Status"].value_counts())

print("\nML input features:")
print(X.columns.tolist())

print("\nTarget variable:")
print(target)

print("\nData preprocessing completed successfully!")

# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n====================================")
print("EXPLORATORY DATA ANALYSIS")
print("====================================")


# ============================================================
# 4.1 BASIC EDA SUMMARY
# ============================================================

print("\nReturn Status:")
print(df_clean["Return_Status"].value_counts())

print("\nProduct Categories:")
print(df_clean["Product_Category"].value_counts())

print("\nReturn Reasons:")
print(df_clean["Return_Reason"].value_counts())

print("\nOrder Date Range:")
print(df_clean["Order_Date"].min())
print(df_clean["Order_Date"].max())


# ============================================================
# 4.2 VISUALIZATION 1
# RETURN STATUS DISTRIBUTION
# ============================================================

return_counts = df_clean["Return_Status"].value_counts()

plt.figure(figsize=(8, 5))

plt.bar(
    return_counts.index,
    return_counts.values
)

plt.title("Return Status Distribution")
plt.xlabel("Return Status")
plt.ylabel("Number of Orders")

plt.tight_layout()
plt.show()


# ============================================================
# 4.3 VISUALIZATION 2
# RETURN RATE BY PRODUCT CATEGORY
# ============================================================

category_return_rate = (
    df_clean.groupby("Product_Category")["Return_Status"]
    .apply(lambda x: (x == "Returned").mean() * 100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 5))

plt.bar(
    category_return_rate.index,
    category_return_rate.values
)

plt.title("Return Rate by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Return Rate (%)")

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()


# ============================================================
# 4.4 VISUALIZATION 3
# MONTHLY RETURN RATE TREND
# ============================================================

monthly_return_rate = (
    df_clean.groupby("Order_Year_Month")["Return_Status"]
    .apply(lambda x: (x == "Returned").mean() * 100)
)

plt.figure(figsize=(12, 5))

plt.plot(
    monthly_return_rate.index,
    monthly_return_rate.values,
    marker="o"
)

plt.title("Monthly Return Rate Trend")
plt.xlabel("Month")
plt.ylabel("Return Rate (%)")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()
plt.show()


# ============================================================
# 4.5 VISUALIZATION 4
# RETURN REASONS DISTRIBUTION
# ============================================================

return_reasons = (
    df_clean[
        df_clean["Return_Status"] == "Returned"
    ]["Return_Reason"]
    .value_counts()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 5))

plt.bar(
    return_reasons.index,
    return_reasons.values
)

plt.title("Distribution of Product Return Reasons")
plt.xlabel("Return Reason")
plt.ylabel("Number of Returned Orders")

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()


# ============================================================
# 4.6 VISUALIZATION 5
# FINANCIAL IMPACT OF RETURNS
# ============================================================

returned_data = df_clean[
    df_clean["Return_Status"] == "Returned"
]

total_returned_order_value = (
    returned_data["Order_Value"].sum()
)

total_return_cost = (
    returned_data["Return_Cost"].sum()
)

financial_values = [
    total_returned_order_value,
    total_return_cost
]

financial_labels = [
    "Returned Order Value",
    "Return Cost"
]

plt.figure(figsize=(9, 5))

plt.bar(
    financial_labels,
    financial_values
)

plt.title("Financial Impact of Returned Orders")
plt.xlabel("Financial Measure")
plt.ylabel("Amount")

plt.xticks(rotation=15)

plt.tight_layout()
plt.show()


# ============================================================
# 4.7 PRINT EDA RESULTS
# ============================================================

print("\n====================================")
print("EDA RESULTS")
print("====================================")

print("\nReturn Rate by Product Category:")
print(category_return_rate.round(2))

print("\nMonthly Return Rate:")
print(monthly_return_rate.round(2))

print("\nReturn Reasons:")
print(return_reasons)

print("\nTotal Returned Order Value:")
print(round(total_returned_order_value, 2))

print("\nTotal Return Cost:")
print(round(total_return_cost, 2))

print("\nExploratory Data Analysis completed successfully!")

# ============================================================
# 5. MACHINE LEARNING
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("\n====================================")
print("MACHINE LEARNING")
print("====================================")


# ------------------------------------------------------------
# 5.1 DEFINE FEATURES AND TARGET
# ------------------------------------------------------------

ml_features = [
    "Product_Category",
    "Product_Price",
    "Order_Quantity",
    "Discount_Applied",
    "Shipping_Method",
    "Payment_Method",
    "User_Age",
    "User_Gender",
    "User_Location",
    "Order_Value"
]

target = "Return_Status"

X = df_clean[ml_features]
y = df_clean[target]


# ------------------------------------------------------------
# 5.2 IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# ------------------------------------------------------------

categorical_features = [
    "Product_Category",
    "Shipping_Method",
    "Payment_Method",
    "User_Gender",
    "User_Location"
]

numerical_features = [
    "Product_Price",
    "Order_Quantity",
    "Discount_Applied",
    "User_Age",
    "Order_Value"
]


# ------------------------------------------------------------
# 5.3 ONE-HOT ENCODING
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ------------------------------------------------------------
# 5.4 TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ------------------------------------------------------------
# 5.5 RANDOM FOREST CLASSIFIER
# ------------------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ------------------------------------------------------------
# 5.6 CREATE MACHINE LEARNING PIPELINE
# ------------------------------------------------------------

model_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", rf_model)
    ]
)


# ------------------------------------------------------------
# 5.7 TRAIN THE MODEL
# ------------------------------------------------------------

model_pipeline.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")


# ------------------------------------------------------------
# 5.8 MAKE PREDICTIONS
# ------------------------------------------------------------

y_pred = model_pipeline.predict(X_test)


# ------------------------------------------------------------
# 5.9 MODEL EVALUATION
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label="Returned"
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="Returned"
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="Returned"
)


print("\n====================================")
print("MODEL EVALUATION RESULTS")
print("====================================")

print("\nAccuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-Score :", round(f1, 4))


# ------------------------------------------------------------
# 5.10 CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Not Returned", "Returned"]
)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(7, 5))

plt.imshow(cm)

plt.title("Confusion Matrix of Random Forest Classifier")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    [0, 1],
    ["Not Returned", "Returned"]
)

plt.yticks(
    [0, 1],
    ["Not Returned", "Returned"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()
plt.show()


# ============================================================
# TOP 10 FEATURE IMPORTANCE
# ============================================================

# Get the fitted preprocessing component from the trained pipeline.
fitted_preprocessor = model_pipeline.named_steps["preprocessing"]

# Get the actual feature names after one-hot encoding.
feature_names = fitted_preprocessor.get_feature_names_out()

# Get the trained Random Forest classifier.
trained_rf_model = model_pipeline.named_steps["classifier"]

# Create feature-importance series.
feature_importance = pd.Series(
    trained_rf_model.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

# Remove preprocessing prefixes for a cleaner chart.
feature_importance.index = (
    feature_importance.index
    .str.replace("categorical__", "", regex=False)
    .str.replace("remainder__", "", regex=False)
)

# Select only the top 10 features for the report chart.
top_features = feature_importance.head(10)

print("\nTop 10 Feature Importance:")
print(top_features.round(6))

plt.figure(figsize=(10, 6))
top_features.sort_values().plot(kind="barh")

plt.title("Top 10 Feature Importance of Random Forest Model")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 5.12 FINAL MACHINE LEARNING RESULTS
# ------------------------------------------------------------

print("\n====================================")
print("MACHINE LEARNING COMPLETED")
print("====================================")

print("\nAccuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-Score :", round(f1, 4))

print("\nConfusion Matrix:")
print(cm)

print("\nMachine Learning analysis completed successfully!")