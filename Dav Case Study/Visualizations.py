import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file_path = r"D:\Dav Case Study\returns_sustainability_dataset.csv"
df = pd.read_csv(file_path)

# Convert date column
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# ============================================================
# 1. RETURN STATUS DISTRIBUTION
# ============================================================

return_counts = df["Return_Status"].value_counts()

plt.figure(figsize=(7, 7))
plt.pie(
    return_counts.values,
    labels=return_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Distribution of Returned and Non-Returned Orders")
plt.tight_layout()
plt.show()


# ============================================================
# 2. RETURN RATE BY PRODUCT CATEGORY
# ============================================================

category_return_rate = (
    df.groupby("Product_Category")["Return_Status"]
    .apply(lambda x: (x == "Returned").mean() * 100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 6))
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
# 3. MONTHLY RETURN RATE TREND
# ============================================================

df["Order_Year_Month"] = (
    df["Order_Date"]
    .dt.to_period("M")
    .astype(str)
)

monthly_return_rate = (
    df.groupby("Order_Year_Month")["Return_Status"]
    .apply(lambda x: (x == "Returned").mean() * 100)
)

plt.figure(figsize=(14, 6))
plt.plot(
    monthly_return_rate.index,
    monthly_return_rate.values,
    marker="o"
)
plt.title("Monthly Return Rate Trend")
plt.xlabel("Month")
plt.ylabel("Return Rate (%)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 4. RETURN REASONS DISTRIBUTION
# ============================================================

returned_orders = df[
    df["Return_Status"] == "Returned"
]

return_reason_counts = (
    returned_orders["Return_Reason"]
    .value_counts()
)

plt.figure(figsize=(9, 6))
plt.bar(
    return_reason_counts.index,
    return_reason_counts.values
)
plt.title("Distribution of Product Return Reasons")
plt.xlabel("Return Reason")
plt.ylabel("Number of Returned Orders")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# ============================================================
# 5. FINANCIAL IMPACT OF RETURNED ORDERS
# ============================================================

total_returned_order_value = (
    returned_orders["Order_Value"].sum()
)

total_return_cost = (
    returned_orders["Return_Cost"].sum()
)

financial_labels = [
    "Returned Order Value",
    "Return Cost"
]

financial_values = [
    total_returned_order_value,
    total_return_cost
]

plt.figure(figsize=(8, 6))
plt.bar(
    financial_labels,
    financial_values
)
plt.title("Financial Impact of Returned Orders")
plt.xlabel("Financial Measure")
plt.ylabel("Amount")
plt.tight_layout()
plt.show()