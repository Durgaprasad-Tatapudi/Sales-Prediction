# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
file_path = "C:/Users/durga/.cache/kagglehub/datasets/yashpaloswal/ann-car-sales-price-prediction/versions/1/car_purchasing.csv"

df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Rename columns for consistency
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Drop unnecessary columns
df.drop(columns=["customer_name", "customer_e-mail", "country"], errors="ignore", inplace=True)

# Convert numeric columns to correct format
numeric_cols = ["age", "annual_salary", "credit_card_debt", "net_worth", "car_purchase_amount"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# Fix Warning: Correct way to handle missing values in "gender"
if "gender" in df.columns:
    df.loc[:, "gender"] = df["gender"].fillna(df["gender"].mode()[0])  # Using loc to avoid chained assignment

# Fill remaining NaN values with 0
df.fillna(0, inplace=True)

# Remove zero-variance columns
zero_variance_cols = [col for col in df.columns if df[col].nunique() == 1]
if zero_variance_cols:
    df.drop(columns=zero_variance_cols, inplace=True)

# Ensure target variable exists
if "car_purchase_amount" not in df.columns:
    raise KeyError("Column 'car_purchase_amount' not found!")

# Print dataset summary
print("\n✅ Dataset Loaded Successfully")
print(df.describe())

# Data Visualization
sns.set(style="whitegrid")

# Histogram of Car Purchase Amount
plt.figure(figsize=(8, 5))
sns.histplot(df["car_purchase_amount"], bins=30, kde=True, color="blue")
plt.title("Distribution of Car Purchase Amount")
plt.xlabel("Car Purchase Amount ($)")
plt.ylabel("Frequency")
plt.show()

# Pairplot of Numerical Features
plt.figure(figsize=(12, 8))
sns.pairplot(df[numeric_cols])
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Scatter Plot: Annual Salary vs Car Purchase Amount
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["annual_salary"], y=df["car_purchase_amount"], hue=df["gender"], alpha=0.7)
plt.title("Annual Salary vs Car Purchase Amount")
plt.xlabel("Annual Salary ($)")
plt.ylabel("Car Purchase Amount ($)")
plt.show()

# Box Plot: Age vs Car Purchase Amount
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["age"], y=df["car_purchase_amount"])
plt.title("Age vs Car Purchase Amount")
plt.xlabel("Age")
plt.ylabel("Car Purchase Amount ($)")
plt.xticks(rotation=45)
plt.show()

# Define features and target variable
X = df.drop(columns=["car_purchase_amount"])
y = df["car_purchase_amount"]

# Convert categorical gender column into numeric
X = pd.get_dummies(X, columns=["gender"], drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n🔹 Model Performance Metrics:")
print(f"✅ Mean Absolute Error (MAE): {mae:.2f}")
print(f"✅ Mean Squared Error (MSE): {mse:.2f}")
print(f"✅ Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"✅ R Squared Score: {r2:.2f}")

# Scatter Plot of Actual vs Predicted Values
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
plt.xlabel("Actual Car Purchase Amount ($)")
plt.ylabel("Predicted Car Purchase Amount ($)")
plt.title("Actual vs Predicted Car Purchase Amount")
plt.show()

# Residual Plot
plt.figure(figsize=(8, 5))
sns.residplot(x=y_test, y=y_pred, color="purple")
plt.xlabel("Actual Car Purchase Amount ($)")
plt.ylabel("Residuals")
plt.title("Residual Plot of Predictions")
plt.show()
