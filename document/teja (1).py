# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("rainfall_india.csv")

# Display first 5 rows
print(df.head())

# Check dataset information
print(df.info())

# Check missing values
print(df.isnull().sum())

# Fill missing values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# -----------------------------
# 1. Year-wise Rainfall Trend
# -----------------------------
yearly_rainfall = df.groupby("YEAR")["ANNUAL"].mean()

plt.figure(figsize=(10,5))
plt.plot(yearly_rainfall)
plt.title("Year-wise Average Rainfall in India")
plt.xlabel("Year")
plt.ylabel("Annual Rainfall (mm)")
plt.show()

# -----------------------------
# 2. State-wise Average Rainfall
# -----------------------------
state_rainfall = df.groupby("SUBDIVISION")["ANNUAL"].mean().sort_values()

plt.figure(figsize=(10,8))
state_rainfall.plot(kind='barh')
plt.title("State-wise Average Annual Rainfall")
plt.xlabel("Rainfall (mm)")
plt.show()

# -----------------------------
# 3. Monthly Rainfall Distribution
# -----------------------------
monthly_cols = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]

monthly_avg = df[monthly_cols].mean()

plt.figure(figsize=(10,5))
monthly_avg.plot(kind='bar')
plt.title("Average Monthly Rainfall in India")
plt.ylabel("Rainfall (mm)")
plt.show()

# -----------------------------
# 4. Correlation Heatmap
# -----------------------------
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Between Rainfall Variables")
plt.show()

# -----------------------------
# 5. Drought Year Detection
# -----------------------------
mean_rainfall = yearly_rainfall.mean()
std_rainfall = yearly_rainfall.std()

drought_years = yearly_rainfall[yearly_rainfall < (mean_rainfall - std_rainfall)]
print("Drought Years:")
print(drought_years)
