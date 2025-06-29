# data_cleaning.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

print("✅ Script Started")

# Load the dataset
try:
    df = pd.read_csv("titanic.csv")
    print("✅ Titanic dataset loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: 'titanic.csv' not found in the current directory.")
    exit()

# 1. Explore basic info
print("\n🔹 First 5 rows:")
print(df.head())

print("\n🔹 Dataset Info:")
print(df.info())

print("\n🔹 Summary Statistics:")
print(df.describe())

# 2. Handle missing values
print("\n🔹 Missing Values Before:")
print(df.isnull().sum())

# Fill missing 'Age' with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill 'Embarked' with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' due to many missing values
if 'Cabin' in df.columns:
    df.drop('Cabin', axis=1, inplace=True)

print("\n🔹 Missing Values After:")
print(df.isnull().sum())

# 3. Encode categorical features
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode 'Embarked'
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# 4. Normalize numerical features
scaler = StandardScaler()
num_cols = ['Age', 'Fare']
df[num_cols] = scaler.fit_transform(df[num_cols])

# 5. Visualize outliers using boxplots
plt.figure(figsize=(10, 4))
for i, col in enumerate(num_cols):
    plt.subplot(1, 2, i+1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.savefig("outlier_boxplots.png")  # Save plot as image instead of showing
print("\n📊 Boxplots saved as 'outlier_boxplots.png'")

# Final output
print("\n✅ Final Cleaned Dataset Columns:")
print(df.columns)

print("\n✅ Cleaned Data Sample:")
print(df.head())

# Optional: save cleaned data
df.to_csv("titanic_cleaned.csv", index=False)
print("\n💾 Cleaned data saved as 'titanic_cleaned.csv'")

input("\n✅ Script finished. Press Enter to exit...")
