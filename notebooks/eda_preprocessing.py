import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
data_path = os.path.join(os.path.dirname(__file__), "../data/bank_data_C.csv")
df = pd.read_csv(data_path)

# Date conversion
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')
df['CustomerDOB'] = pd.to_datetime(df['CustomerDOB'], errors='coerce')

# create age and create a new column for age
def calculate_age(df):
    df["age"] = df["TransactionDate"].dt.year - df["CustomerDOB"].dt.year
    return df
df = calculate_age(df)

def adjust_year(date):
    if pd.notnull(date) and date.year > 2016:
        return date.replace(year=date.year - 100)
    return date
df["CustomerDOB"] = df["CustomerDOB"].apply(adjust_year)
df = calculate_age(df)

## Define functions to replace outliers in ages 
def replace_age_outlier(df):
    DOB_threshold = 1900

    age_outliers = df[df["CustomerDOB"].dt.year < DOB_threshold].index

    mean_DOB = df.loc[~df.index.isin(age_outliers), "CustomerDOB"].mean()

    df.loc[age_outliers, "CustomerDOB"] = mean_DOB

    return df
df = replace_age_outlier(df)

# Recalculate age after fixing DOBs
df = calculate_age(df)

#Replace T in Gender with M
df["CustGender"] = df["CustGender"].replace('T','M')

#Drop all customers with 0 transaction amount 
df.drop(df[df["TransactionAmount (INR)"] == 0].index, axis=0, inplace=True)

# Plotting Age Distribution
plt.figure(figsize=(12, 8))
sns.histplot(df["age"], bins=10, kde=False)
plt.xlabel("Age")           # <-- corrected from zlabel to xlabel
plt.ylabel("Frequency")     
plt.title("Age Distribution")
plt.show()

#Plot a distribution for data accross the unique transaction date
plt.figure(figsize=(12,8))
sns.histplot(df["TransactionDate"],bins=3,kde=False)
plt.xlabel("Transaction date")
plt.ylabel("frequency")
plt.title("transaction date distribution")
plt.show()

#Create a pie chart for gender frequency
plt.figure(figsize=(6,6))
gender_count = df["CustGender"].value_counts()
plt.pie(gender_count,labels = gender_count.index, autopct="%1.1f%%", startangle=180)
plt.title("Pie chart of Gender")
plt.show()

# Save cleaned data to a CSV file
output_path = os.path.join(os.path.dirname(__file__), "../output/cleaned_data.csv")
df.to_csv(output_path, index=False)