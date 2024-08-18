import pandas as pd
import numpy as np

dataset = pd.read_csv(file_path)

print(dataset.info())

duplicates = dataset.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

dataset_cleaned = dataset.drop_duplicates()
print(f"Shape of the dataset after removing duplicates: {dataset_cleaned.shape}")

# Convert Yes or No (Binary value) into numeric value of 0 and 1
dataset_cleaned['Dependents'] = dataset_cleaned['Dependents'].map({'Yes': 1, 'No': 0}).astype(int)
dataset_cleaned['PhoneService'] = dataset_cleaned['PhoneService'].map({'Yes': 1, 'No': 0}).astype(int)
dataset_cleaned['MultipleLines'] = dataset_cleaned['MultipleLines'].map({'Yes': 1, 'No': 0}).astype(int)
dataset_cleaned['Churn'] = dataset_cleaned['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

print(dataset_cleaned.head())

