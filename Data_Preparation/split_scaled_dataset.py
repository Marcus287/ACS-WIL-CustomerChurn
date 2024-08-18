import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

file_path = 'raw_data.csv'
dataset = pd.read_csv(file_path)

dataset.info()

# Categorical to Number
label_encoder = LabelEncoder()
columns_to_encode = ['gender', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'Contract', 'Churn']
for column in columns_to_encode:
    dataset_cleaned[column] = label_encoder.fit_transform(dataset_cleaned[column])


X = dataset_cleaned.drop(columns=['Churn'])
y = dataset_cleaned['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
