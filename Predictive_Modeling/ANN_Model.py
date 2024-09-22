import math
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve



file_path = 'cleaned_data.csv'

df = pd.read_csv(file_path)
df.head()
X = df.drop('Churn',axis=1).to_numpy()
Y = df['Churn'].to_numpy()

#Data processing with Standardize and Normalization
scaler = MinMaxScaler()
X_processed = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_processed, Y, test_size= 0.3, random_state=42)

print(f'number of training size is {X_train.shape}, and testing size is {X_test.shape} ')

inputs = layers.Input(shape = (X.shape[1],))

x = layers.Dense(32, activation='relu')(inputs)
x = layers.Dense(16, activation='relu')(x)
x = layers.Dense(8, activation='relu')(x)
outputs = layers.Dense(units = 1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs, name = 'ANN_Model')


model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience= 5, restore_best_weights=True)
history = model.fit(X_train, Y_train, validation_split=0.2, epochs=30, batch_size=64, verbose = 1, callbacks=[early_stopping])

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob >= 0.5).astype(int)

accuracy = accuracy_score(Y_test, y_pred)
conf_matrix = confusion_matrix(Y_test, y_pred)
class_report = classification_report(Y_test, y_pred)
roc_auc = roc_auc_score(Y_test, y_pred_prob)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrx:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')
print(f'ROC-AUC Score: {roc_auc}')
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC={roc_auc:.2f})')
plt.plot([0,1],[0,1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()