import matplotlib.pylot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
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

def vis_learning_curve(history):
  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(1, len(accuracy) + 1)

  # Plot training and validation accuracy
  plt.figure(figsize=(12, 6))

  # Plot accuracy
  plt.subplot(1, 2, 1)
  plt.plot(epochs, accuracy, 'b', label='Training accuracy')
  plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
  plt.title('Training and Validation Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()

  # Plot loss
  plt.subplot(1, 2, 2)
  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title('Training and Validation Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  plt.show()

file_path = 'cleaned_data.csv'

print("Read processed files, build models and testing)

#Split Testing and Training Data
df = pd.read_csv(file_path)
X = df.drop('Churn',axis=1).to_numpy()
Y = df['Churn'].to_numpy()

#Data processing with Standardize and Normalization
scaler = MinMaxScaler()
X_processed = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_processed, Y, test_size= 0.3, random_state=42)

print(f'number of training size is {X_train.shape}, and testing size is {X_test.shape} ')

# Build Model
inputs = layers.Input(shape = (X.shape[1],))

x = layers.Dense(32, activation='relu')(inputs)
x = layers.Dense(16, activation='relu')(x)
x = layers.Dense(8, activation='relu')(x)
outputs = layers.Dense(units = 1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs, name = 'ANN_Model')


model.summary()

# Train Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience= 5, restore_best_weights=True)
history = model.fit(X_train, Y_train, validation_split=0.2, epochs=30, batch_size=64, verbose = 1, callbacks=[early_stopping])

# Test Model

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
