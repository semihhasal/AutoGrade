import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Set seeds for reproducibility
np.random.seed(20)
random.seed(20)
tf.random.set_seed(20)

# Read and preprocess training data
train_data_path = 'train_data.csv'  # Update with the correct path
train_df = pd.read_csv(train_data_path, delimiter=';', header=None, skiprows=1,
                       names=['ORB', 'SSIM', 'VGG16', 'GRADED'])

# Normalize features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_df[['ORB', 'SSIM', 'VGG16']])
y_train = train_df['GRADED'].values

# Define cross-validation parameters
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=20)

# Cross-validation loop
val_scores = []
for train_index, val_index in kf.split(X_train):
    X_train_kfold, X_val_kfold = X_train[train_index], X_train[val_index]
    y_train_kfold, y_val_kfold = y_train[train_index], y_train[val_index]

    # Define and compile the model
    model = Sequential([
        Dense(10, input_dim=3, activation='relu'),  # Input layer with 3 inputs, hidden layer with 24 neurons
        Dense(10, activation='relu'),  # Second hidden layer
        Dense(1, activation='linear')  # Output layer with a single output (grade)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    # Use early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    model.fit(X_train_kfold, y_train_kfold, epochs=200, validation_data=(X_val_kfold, y_val_kfold),
              callbacks=[early_stopping], verbose=0)

    # Evaluate the model on the validation data
    val_loss, val_mae = model.evaluate(X_val_kfold, y_val_kfold, verbose=0)
    val_scores.append(val_mae)

# Report the average performance across all folds
average_mae = np.mean(val_scores)
print(f"Average Validation MAE: {average_mae}")
