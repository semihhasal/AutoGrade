import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# ANN Modeli ve Ölçekleyiciyi Eğitmek İçin Fonksiyon
def train_ann_model():
    # Eğitim verisini oku ve işle
    train_data_path = 'train_data.csv'
    train_df = pd.read_csv(train_data_path, delimiter=';', header=None, skiprows=1,
                           names=['ORB', 'SSIM', 'VGG16', 'GRADED'])

    # Özellikleri normalize et
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_df[['ORB', 'SSIM', 'VGG16']])
    y_train = train_df['GRADED'].values

    # Modeli tanımla ve derle
    model = Sequential([
        Dense(10, input_dim=3, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    # Erken durdurma kullanarak modeli eğit
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=200, validation_split=0.3, callbacks=[early_stopping], verbose=1)

    return model, scaler


# ANN Modelini Kullanarak Tahmin Yapmak İçin Fonksiyon
def evaluate_ann_model(model, scaler, orb, ssim, vgg16):
    # Özellikleri normalize et
    features = np.array([[orb, ssim, vgg16]])
    features = scaler.transform(features)

    # Tahmin yap
    predicted_grade = model.predict(features)[0][0]

    return predicted_grade


# ANN modelini eğit
ann_model, ann_scaler = train_ann_model()