# Script for training the model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
import numpy as np
import pandas as pd

def train_model():
    try:
        data = pd.read_csv("k8s_train_data.csv")
        print("Training data loaded successfully.")
    except Exception as e:
        print(f"Error loading training data: {e}")
        return

    # Preprocess training data
    seq_length = 10
    feature_cols = ['cpu_usage', 'memory_usage', 'pod_status', 'network_io', 'disk_usage']
    scaler = joblib.load("scaler.pkl")  # Load pre-saved scaler
    data_scaled = scaler.transform(data[feature_cols])

    X_train = []
    Y_train = []
    for i in range(len(data_scaled) - seq_length):
        X_train.append(data_scaled[i:i + seq_length])
        Y_train.append(data_scaled[i + seq_length])

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    # Define model architecture
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, len(feature_cols))),
        LSTM(50, activation='relu', return_sequences=True),
        Dense(len(feature_cols))
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    model.fit(X_train, Y_train, epochs=10, batch_size=16, validation_split=0.1)

    # Save trained model
    model.save("k8s_failure_model.h5")
    print("Model training complete and saved.")

# Uncomment below to train when executing
# train_model()
