# Main script for model execution
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from model_training import train_model
from evaluation import evaluate_model

# Load the trained model
try:
    model = load_model("k8s_failure_model.h5", compile=False)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load the scaler
try:
    scaler = joblib.load("scaler.pkl")
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")
    exit()

# Load test data
try:
    test_data = pd.read_csv("k8s_test_data.csv")
    print("Test data loaded successfully.")
except Exception as e:
    print(f"Error loading test data: {e}")
    exit()

# Define feature columns and preprocess test data
seq_length = 10
feature_cols = ['cpu_usage', 'memory_usage', 'pod_status', 'network_io', 'disk_usage']
test_scaled = scaler.transform(test_data[feature_cols])

X_test = []
Y_test = []
for i in range(len(test_scaled) - seq_length):
    X_test.append(test_scaled[i:i + seq_length])
    Y_test.append(test_scaled[i + seq_length])

X_test = np.array(X_test)
Y_test = np.array(Y_test)

# Train the model (if required)
# train_model()  # Uncomment if you want to retrain the model

# Evaluate the model
evaluate_model(model, scaler, X_test, Y_test, feature_cols)
