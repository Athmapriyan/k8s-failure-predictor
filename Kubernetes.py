import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt


# Load the trained model
try:
    model = load_model("k8s_failure_model.h5", compile=False)  # Load model without compiling
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Recompile with correct loss
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

# Preprocess test data
seq_length = 10
test_scaled = scaler.transform(test_data[['cpu_usage', 'memory_usage', 'pod_status', 'network_io', 'disk_usage']])

X_test = []
for i in range(len(test_scaled) - seq_length):
    X_test.append(test_scaled[i:i + seq_length])

X_test = np.array(X_test)

# Make predictions
predictions = model.predict(X_test)

# Inverse transform predictions
predicted_values = scaler.inverse_transform(predictions.reshape(-1, predictions.shape[2]))
predicted_df = pd.DataFrame(predicted_values, columns=['CPU Usage', 'Memory Usage', 'Pod Status', 'Network IO', 'Disk Usage'])

# Print sample predictions
print("\nSample Predictions:")
print(predicted_df.head(10))

# Scatter Plot of Predictions
def scatter_plot_predictions(Y_test, predictions, num_samples=50):
    predictions = predictions[:, -1, :]  # Shape: (num_samples, num_features)

    # Ensure correct shape for Y_test (actual values)
    Y_test = Y_test[-predictions.shape[0]:, :]  # Match prediction size

    # Inverse transform back to original scale
    actual_values = scaler.inverse_transform(Y_test)
    predicted_values = scaler.inverse_transform(predictions)

    # Check shapes to avoid errors
    print(f"Actual Values Shape: {actual_values.shape}")
    print(f"Predicted Values Shape: {predicted_values.shape}")

    plt.figure(figsize=(12, 8))
    metrics = ['CPU Usage', 'Memory Usage', 'Pod Status', 'Network IO', 'Disk Usage']
    
    for i, metric in enumerate(metrics):
        plt.subplot(3, 2, i + 1)
        scatter = plt.scatter(
            actual_values[:, i], predicted_values[:, i], 
            c=np.arange(len(actual_values)), cmap='coolwarm', 
            edgecolors='black', alpha=0.7, s=40, marker='o'
        )
        plt.colorbar(scatter, label='Index')  # Add colorbar for better visualization
        plt.xlabel(f'Actual {metric}')
        plt.ylabel(f'Predicted {metric}')
        plt.title(f'Predicted vs Actual {metric}')
    
    plt.tight_layout()
    plt.show()

# Plot scatter predictions
scatter_plot_predictions(test_scaled, predictions)



