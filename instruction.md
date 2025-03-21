# Kubernetes Failure Prediction

This repository contains a machine learning model to predict Kubernetes pod failures based on system metrics such as CPU usage, memory usage, network I/O, and disk usage.

## Files Included

1. **`Kubernetes.py`** – Python script for loading the model, making predictions, and visualizing results.
2. **`k8s_failure_model.h5`** – Pre-trained TensorFlow model.
3. **`scaler.pkl`** – MinMaxScaler object for normalizing data.
4. **`k8s_test_data.csv`** – Sample test data.
5. **`requirements.txt`** – List of dependencies.
6. **`README.md`** – Instructions and setup guide.

## Installation & Setup

### Prerequisites
- Python 3.8 or later
- pip (Python package manager)

### Installation Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/kubernetes-failure-prediction.git
   cd kubernetes-failure-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the model and scaler files are present in the project directory.

## Running the Prediction Script

Execute the script using Python:
```bash
python Kubernetes.py
```

## Expected Output
- The model will load and predict system metrics based on test data.
- A scatter plot will be displayed comparing actual vs. predicted values.

## Troubleshooting
- If you get a `ModuleNotFoundError`, install missing packages using `pip install <package-name>`.
- If the model fails to load, ensure `k8s_failure_model.h5` is in the correct directory.

## License
This project is open-source and available under the MIT License.
