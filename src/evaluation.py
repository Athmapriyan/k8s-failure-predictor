# Script for evaluating the model
import numpy as np
import pandas as pd
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(model, scaler, X_test, Y_test, feature_cols):
    # Make predictions
    predictions = model.predict(X_test)

    # Inverse transform predictions & actual values
    actual_values = scaler.inverse_transform(Y_test)
    predicted_values = scaler.inverse_transform(predictions[:, -1, :])

    # Compute Model Performance Metrics
    mae = mean_absolute_error(actual_values, predicted_values)
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)

    print(f"Model Performance:\nMAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}")

    # Feature Importance Analysis using SHAP
    explainer = shap.Explainer(model, masker=shap.maskers.Independent(X_test))
    shap_values = explainer(X_test[:100])  # Use a subset to save computation
    shap.summary_plot(shap_values, feature_names=feature_cols)

    # Scatter plot for predictions
    scatter_plot_predictions(actual_values, predicted_values, feature_cols)

    # Save predictions
    predicted_df = pd.DataFrame(predicted_values, columns=feature_cols)
    predicted_df.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")

def scatter_plot_predictions(actual_values, predicted_values, feature_cols):
    plt.figure(figsize=(12, 8))

    for i, metric in enumerate(feature_cols):
        plt.subplot(3, 2, i + 1)
        sns.scatterplot(x=actual_values[:, i], y=predicted_values[:, i], alpha=0.6, edgecolor='k')
        plt.xlabel(f'Actual {metric}')
        plt.ylabel(f'Predicted {metric}')
        plt.title(f'Predicted vs Actual {metric}')
        plt.axline((0, 0), slope=1, color='r', linestyle='--')  # Reference line
    
    plt.tight_layout()
    plt.show()
