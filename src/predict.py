import argparse
import os
import pickle
import numpy as np
import pandas as pd
from data_preprocessing import preprocess_data
from train_model import add_bias, polynomial_features, evaluate, evaluate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def evaluate(y_true, y_pred): # writing one without theta for ease
    """Evaluate regression metrics directly on predictions"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return mse, rmse, r2


def main(args):
    model_path = args.model_path or os.path.join(BASE_DIR, "models", "regression_model_final.pkl")
    data_path = args.data_path or os.path.join(BASE_DIR, "data", "train_data.csv")
    metrics_output_path = args.metrics_output_path or os.path.join(BASE_DIR, "results", "train_metrics.txt")
    predictions_output_path = args.predictions_output_path or os.path.join(BASE_DIR, "results", "train_predictions.csv")

    X_train, X_test, y_train, y_test = preprocess_data(data_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    if isinstance(model, tuple):
        theta, degree = model
        X_test = polynomial_features(X_test, degree)
    else:
        theta = model
    
    X_b = add_bias(X_test)
    y_pred = X_b @ theta

    # Save predictions
    pd.DataFrame(y_pred).to_csv(args.predictions_output_path, index=False, header=False)

    mse, rmse, r2 = evaluate(y_test, y_pred)

    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, "w") as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")
        f.write(f"R-squared (R^2) Score: {r2:.2f}\n")


    print(f"Predictions in {predictions_output_path}")
    print(f"Metrics in {metrics_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved regression model")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--metrics_output_path", type=str)
    parser.add_argument("--predictions_output_path", type=str)

    args = parser.parse_args()
    main(args)