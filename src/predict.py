import argparse
import os
import pickle
import numpy as np
import pandas as pd
from data_preprocessing import preprocess_data
from train_model import add_bias, polynomial_features, evaluate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main(args):
    model_path = args.model_path or os.path.join(BASE_DIR, "models", "regression_model_final.pkl")
    data_path = args.data_path or os.path.join(BASE_DIR, "data", "train_data.csv")
    metrics_output_path = args.metrics_output_path or os.path.join(BASE_DIR, "results", "train_metrics.txt")
    predictions_output_path = args.predictions_output_path or os.path.join(BASE_DIR, "results", "train_predictions.csv")

    X, y = preprocess_data(data_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    if isinstance(model, tuple):
        theta, degree = model
        X = polynomial_features(X, degree)
    else:
        theta = model

    preds = add_bias(X) @ theta

    os.makedirs(os.path.dirname(predictions_output_path), exist_ok=True)
    pd.DataFrame(preds).to_csv(predictions_output_path, index=False, header=False)

    mse, rmse, r2 = evaluate(X, y, theta)

    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, "w") as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")
        f.write(f"R-squared (R²) Score: {r2:.2f}\n")

    print(f"Predictions in {predictions_output_path}")
    print(f"Metrics in {metrics_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved regression model")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--metrics_output_path", type=str, default=None)
    parser.add_argument("--predictions_output_path", type=str, default=None)

    args = parser.parse_args()
    main(args)