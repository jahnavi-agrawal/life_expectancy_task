import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # finds my raw data 
data_path = os.path.join(BASE_DIR, "data", "train_data.csv")

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(path, target_col="life_expectancy", save_csv=True):
    df = load_data(path)
    print(f"Loaded dataset from {path}")

    # Clean column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )

    df = df.drop_duplicates()
    df = df.drop(columns=["year", "country"])
    df["status"] = df["status"].map({"Developing": 0.0, "Developed": 1.0})

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes

    # Standardize
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

    # Prepare features and labels
    X = df.drop(columns=[target_col]).values.astype(float)
    y = df[target_col].values.astype(float)
    print(f"Shapes: X_train {X.shape}, y_train {y.shape}")

    return X, y 

if __name__ == "__main__":
    preprocess_data(data_path)
