import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # finds my raw data 
data_path = os.path.join(BASE_DIR, "data", "train_data.csv")


def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def preprocess_data(data_path, target_col="Life_expectancy", save_csv=True):
    df = load_data(data_path)
    print(f"Loaded dataset from {data_path}")

    # Clean column names
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )

    if "Country" in df.columns:
        df = df.drop(columns=["Country"])

    df = df.drop_duplicates()

    # Encode 
    if "Status" in df.columns:
        df["Status"] = df["Status"].map({"Developing": 0, "Developed": 1})

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes

    # Standardize
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in [target_col, "Year"]:
        if col in numeric_cols:
            numeric_cols.remove(col)
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

    # Separate features and target
    X = df.drop(columns=[target_col, "Year"], errors="ignore").values.astype(float)
    y = df[target_col].values.astype(float)

    # Train-test split
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split = int(0.8 * len(X))
    train_i, test_i = indices[:split], indices[split:]

    X_train, X_test = X[train_i], X[test_i]
    y_train, y_test = y[train_i], y[test_i]

    print(f"Shapes: X_train {X_train.shape}, X_test {X_test.shape}")

    if save_csv:
        df.to_csv(data_path, index=False)
        print(f"Stored preprocessed data in: {data_path}")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data(data_path)
