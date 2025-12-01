"""
evaluate.py

Loads the trained phishing detection model and evaluates it on the full
processed dataset (data/processed/phishing_processed.csv).

This should mirror the same feature handling as train.py:
- Uses only numeric feature columns
- Target column: 'label'
"""

import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report

MODEL_PATH = "models/phishing_model.pkl"
PROCESSED_DATA = "data/processed/phishing_processed.csv"
TARGET_COL = "label"  # same as in train.py


def main():
    # Check that model and data exist
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")

    if not os.path.exists(PROCESSED_DATA):
        raise FileNotFoundError(f"Processed dataset not found at {PROCESSED_DATA}. Run preprocess.py first.")

    print("Loading model and processed dataset...")
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(PROCESSED_DATA)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' missing from dataset. "
            f"Columns: {df.columns.tolist()}"
        )

    # Split features and labels
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Use only numeric feature columns (same as train.py)
    X = X.select_dtypes(include="number")
    print(f"Using feature columns for evaluation: {X.columns.tolist()}")

    print("Evaluating model on the full processed dataset...")
    preds = model.predict(X)
    print(classification_report(y, preds))


if __name__ == "__main__":
    main()
