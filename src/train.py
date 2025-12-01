"""
train.py

Trains a machine-learning model on the processed phishing dataset.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

PROCESSED_DATA = "data/processed/phishing_processed.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_model.pkl")
TARGET_COL = "label"  # from preprocess/build_dataset


def main():
    if not os.path.exists(PROCESSED_DATA):
        raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA}.")

    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing from dataset. Columns: {df.columns.tolist()}")

    # Split into features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # 👉 VERY IMPORTANT: keep only numeric columns
    X = X.select_dtypes(include="number")

    print(f"Using feature columns: {X.columns.tolist()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model on test set...")
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
