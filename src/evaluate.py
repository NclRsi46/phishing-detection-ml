"""
evaluate.py

Evaluates the trained phishing detection model.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_PATH = "models/phishing_model.pkl"
PROCESSED_DATA = "data/processed/phishing_processed.csv"
TARGET_COL = "label"

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found.")
    if not os.path.exists(PROCESSED_DATA):
        raise FileNotFoundError("Processed data not found.")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(PROCESSED_DATA)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Evaluating model...")
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    main()
