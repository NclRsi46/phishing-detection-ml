"""
predict.py

Predicts whether a given URL is phishing or legitimate.
"""

import os
import sys
import joblib
import pandas as pd

MODEL_PATH = "models/phishing_model.pkl"

def extract_features_from_url(url: str) -> pd.DataFrame:
    """
    Placeholder for feature extraction.
    """
    features = {}
    return pd.DataFrame([features])

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <url>")
        sys.exit(1)

    url = sys.argv[1]
    print(f"Predicting URL: {url}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found.")

    model = joblib.load(MODEL_PATH)
    features = extract_features_from_url(url)

    if features.empty:
        print("Feature extraction not implemented yet.")
        sys.exit(1)

    prediction = model.predict(features)[0]
    label = "phishing" if prediction == 1 else "legitimate"
    print(f"Prediction: {label}")

if __name__ == "__main__":
    main()
