"""
predict.py

Command-line tool to predict whether a given URL is phishing or legitimate
using the trained Random Forest model.

Usage:
    python src/predict.py "https://example.com"
"""

import os
import sys
from typing import Dict

import joblib
import pandas as pd

MODEL_PATH = "models/phishing_model.pkl"


def extract_features_from_url(url: str) -> Dict[str, int]:
    """
    Extract the same URL-based features used during training.
    These must match the features in preprocess.py.
    """
    if url is None:
        url = ""

    url_str = str(url)

    features = {
        "url_length": len(url_str),
        "num_dots": url_str.count("."),
        "num_hyphens": url_str.count("-"),
        "num_slashes": url_str.count("/"),
        "num_question_marks": url_str.count("?"),
        "num_equals": url_str.count("="),
        "has_at_symbol": 1 if "@" in url_str else 0,
        "uses_https": 1 if url_str.lower().startswith("https") else 0,
    }

    return features


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at '{path}'. "
            f"Make sure you have run src/train.py first."
        )
    return joblib.load(path)


def main():
    # Expect exactly one argument: the URL
    if len(sys.argv) != 2:
        print("Usage:")
        print('  python src/predict.py "https://example.com"')
        sys.exit(1)

    url = sys.argv[1]

    # Load trained model
    model = load_model(MODEL_PATH)

    # Extract features
    features = extract_features_from_url(url)
    X = pd.DataFrame([features])  # single-row DataFrame

    # Predict
    pred = model.predict(X)[0]

    # Try to get probability if supported
    proba_str = ""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        prob_legit = proba[0]  # class 0
        prob_phish = proba[1]  # class 1
        proba_str = (
            "\nEstimated probabilities:\n"
            f"  Legitimate (0): {prob_legit:.3f}\n"
            f"  Phishing   (1): {prob_phish:.3f}"
        )

    label_map = {0: "Legitimate (0)", 1: "Phishing (1)"}
    label_text = label_map.get(pred, str(pred))

    print(f"URL: {url}")
    print(f"Prediction: {label_text}{proba_str}")


if __name__ == "__main__":
    main()
