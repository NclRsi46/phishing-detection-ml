"""
predict.py

Command-line tool that scores a URL with a trained model. Only the domain
name is scored; the path and query are ignored, same as in training.

Usage:
    python src/predict.py "https://example.com/login"
    python src/predict.py --model models/host_random_forest.pkl "https://example.com"
"""

import argparse
import os
import sys

import joblib
import pandas as pd

from features import extract_features, hostname, registered_domain

DEFAULT_MODEL = "models/domain_random_forest.pkl"


def load_model(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at '{path}'. Run src/train.py first.")
    return joblib.load(path)


def score(bundle: dict, url: str) -> float:
    """Probability that the URL is phishing."""
    # Column order comes from the saved model, not from this file, so a
    # change to features.py can't silently reorder the model's inputs.
    # The n-gram model reads the name itself, built the same way as the
    # "domain" and "host" columns in build_dataset.py.
    features = extract_features(url, bundle["level"])
    row = {**features, "domain": registered_domain(url), "host": hostname(url)}
    X = pd.DataFrame([row], columns=bundle["features"])
    return float(bundle["model"].predict_proba(X)[0][1])


def main():
    parser = argparse.ArgumentParser(description="Score a URL as phishing or legitimate.")
    parser.add_argument("url")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    bundle = load_model(args.model)
    level = bundle["level"]
    scored = registered_domain(args.url) if level == "domain" else hostname(args.url)
    if not scored:
        print(f"Could not find a valid domain in: {args.url}")
        sys.exit(1)

    p = score(bundle, args.url)
    print(f"URL:          {args.url}")
    print(f"Scored:       {scored} ({level} level, path and query ignored)")
    print(f"P(phishing):  {p:.3f}")
    print(f"Prediction:   {'Phishing' if p > 0.5 else 'Legitimate'}")


if __name__ == "__main__":
    main()
