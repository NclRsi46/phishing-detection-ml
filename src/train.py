"""
train.py

Trains a Logistic Regression baseline and a Random Forest on each training
split from preprocess.py. Each model is saved together with its feature
level and column order, so predict.py builds exactly the same input.

Writes models/{level}_{model}.pkl.
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import feature_names

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
LEVELS = ["domain", "host"]
SEED = 42


def make_models() -> dict:
    return {
        # Features are on different scales (length vs ratios), so the linear
        # model needs scaling. The forest doesn't.
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    for level in LEVELS:
        path = os.path.join(PROCESSED_DIR, f"{level}_train.csv")
        if not os.path.exists(path):
            print(f"{path} not found, skipping. Run preprocess.py first.")
            continue

        train = pd.read_csv(path)
        columns = feature_names(level)

        for name, model in make_models().items():
            model.fit(train[columns], train["label"])
            out = os.path.join(MODEL_DIR, f"{level}_{name}.pkl")
            joblib.dump({"model": model, "level": level, "features": columns}, out)
            print(f"Trained {name} on {len(train):,} {level} rows -> {out}")


if __name__ == "__main__":
    main()
