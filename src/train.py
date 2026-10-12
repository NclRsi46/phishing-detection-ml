"""
train.py

Trains three models on each training split from preprocess.py: a Logistic
Regression baseline and a Random Forest on the hand-built features, and a
Logistic Regression on character n-grams of the scored name itself, to see
what the hand-built features miss. Each model is saved together with its
feature level and input columns, so predict.py builds exactly the same input.

Writes models/{level}_{model}.pkl.
"""

import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import feature_names

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
LEVELS = ["domain", "host"]
SEED = 42
NGRAM_MODEL = "char_ngram_logistic_regression"


def make_models(level: str) -> dict:
    return {
        # Features are on different scales (length vs ratios), so the linear
        # model needs scaling. The forest doesn't.
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        # One job: in parallel, the trees' scores are summed in a different
        # order on each call, so the same row's score can differ in the last
        # bits between runs. The trees themselves are the same either way.
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1),
        # Reads the scored name, suffix included ("paypal-login-secure.com"),
        # instead of the hand-built features. The settings were fixed before
        # the first run, not tuned. min_df=2 drops n-grams found in only one
        # training name, which could only memorize that name. TF-IDF rows are
        # already normalized, so there is no scaler. The column is named as a
        # string, not a list, so the vectorizer gets a 1-D column of names.
        NGRAM_MODEL: make_pipeline(
            ColumnTransformer([("ngrams", TfidfVectorizer(analyzer="char", ngram_range=(1, 5), min_df=2), level)]),
            LogisticRegression(max_iter=1000),
        ),
    }


def model_columns(name: str, level: str) -> list:
    """Input columns: the "domain" or "host" column for the n-gram model,
    the hand-built features for the others."""
    return [level] if name == NGRAM_MODEL else feature_names(level)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    for level in LEVELS:
        path = os.path.join(PROCESSED_DIR, f"{level}_train.csv")
        if not os.path.exists(path):
            print(f"{path} not found, skipping. Run preprocess.py first.")
            continue

        train = pd.read_csv(path)

        for name, model in make_models(level).items():
            columns = model_columns(name, level)
            model.fit(train[columns], train["label"])
            out = os.path.join(MODEL_DIR, f"{level}_{name}.pkl")
            joblib.dump({"model": model, "level": level, "features": columns}, out)
            print(f"Trained {name} on {len(train):,} {level} rows -> {out}")


if __name__ == "__main__":
    main()
