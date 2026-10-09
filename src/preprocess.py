"""
preprocess.py

Turns each dataset from build_dataset.py into features and splits it into
train and test sets, grouped by registered domain so no domain appears in
both. PhishTank often lists many URLs on one domain; a random row split
would put the same domain on both sides and inflate the test score.

Writes data/processed/{level}_train.csv and {level}_test.csv.
"""

import os

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from features import extract_features, feature_names

PROCESSED_DIR = "data/processed"
LEVELS = ["domain", "host"]
TEST_SIZE = 0.2
SEED = 42


def featurize(df: pd.DataFrame, level: str) -> pd.DataFrame:
    rows = [extract_features(url, level) for url in df["url"]]
    features = pd.DataFrame(rows, columns=feature_names(level), index=df.index)
    return pd.concat([df[["domain", "host"]], features, df[["label"]]], axis=1)


def split_by_domain(df: pd.DataFrame):
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(splitter.split(df, df["label"], groups=df["domain"]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]

    overlap = set(train["domain"]) & set(test["domain"])
    assert not overlap, f"{len(overlap)} domains in both train and test"
    return train, test


def main():
    for level in LEVELS:
        path = os.path.join(PROCESSED_DIR, f"dataset_{level}.csv")
        if not os.path.exists(path):
            print(f"{path} not found, skipping. Run build_dataset.py first.")
            continue

        df = pd.read_csv(path)
        train, test = split_by_domain(featurize(df, level))

        print(f"== {level} ==")
        for name, part in (("train", train), ("test", test)):
            counts = part["label"].value_counts().to_dict()
            print(f"  {name}: {len(part):,} rows (benign {counts.get(0, 0):,}, phishing {counts.get(1, 0):,})")
            part.to_csv(os.path.join(PROCESSED_DIR, f"{level}_{name}.csv"), index=False)


if __name__ == "__main__":
    main()
