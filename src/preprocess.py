"""
preprocess.py

Turns each dataset from build_dataset.py into features and splits it into
train and test sets so the test set only has domains the model hasn't seen.

The split is grouped by "campaign": the registered domain with every run of
digits replaced by "#". PhishTank often lists many URLs on one domain, and
phishing kits register many domains that differ only by a number
(pl-oferta48213.sbs, pl-oferta77120.sbs). Either one on both sides of the
split lets the model memorize a campaign instead of learning anything.

Writes data/processed/{level}_train.csv and {level}_test.csv.
"""

import os
import re

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


def campaign_key(domain: str) -> str:
    return re.sub(r"\d+", "#", domain)


def split_by_campaign(df: pd.DataFrame):
    groups = df["domain"].map(campaign_key)
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(splitter.split(df, df["label"], groups=groups))
    train, test = df.iloc[train_idx], df.iloc[test_idx]

    overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    assert not overlap, f"{len(overlap)} campaigns in both train and test"
    return train, test


def main():
    for level in LEVELS:
        path = os.path.join(PROCESSED_DIR, f"dataset_{level}.csv")
        if not os.path.exists(path):
            print(f"{path} not found, skipping. Run build_dataset.py first.")
            continue

        df = pd.read_csv(path)
        train, test = split_by_campaign(featurize(df, level))

        print(f"== {level} ==")
        for name, part in (("train", train), ("test", test)):
            counts = part["label"].value_counts().to_dict()
            print(f"  {name}: {len(part):,} rows (benign {counts.get(0, 0):,}, phishing {counts.get(1, 0):,})")
            part.to_csv(os.path.join(PROCESSED_DIR, f"{level}_{name}.csv"), index=False)


if __name__ == "__main__":
    main()
