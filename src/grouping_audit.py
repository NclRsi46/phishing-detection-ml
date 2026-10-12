"""
grouping_audit.py

Compares the campaign grouping in preprocess.py (campaign_group: registered
domain without its public suffix, digit runs as "#") with the older key that
kept the suffix (campaign_key on the full registered domain), so the README
can say what the change merged and how much it closed:

    groups        how many campaigns each key makes, how many old campaigns
                  merged, and the rows in them
    largest       the biggest new campaigns, to catch generic names that
                  merge unrelated domains
    mixed label   campaigns that hold both phishing and benign rows
    TLD siblings  over the old key's 10 random splits (same seeds as
                  repeat_splits.py), the share of test phishing whose name
                  without the suffix is also a training phishing name: the
                  rows the old split let through and the new one doesn't

Writes results/grouping.json.

Usage:
    python src/grouping_audit.py
"""

import json
import os

import numpy as np
import pandas as pd

from preprocess import LEVELS, PROCESSED_DIR, campaign_group, campaign_key
from repeat_splits import RESULTS_DIR, random_splits

LARGEST = 5


def mixed_label(labels: pd.Series, keys: pd.Series) -> dict:
    by_key = labels.groupby(keys).agg(["min", "max", "size"])
    mixed = by_key[(by_key["min"] == 0) & (by_key["max"] == 1)]
    return {"groups": len(mixed), "rows": int(mixed["size"].sum())}


def merged(labels: pd.Series, old: pd.Series, new: pd.Series) -> dict:
    """New groups made of more than one old group."""
    old_per_new = old.groupby(new).nunique()
    keys = old_per_new.index[old_per_new > 1]
    rows = new.isin(keys)
    return {
        "groups": len(keys),
        "old_groups": int(old_per_new[keys].sum()),
        "phishing_rows": int((labels[rows] == 1).sum()),
        "benign_rows": int((labels[rows] == 0).sum()),
    }


def largest(labels: pd.Series, old: pd.Series, new: pd.Series) -> dict:
    sizes = new.value_counts()
    return {
        key: {
            "rows": int(sizes[key]),
            "phishing_rows": int((labels[new == key] == 1).sum()),
            "old_groups": int(old[new == key].nunique()),
        }
        for key in sizes.index[:LARGEST]
    }


def tld_sibling_share(labels: pd.Series, new: pd.Series, splits) -> list:
    """For each split, the share of test phishing whose new key is also the
    key of a training phishing row."""
    phish = (labels == 1).to_numpy()
    shares = []
    for train_idx, test_idx in splits:
        seen = set(new.iloc[train_idx][phish[train_idx]])
        test_keys = new.iloc[test_idx][phish[test_idx]]
        shares.append(float(test_keys.isin(seen).mean()))
    return shares


def audit(df: pd.DataFrame) -> dict:
    labels = df["label"]
    old = df["domain"].map(campaign_key)
    new = df["domain"].map(campaign_group)
    shares = tld_sibling_share(labels, new, random_splits(df, key=campaign_key))
    return {
        "groups": {"with_suffix": int(old.nunique()), "without_suffix": int(new.nunique())},
        "merged": merged(labels, old, new),
        "largest_group_rows": {"with_suffix": int(old.value_counts().max()), "without_suffix": int(new.value_counts().max())},
        "largest_without_suffix": largest(labels, old, new),
        "mixed_label": {"with_suffix": mixed_label(labels, old), "without_suffix": mixed_label(labels, new)},
        "tld_sibling_share_of_test_phishing": {
            "mean": round(float(np.mean(shares)), 4),
            "min": round(min(shares), 4),
            "max": round(max(shares), 4),
        },
    }


def main():
    results = {}
    for level in LEVELS:
        path = os.path.join(PROCESSED_DIR, f"dataset_{level}.csv")
        if not os.path.exists(path):
            print(f"{path} not found, skipping.")
            continue

        results[level] = audit(pd.read_csv(path))
        r = results[level]
        print(f"== {level} ==")
        print(f"  campaigns {r['groups']['with_suffix']:,} -> {r['groups']['without_suffix']:,}; "
              f"{r['merged']['old_groups']} merged into {r['merged']['groups']}")
        print(f"  mixed-label campaigns {r['mixed_label']['with_suffix']['groups']} -> {r['mixed_label']['without_suffix']['groups']}")
        print(f"  test phishing with a TLD sibling in training: {r['tld_sibling_share_of_test_phishing']['mean']:.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "grouping.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {RESULTS_DIR}/grouping.json")


if __name__ == "__main__":
    main()
