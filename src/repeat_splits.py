"""
repeat_splits.py

The metrics in evaluate.py come from one train/test split. This repeats the
same campaign-grouped split with 10 different seeds and reports the mean and
standard deviation, to show how much one split's numbers can move by chance.

Writes results/repeated_splits.json.

Usage:
    python src/repeat_splits.py
"""

import json
import os

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

from features import feature_names
from preprocess import LEVELS, PROCESSED_DIR, TEST_SIZE, campaign_key, featurize
from train import make_models

RESULTS_DIR = "results"
N_SPLITS = 10


def score(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "false_positive_rate": fp / (fp + tn),
    }


def main():
    results = {}

    for level in LEVELS:
        path = os.path.join(PROCESSED_DIR, f"dataset_{level}.csv")
        if not os.path.exists(path):
            print(f"{path} not found, skipping.")
            continue

        df = featurize(pd.read_csv(path), level)
        X, y = df[feature_names(level)], df["label"]
        groups = df["domain"].map(campaign_key)
        splitter = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=0)

        rows = []
        for train_idx, test_idx in splitter.split(X, y, groups=groups):
            for name, model in make_models().items():
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                metrics = score(y.iloc[test_idx], model.predict(X.iloc[test_idx]))
                rows.append({"model": name, **metrics})

        summary = pd.DataFrame(rows).groupby("model").agg(["mean", "std"]).round(4)
        results[level] = {
            model: {metric: {"mean": summary.loc[model, (metric, "mean")], "std": summary.loc[model, (metric, "std")]}
                    for metric in ("precision", "recall", "f1", "false_positive_rate")}
            for model in summary.index
        }

        print(f"== {level}: mean (std) over {N_SPLITS} splits ==")
        for model, metrics in results[level].items():
            cells = "  ".join(f"{m} {v['mean']:.3f} ({v['std']:.3f})" for m, v in metrics.items())
            print(f"  {model:20} {cells}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "repeated_splits.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved {RESULTS_DIR}/repeated_splits.json")


if __name__ == "__main__":
    main()
