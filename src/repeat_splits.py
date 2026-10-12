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
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

from evaluate import OPERATING_POINTS, recall_at_fpr
from preprocess import LEVELS, PROCESSED_DIR, TEST_SIZE, campaign_group, featurize
from train import make_models, model_columns

RESULTS_DIR = "results"
N_SPLITS = 10
METRICS = ["precision", "recall", "f1", "false_positive_rate", *OPERATING_POINTS, "average_precision"]


def score(y_true, y_pred, scores) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "false_positive_rate": fp / (fp + tn),
        **{key: recall_at_fpr(y_true, scores, target) for key, target in OPERATING_POINTS.items()},
        "average_precision": average_precision_score(y_true, scores),
    }


def main():
    results = {}

    for level in LEVELS:
        path = os.path.join(PROCESSED_DIR, f"dataset_{level}.csv")
        if not os.path.exists(path):
            print(f"{path} not found, skipping.")
            continue

        df = featurize(pd.read_csv(path), level)
        y = df["label"]
        groups = df["domain"].map(campaign_group)
        splitter = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=0)

        rows = []
        for train_idx, test_idx in splitter.split(df, y, groups=groups):
            for name, model in make_models(level).items():
                X = df[model_columns(name, level)]
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                X_test = X.iloc[test_idx]
                metrics = score(y.iloc[test_idx], model.predict(X_test), model.predict_proba(X_test)[:, 1])
                rows.append({"model": name, **metrics})

        summary = pd.DataFrame(rows).groupby("model").agg(["mean", "std"]).round(4)
        results[level] = {
            model: {metric: {"mean": summary.loc[model, (metric, "mean")], "std": summary.loc[model, (metric, "std")]}
                    for metric in METRICS}
            for model in make_models(level)
        }

        print(f"== {level}: mean (std) over {N_SPLITS} splits ==")
        for model, metrics in results[level].items():
            cells = "  ".join(f"{m} {v['mean']:.3f} ({v['std']:.3f})" for m, v in metrics.items())
            print(f"  {model:30} {cells}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "repeated_splits.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved {RESULTS_DIR}/repeated_splits.json")


if __name__ == "__main__":
    main()
