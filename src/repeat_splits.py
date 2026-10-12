"""
repeat_splits.py

The metrics in evaluate.py come from one train/test split. This repeats the
same campaign-grouped split with 10 different seeds and reports the mean and
standard deviation, to show how much one split's numbers can move by chance.
ngram_audit.py and grouping_audit.py use the same 10 splits.

Recall at 1% FPR moves the most between splits, so each split's value is kept
too, with the number of benign and phishing rows tied at the model's highest
score. When more benign rows tie there than 1% FPR allows, no threshold can
take any of them and that split's recall at 1% FPR is 0.

Writes results/repeated_splits.json.

Usage:
    python src/repeat_splits.py
"""

import json
import os

import numpy as np
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


def random_splits(df: pd.DataFrame, key=campaign_group):
    """The 10 campaign-grouped (train_idx, test_idx) splits."""
    groups = df["domain"].map(key)
    splitter = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=0)
    return splitter.split(df, df["label"], groups=groups)


def top_tie(y_true, scores) -> dict:
    """Benign and phishing rows tied at the highest score."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    top = scores == scores.max()
    return {"benign": int((top & (y_true == 0)).sum()), "phishing": int((top & (y_true == 1)).sum())}


def main():
    results = {}

    for level in LEVELS:
        path = os.path.join(PROCESSED_DIR, f"dataset_{level}.csv")
        if not os.path.exists(path):
            print(f"{path} not found, skipping.")
            continue

        df = featurize(pd.read_csv(path), level)
        y = df["label"]

        rows = []
        for train_idx, test_idx in random_splits(df):
            for name, model in make_models(level).items():
                X = df[model_columns(name, level)]
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                X_test = X.iloc[test_idx]
                scores = model.predict_proba(X_test)[:, 1]
                metrics = score(y.iloc[test_idx], model.predict(X_test), scores)
                tie = top_tie(y.iloc[test_idx], scores)
                rows.append({"model": name, **metrics, "top_tie_benign": tie["benign"], "top_tie_phishing": tie["phishing"]})

        rows = pd.DataFrame(rows)
        summary = rows.groupby("model").agg(["mean", "std"]).round(4)
        results[level] = {
            model: {
                **{metric: {"mean": summary.loc[model, (metric, "mean")], "std": summary.loc[model, (metric, "std")]}
                   for metric in METRICS},
                "per_split": {
                    column: [round(float(v), 4) for v in rows.loc[rows["model"] == model, column]]
                    for column in ("recall_at_1pct_fpr", "top_tie_benign", "top_tie_phishing")
                },
            }
            for model in make_models(level)
        }

        print(f"== {level}: mean (std) over {N_SPLITS} splits ==")
        for model, metrics in results[level].items():
            cells = "  ".join(f"{m} {metrics[m]['mean']:.3f} ({metrics[m]['std']:.3f})" for m in METRICS)
            print(f"  {model:30} {cells}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "repeated_splits.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved {RESULTS_DIR}/repeated_splits.json")


if __name__ == "__main__":
    main()
