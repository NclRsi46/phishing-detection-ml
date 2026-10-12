"""
evaluate.py

Scores every trained model on its held-out test split only (never on rows
it was trained on). Reports precision, recall, F1, and false positive rate at
the default 0.5 threshold, plus the best recall at a 1% and 5% false positive
rate and the average precision, which don't depend on one threshold.

Writes results/metrics.json, results/confusion_{level}_{model}.png, and
results/pr_curve_{level}.png.
"""

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
RESULTS_DIR = "results"
LEVELS = ["domain", "host"]
MODEL_NAMES = ["logistic_regression", "random_forest", "char_ngram_logistic_regression"]
TOP_NGRAMS = 15

# Target false positive rates for the operating-point metrics.
OPERATING_POINTS = {"recall_at_1pct_fpr": 0.01, "recall_at_5pct_fpr": 0.05}


def recall_at_fpr(y_true, scores, target: float) -> float:
    """Best recall at any threshold whose false positive rate is at most target.

    The threshold is read off this data's ROC curve, so this describes how well
    the model ranks rows, not a threshold that would hold up on new data.
    """
    # With one class the rate is undefined: all-benign gives nan, which a
    # mean over splits would silently skip.
    if np.unique(y_true).size != 2:
        raise ValueError("recall_at_fpr needs both benign and phishing rows")
    # Keep every threshold: the default drops points on straight stretches of
    # the curve, and one of those can be the best point under the target.
    fpr, tpr, _ = roc_curve(y_true, scores, drop_intermediate=False)
    return float(tpr[fpr <= target].max())


def evaluate(bundle: dict, test: pd.DataFrame) -> dict:
    model = bundle["model"]
    y_true = test["label"]
    y_pred = model.predict(test[bundle["features"]])
    scores = model.predict_proba(test[bundle["features"]])[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "false_positive_rate": round(fp / (fp + tn), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        **{key: round(recall_at_fpr(y_true, scores, target), 4) for key, target in OPERATING_POINTS.items()},
        "average_precision": round(average_precision_score(y_true, scores), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    if hasattr(model, "feature_importances_"):
        ranked = sorted(zip(bundle["features"], model.feature_importances_), key=lambda t: -t[1])
        metrics["feature_importance"] = {name: round(float(v), 3) for name, v in ranked}

    if "columntransformer" in getattr(model, "named_steps", {}):
        metrics["top_ngrams"] = top_ngrams(model)

    return metrics


def top_ngrams(model) -> dict:
    """The n-grams with the largest weights toward each class, to check what
    the n-gram model learned (a source artifact would show up here)."""
    names = model[0].named_transformers_["ngrams"].get_feature_names_out()
    weights = model[-1].coef_[0]
    order = np.argsort(weights, kind="stable")
    return {
        "phishing": {names[i]: round(float(weights[i]), 3) for i in order[::-1][:TOP_NGRAMS]},
        "benign": {names[i]: round(float(weights[i]), 3) for i in order[:TOP_NGRAMS]},
    }


def save_confusion_matrix(metrics: dict, level: str, name: str) -> None:
    cm = metrics["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    disp = ConfusionMatrixDisplay(matrix, display_labels=["benign", "phishing"])
    disp.plot(cmap="Blues", values_format="d", colorbar=False)
    disp.ax_.set_title(f"{level} level, {name.replace('_', ' ')}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_{level}_{name}.png"), dpi=120)
    plt.close()


def save_pr_curve(bundles: dict, test: pd.DataFrame, level: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for i, (name, bundle) in enumerate(bundles.items()):
        PrecisionRecallDisplay.from_estimator(
            bundle["model"], test[bundle["features"]], test["label"],
            name=name.replace("_", " "), ax=ax, plot_chance_level=(i == len(bundles) - 1),
        )
    ax.set_xlabel("Recall (phishing)")
    ax.set_ylabel("Precision (phishing)")
    ax.set_title(f"{level} level, precision-recall (test split)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"pr_curve_{level}.png"), dpi=120)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}

    for level in LEVELS:
        test_path = os.path.join(PROCESSED_DIR, f"{level}_test.csv")
        train_path = os.path.join(PROCESSED_DIR, f"{level}_train.csv")
        if not os.path.exists(test_path):
            print(f"{test_path} not found, skipping.")
            continue

        test = pd.read_csv(test_path)
        train_rows = len(pd.read_csv(train_path, usecols=["label"]))
        results[level] = {"train_rows": train_rows, "test_rows": len(test), "models": {}}
        bundles = {name: joblib.load(os.path.join(MODEL_DIR, f"{level}_{name}.pkl")) for name in MODEL_NAMES}

        for name, bundle in bundles.items():
            metrics = evaluate(bundle, test)
            results[level]["models"][name] = metrics
            save_confusion_matrix(metrics, level, name)

            print(
                f"{level:6} {name:30} precision {metrics['precision']:.4f}  recall {metrics['recall']:.4f}  "
                f"F1 {metrics['f1']:.4f}  FPR {metrics['false_positive_rate']:.4f}  "
                f"R@1%FPR {metrics['recall_at_1pct_fpr']:.4f}  R@5%FPR {metrics['recall_at_5pct_fpr']:.4f}  "
                f"AP {metrics['average_precision']:.4f}"
            )

        save_pr_curve(bundles, test, level)

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {RESULTS_DIR}/metrics.json")


if __name__ == "__main__":
    main()
