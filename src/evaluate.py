"""
evaluate.py

Scores every trained model on its held-out test split only (never on rows
it was trained on). Reports precision, recall, F1, and false positive rate.

Writes results/metrics.json and results/confusion_{level}_{model}.png.
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
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
RESULTS_DIR = "results"
LEVELS = ["domain", "host"]
MODEL_NAMES = ["logistic_regression", "random_forest"]


def evaluate(bundle: dict, test: pd.DataFrame) -> dict:
    model = bundle["model"]
    y_true = test["label"]
    y_pred = model.predict(test[bundle["features"]])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "false_positive_rate": round(fp / (fp + tn), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    if hasattr(model, "feature_importances_"):
        ranked = sorted(zip(bundle["features"], model.feature_importances_), key=lambda t: -t[1])
        metrics["feature_importance"] = {name: round(float(v), 3) for name, v in ranked}

    return metrics


def save_confusion_matrix(metrics: dict, level: str, name: str) -> None:
    cm = metrics["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    disp = ConfusionMatrixDisplay(matrix, display_labels=["benign", "phishing"])
    disp.plot(cmap="Blues", values_format="d", colorbar=False)
    disp.ax_.set_title(f"{level} level, {name.replace('_', ' ')}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_{level}_{name}.png"), dpi=120)
    plt.close()


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

        for name in MODEL_NAMES:
            bundle = joblib.load(os.path.join(MODEL_DIR, f"{level}_{name}.pkl"))
            metrics = evaluate(bundle, test)
            results[level]["models"][name] = metrics
            save_confusion_matrix(metrics, level, name)

            print(
                f"{level:6} {name:20} precision {metrics['precision']:.4f}  recall {metrics['recall']:.4f}  "
                f"F1 {metrics['f1']:.4f}  FPR {metrics['false_positive_rate']:.4f}"
            )

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {RESULTS_DIR}/metrics.json")


if __name__ == "__main__":
    main()
