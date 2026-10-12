"""
ngram_audit.py

Checks where the char n-gram model's gain over the hand-built features comes
from, over the same 10 splits as repeat_splits.py:

    TLD or name   the hand-built features plus the full public suffix, the
                  n-gram model on the name with the suffix stripped, and a
                  logistic regression on the suffix alone.
    Allegro       recall at 1% FPR without Allegro-themed phishing, and how
                  much of each model's 1%-FPR slice is Allegro. Allegro is
                  8.5% of the phishing rows, mostly from a few kits that vary
                  letters, which the campaign grouping can't merge.
    kit wording   recall at 1% FPR on test phishing names that share no
                  8-character substring (digit runs as "#", suffix stripped,
                  since the TLD is checked above) with a training phishing
                  name. Kits reuse words ("pl-oferta", "coursier") across
                  names the campaign grouping keeps apart. The filter
                  also drops names that share ordinary words, so this likely
                  understates recall on new names; it isn't a clean measure.
                  Names under 8 characters (digits collapsed) have no
                  substring to share, so they are never flagged.

Allegro-themed means PhishTank's "target" column says Allegro.

Writes results/ngram_audit.json.

Usage:
    python src/ngram_audit.py
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from build_dataset import PHISHTANK_FILE
from evaluate import recall_at_fpr
from features import EXTRACT, feature_names
from preprocess import LEVELS, PROCESSED_DIR, TEST_SIZE, campaign_key, featurize
from repeat_splits import N_SPLITS, RESULTS_DIR, score
from train import make_models, model_columns

SUBSTRING_LENGTH = 8


def name_and_suffix(text: str, level: str):
    """The scored name split into the part a registrant chooses and the
    public suffix, the same way features.py measures the name."""
    ext = EXTRACT(text)
    ip = ext.ipv4 or ext.ipv6
    if ip:
        return ip, ""
    if level == "host" and ext.subdomain:
        return f"{ext.subdomain}.{ext.domain}", ext.suffix
    return ext.domain, ext.suffix


def audit_models(level: str) -> dict:
    """The three models from train.py plus the diagnostic ones, each with its
    input columns. The name-only model has the same settings as the n-gram
    model in train.py."""
    hand = feature_names(level)
    models = {name: (model, model_columns(name, level)) for name, model in make_models(level).items()}
    models["hand_plus_suffix"] = (
        make_pipeline(
            ColumnTransformer([
                ("hand", StandardScaler(), hand),
                ("suffix", OneHotEncoder(handle_unknown="ignore"), ["suffix"]),
            ]),
            LogisticRegression(max_iter=1000),
        ),
        hand + ["suffix"],
    )
    models["ngram_name_only"] = (
        make_pipeline(
            ColumnTransformer([("ngrams", TfidfVectorizer(analyzer="char", ngram_range=(1, 5), min_df=2), "name")]),
            LogisticRegression(max_iter=1000),
        ),
        ["name"],
    )
    models["suffix_only"] = (
        make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=1000)),
        ["suffix"],
    )
    return models


def caught_at_fpr(y_true, scores, target: float):
    """Phishing rows caught at the threshold recall_at_fpr picks: everything
    scored above the highest benign score that would break the target."""
    benign = np.sort(scores[y_true == 0])[::-1]
    allowed = int(np.floor(target * len(benign)))
    threshold = benign[allowed] if allowed < len(benign) else -np.inf
    caught = (scores > threshold) & (y_true == 1)
    # Same point on the ROC curve as the reported metric, or this audit
    # would describe a different slice than the README's numbers.
    assert np.isclose(caught.sum() / (y_true == 1).sum(), recall_at_fpr(y_true, scores, target))
    return caught


def allegro_metrics(y_true, scores, allegro) -> dict:
    # With no Allegro rows these are nan, which the mean over splits would
    # silently skip.
    if not allegro.any():
        raise ValueError("no Allegro-themed rows in this test split")
    caught = caught_at_fpr(y_true, scores, 0.01)
    other = ~allegro
    return {
        "recall_at_1pct_fpr_without_allegro": recall_at_fpr(y_true[other], scores[other], 0.01),
        "allegro_recall_at_1pct_fpr": float(caught[allegro].mean()),
        # Undefined when the model catches nothing at 1% FPR. The mean is
        # then over the splits where it catches something.
        "allegro_share_of_1pct_fpr_catches": float(caught[allegro].sum() / caught.sum()) if caught.any() else float("nan"),
        "allegro_share_of_test_phishing": float(allegro.sum() / (y_true == 1).sum()),
    }


def substrings(name: str) -> set:
    text = campaign_key(name)
    return {text[i:i + SUBSTRING_LENGTH] for i in range(len(text) - SUBSTRING_LENGTH + 1)}


def repeated_names(train_names, test_names):
    """True for each test name that shares a substring (digit runs as "#")
    with any of the training names."""
    seen = set().union(*(substrings(name) for name in train_names))
    return np.array([not substrings(name).isdisjoint(seen) for name in test_names])


def unrepeated_metrics(y_true, scores, repeated) -> dict:
    keep = ~repeated
    return {
        "recall_at_1pct_fpr_unrepeated": recall_at_fpr(y_true[keep], scores[keep], 0.01),
        "repeated_share_of_test_phishing": float(repeated.sum() / (y_true == 1).sum()),
    }


def main():
    targets = pd.read_csv(PHISHTANK_FILE, usecols=["url", "target"]).drop_duplicates("url").set_index("url")["target"]
    results = {}

    for level in LEVELS:
        path = os.path.join(PROCESSED_DIR, f"dataset_{level}.csv")
        if not os.path.exists(path):
            print(f"{path} not found, skipping.")
            continue

        raw = pd.read_csv(path)
        df = featurize(raw, level)
        df["name"], df["suffix"] = zip(*df[level].map(lambda text: name_and_suffix(text, level)))
        y = df["label"]
        allegro = (raw["url"].map(targets) == "Allegro").values
        groups = df["domain"].map(campaign_key)
        splitter = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=0)

        rows = []
        for train_idx, test_idx in splitter.split(df, y, groups=groups):
            y_test = y.iloc[test_idx].values
            train_phishing = df["name"].iloc[train_idx][y.iloc[train_idx] == 1]
            # Only phishing rows are dropped, so the 1% FPR threshold stays
            # on the same benign rows.
            repeated = repeated_names(train_phishing, df["name"].iloc[test_idx]) & (y_test == 1)
            for name, (model, columns) in audit_models(level).items():
                X = df[columns]
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                y_pred = model.predict(X.iloc[test_idx])
                scores = model.predict_proba(X.iloc[test_idx])[:, 1]
                rows.append({
                    "model": name,
                    **score(y_test, y_pred, scores),
                    **allegro_metrics(y_test, scores, allegro[test_idx]),
                    **unrepeated_metrics(y_test, scores, repeated),
                })

        summary = pd.DataFrame(rows).groupby("model", sort=False).agg(["mean", "std"]).round(4)
        metrics = summary.columns.get_level_values(0).unique()
        results[level] = {
            "allegro_phishing_rows": int(allegro.sum()),
            "models": {
                model: {m: {"mean": summary.loc[model, (m, "mean")], "std": summary.loc[model, (m, "std")]} for m in metrics}
                for model in summary.index
            },
        }

        print(f"== {level}: mean over {N_SPLITS} splits ({allegro.sum()} Allegro-themed phishing rows) ==")
        for model, m in results[level]["models"].items():
            print(
                f"  {model:30} AP {m['average_precision']['mean']:.3f}  R@1%FPR {m['recall_at_1pct_fpr']['mean']:.3f}  "
                f"without Allegro {m['recall_at_1pct_fpr_without_allegro']['mean']:.3f}  "
                f"unrepeated {m['recall_at_1pct_fpr_unrepeated']['mean']:.3f}  "
                f"Allegro share of 1% slice {m['allegro_share_of_1pct_fpr_catches']['mean']:.3f}"
            )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "ngram_audit.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved {RESULTS_DIR}/ngram_audit.json")


if __name__ == "__main__":
    main()
