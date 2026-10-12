"""
time_split.py

The split in preprocess.py and repeat_splits.py is random: test campaigns come
from the same months as training campaigns. This splits phishing by time
instead, to see how well models trained on older phishing catch phishing that
appeared later.

A phishing row's time is the earliest PhishTank submission of its registered
domain. A campaign starts at its earliest phishing row, and the whole campaign
goes to test if it started on or after the cut, to train if before. A kit seen
before and after the cut therefore stays in train, as in the random split. The
cut is the 80th percentile of campaign start times (1 - TEST_SIZE), chosen
before any model was trained on this split.

Benign rows have no timestamps (Tranco and Umbrella are one snapshot each), so
benign campaigns are split at random as in repeat_splits.py, over 10 seeds. The
phishing side is the same in every run, so the spread over runs only shows how
much the benign sample moves the numbers.

verified_online.csv only lists phishing that was still online when it was
downloaded, and most of its domains were first submitted in the last few
months before that. Older training rows are the ones that stayed up, so the
gap between train and test is partly long-lived versus fresh phishing, not
only time.

Writes results/time_split.json.

Usage:
    python src/time_split.py
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from build_dataset import PHISHTANK_FILE
from features import registered_domain
from preprocess import LEVELS, PROCESSED_DIR, TEST_SIZE, campaign_key, featurize
from repeat_splits import METRICS, N_SPLITS, RESULTS_DIR, score
from train import make_models, model_columns


def first_seen_by_domain(raw: pd.DataFrame) -> pd.Series:
    """Earliest submission time of each registered domain in PhishTank."""
    domains = raw["url"].fillna("").astype(str).map(registered_domain)
    times = pd.to_datetime(raw["submission_time"], utc=True)
    return times.groupby(domains).min()


def campaign_starts(df: pd.DataFrame) -> pd.Series:
    """Earliest phishing row time of each campaign that has phishing rows."""
    phish = df[df["label"] == 1]
    return phish["first_seen"].groupby(phish["domain"].map(campaign_key)).min()


def time_cut(starts: pd.Series) -> pd.Timestamp:
    return starts.quantile(1 - TEST_SIZE)


def time_splits(df: pd.DataFrame, n_splits: int, seed: int = 0):
    """Yields (train_idx, test_idx) positions. Campaigns with phishing rows go
    by their start time, benign-only campaigns are split at random."""
    groups = df["domain"].map(campaign_key)
    starts = campaign_starts(df)
    late = groups.isin(starts.index[starts >= time_cut(starts)]).to_numpy()
    untimed = np.flatnonzero(~groups.isin(starts.index).to_numpy())

    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=TEST_SIZE, random_state=seed)
    for _, benign_test in splitter.split(untimed, groups=groups.iloc[untimed]):
        is_test = late.copy()
        is_test[untimed[benign_test]] = True
        yield np.flatnonzero(~is_test), np.flatnonzero(is_test)


def describe(df: pd.DataFrame, allegro: np.ndarray, splits: list) -> dict:
    """Facts about the split that the metrics alone don't show."""
    starts = campaign_starts(df)
    cut = time_cut(starts)
    phish = df["label"] == 1
    test = np.zeros(len(df), dtype=bool)
    test[splits[0][1]] = True
    train_phish, test_phish = phish & ~test, phish & test

    # Campaigns that started before the cut but have rows first seen after it.
    groups = df["domain"].map(campaign_key)
    last = df[phish]["first_seen"].groupby(groups[phish]).max()
    straddling = starts.index[(starts < cut) & (last >= cut)]
    moved = train_phish & groups.isin(straddling) & (df["first_seen"] >= cut)

    benign_test = [int((df.iloc[test_idx]["label"] == 0).sum()) for _, test_idx in splits]
    years = df[train_phish]["first_seen"].dt.year.value_counts().sort_index()
    return {
        "cut": cut.isoformat(),
        "test_phishing_first_seen": {
            "min": df[test_phish]["first_seen"].min().isoformat(),
            "max": df[test_phish]["first_seen"].max().isoformat(),
        },
        "phishing_train": int(train_phish.sum()),
        "phishing_test": int(test_phish.sum()),
        "benign_test_mean": round(float(np.mean(benign_test)), 1),
        "straddling_campaigns": len(straddling),
        "straddling_rows_after_cut_in_train": int(moved.sum()),
        "allegro_share_train_phishing": round(float(allegro[train_phish.to_numpy()].mean()), 4),
        "allegro_share_test_phishing": round(float(allegro[test_phish.to_numpy()].mean()), 4),
        "train_phishing_by_first_seen_year": {str(year): int(n) for year, n in years.items()},
    }


def main():
    raw = pd.read_csv(PHISHTANK_FILE, usecols=["url", "submission_time", "target"])
    seen = first_seen_by_domain(raw)
    targets = raw.drop_duplicates("url").set_index("url")["target"]
    results = {}

    for level in LEVELS:
        path = os.path.join(PROCESSED_DIR, f"dataset_{level}.csv")
        if not os.path.exists(path):
            print(f"{path} not found, skipping.")
            continue

        data = pd.read_csv(path)
        df = featurize(data, level)
        # Only phishing rows get a time: a benign domain can also appear in
        # PhishTank (an abused legitimate site), which says nothing about it.
        df["first_seen"] = df["domain"].map(seen).where(df["label"] == 1)
        allegro = ((data["url"].map(targets) == "Allegro") & (data["label"] == 1)).to_numpy()
        y = df["label"]
        splits = list(time_splits(df, N_SPLITS))

        rows = []
        for train_idx, test_idx in splits:
            for name, model in make_models(level).items():
                X = df[model_columns(name, level)]
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                X_test = X.iloc[test_idx]
                metrics = score(y.iloc[test_idx], model.predict(X_test), model.predict_proba(X_test)[:, 1])
                rows.append({"model": name, **metrics})

        summary = pd.DataFrame(rows).groupby("model").agg(["mean", "std"]).round(4)
        results[level] = {
            "split": describe(df, allegro, splits),
            "models": {
                model: {metric: {"mean": summary.loc[model, (metric, "mean")], "std": summary.loc[model, (metric, "std")]}
                        for metric in METRICS}
                for model in make_models(level)
            },
        }

        split = results[level]["split"]
        print(f"== {level}: cut {split['cut']}, {split['phishing_train']:,} train / "
              f"{split['phishing_test']:,} test phishing; mean (std) over {N_SPLITS} benign samples ==")
        for model, metrics in results[level]["models"].items():
            cells = "  ".join(f"{m} {v['mean']:.3f} ({v['std']:.3f})" for m, v in metrics.items())
            print(f"  {model:30} {cells}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "time_split.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved {RESULTS_DIR}/time_split.json")


if __name__ == "__main__":
    main()
