"""
Checks for the time-based split in src/time_split.py. Run with:
    python tests/test_time_split.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import math

import pandas as pd

from preprocess import campaign_key
from time_split import TEST_SIZE, campaign_starts, first_seen_by_domain, time_cut, time_splits


def day(n: int) -> pd.Timestamp:
    return pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=n)


def make_df() -> pd.DataFrame:
    # Ten one-domain phishing campaigns that start on days 0..9, one
    # numbered kit that starts on day 1 and is seen again on day 9, a late
    # and an early campaign with a phishing and a benign row each, 26 benign
    # domains, and a benign campaign of five numbered domains.
    rows = [(f"phish{c}.com", 1, day(i)) for i, c in enumerate("abcdefghij")]
    rows += [("kit11.sbs", 1, day(1)), ("kit22.sbs", 1, day(9))]
    rows += [("shared.net", 1, day(9)), ("shared.net", 0, pd.NaT)]
    rows += [("early.net", 1, day(0)), ("early.net", 0, pd.NaT)]
    rows += [(f"benign{c}.org", 0, pd.NaT) for c in "abcdefghijklmnopqrstuvwxyz"]
    rows += [(f"shop{i}.org", 0, pd.NaT) for i in range(5)]
    return pd.DataFrame(rows, columns=["domain", "label", "first_seen"])


def test_first_seen_is_the_earliest_submission():
    # PhishTank's CSV is newest first, so the first row of a domain is its
    # latest submission, not its first.
    raw = pd.DataFrame({
        "url": ["http://a.example.com/x", "http://b.example.com/y", "http://other.net/"],
        "submission_time": ["2025-11-30T10:00:00+00:00", "2024-03-01T08:00:00+00:00", "2025-06-01T00:00:00+00:00"],
    })
    seen = first_seen_by_domain(raw)
    assert seen["example.com"] == pd.Timestamp("2024-03-01T08:00:00", tz="UTC")
    assert seen["other.net"] == pd.Timestamp("2025-06-01", tz="UTC")


def test_campaign_start_is_the_earliest_phishing_row():
    starts = campaign_starts(make_df())
    assert starts[campaign_key("kit11.sbs")] == day(1)
    assert starts[campaign_key("shared.net")] == day(9)
    # Benign-only campaigns have no start.
    assert campaign_key("benigna.org") not in starts.index


def test_cut_is_the_80th_percentile_of_campaign_starts():
    starts = campaign_starts(make_df())
    assert time_cut(starts) == starts.quantile(0.8)


def test_campaign_starting_on_the_cut_goes_to_test():
    # Eleven phishing campaigns on days 0..10: the 80th percentile is
    # exactly day 8.
    rows = [(f"phish{c}.com", 1, day(i)) for i, c in enumerate("abcdefghijk")]
    rows += [(f"benign{c}.org", 0, pd.NaT) for c in "abcdefghij"]
    df = pd.DataFrame(rows, columns=["domain", "label", "first_seen"])
    assert time_cut(campaign_starts(df)) == day(8)
    _, test_idx = next(time_splits(df, n_splits=1))
    test = df.iloc[test_idx]
    assert set(test[test["label"] == 1]["first_seen"]) == {day(8), day(9), day(10)}


def test_splits_never_share_a_campaign():
    df = make_df()
    groups = df["domain"].map(campaign_key)
    for train_idx, test_idx in time_splits(df, n_splits=5):
        assert not set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
        assert len(train_idx) + len(test_idx) == len(df)


def test_test_phishing_starts_on_or_after_the_cut():
    df = make_df()
    starts = campaign_starts(df)
    cut = time_cut(starts)
    start = df["domain"].map(campaign_key).map(starts)
    for train_idx, test_idx in time_splits(df, n_splits=5):
        test_phish = df.iloc[test_idx]["label"] == 1
        train_phish = df.iloc[train_idx]["label"] == 1
        assert (start.iloc[test_idx][test_phish] >= cut).all()
        assert (start.iloc[train_idx][train_phish] < cut).all()
        assert test_phish.any()


def test_straddling_campaign_goes_to_train():
    # kit22.sbs is first seen after the cut, but its kit started before it.
    df = make_df()
    kit = df.index[df["domain"].str.startswith("kit")]
    for train_idx, _ in time_splits(df, n_splits=5):
        assert set(kit) <= set(train_idx)


def test_benign_row_follows_its_campaigns_phishing_time():
    df = make_df()
    shared = df.index[(df["domain"] == "shared.net") & (df["label"] == 0)][0]
    for _, test_idx in time_splits(df, n_splits=5):
        assert shared in set(test_idx)


def test_benign_row_of_an_early_campaign_stays_in_train():
    df = make_df()
    early = df.index[(df["domain"] == "early.net") & (df["label"] == 0)][0]
    for train_idx, _ in time_splits(df, n_splits=5):
        assert early in set(train_idx)


def test_benign_only_campaigns_split_by_test_size():
    df = make_df()
    groups = df["domain"].map(campaign_key)
    benign_only = groups[~groups.isin(campaign_starts(df).index)]
    n = benign_only.nunique()
    for _, test_idx in time_splits(df, n_splits=5):
        in_test = benign_only[benign_only.index.isin(test_idx)].nunique()
        assert math.floor(TEST_SIZE * n) <= in_test <= math.ceil(TEST_SIZE * n), in_test


def test_only_the_benign_side_changes_between_splits():
    df = make_df()
    phish = set(df.index[df["label"] == 1])
    test_sets = [set(test_idx) for _, test_idx in time_splits(df, n_splits=5)]
    assert len({frozenset(t & phish) for t in test_sets}) == 1
    assert len({frozenset(t - phish) for t in test_sets}) > 1


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name} {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
