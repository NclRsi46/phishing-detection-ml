"""
Checks for src/grouping_audit.py. Run with:
    python tests/test_grouping.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from grouping_audit import merged, mixed_label, tld_sibling_share
from preprocess import campaign_group, campaign_key


def make_df() -> pd.DataFrame:
    # One kit under three TLDs, a name that is phishing under one TLD and
    # benign under another, and two unrelated benign domains.
    rows = [("kit1.sbs", 1), ("kit2.cfd", 1), ("kit3.icu", 1), ("kit4.icu", 1),
            ("brand.info", 1), ("brand.fr", 0), ("plain.com", 0), ("other.org", 0)]
    return pd.DataFrame(rows, columns=["domain", "label"])


def keys(df):
    return df["domain"].map(campaign_key), df["domain"].map(campaign_group)


def test_merged_counts_old_groups_and_rows():
    df = make_df()
    old, new = keys(df)
    # kit#.sbs, kit#.cfd, kit#.icu -> kit#; brand.info, brand.fr -> brand.
    assert merged(df["label"], old, new) == {"groups": 2, "old_groups": 5, "phishing_rows": 5, "benign_rows": 1}


def test_mixed_label_only_after_the_merge():
    df = make_df()
    old, new = keys(df)
    assert mixed_label(df["label"], old) == {"groups": 0, "rows": 0, "keys": []}
    assert mixed_label(df["label"], new) == {"groups": 1, "rows": 2, "keys": ["brand"]}


def test_tld_sibling_share_counts_test_phishing_only():
    df = make_df()
    _, new = keys(df)
    # Split 1: train has kit1.sbs; test has kit2.cfd (a sibling),
    # brand.info (not), and benign brand.fr, which must not count.
    # Split 2: brand.info's only match in train is benign brand.fr, which
    # doesn't make it a sibling.
    splits = [([0, 6, 7], [1, 4, 5]), ([0, 5, 6, 7], [1, 4])]
    assert tld_sibling_share(df["label"], new, splits) == [0.5, 0.5]


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
