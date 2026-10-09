"""
Checks for the train/test grouping in src/preprocess.py. Run with:
    python tests/test_split.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from preprocess import campaign_key, split_by_campaign


def test_campaign_key_merges_numbered_kit_domains():
    # Phishing kits register many domains that differ only in a number.
    assert campaign_key("pl-oferta48213.sbs") == campaign_key("pl-oferta77120.sbs")
    assert campaign_key("pl-oferta48213.sbs") == "pl-oferta#.sbs"


def test_campaign_key_keeps_different_names_apart():
    assert campaign_key("paypal-login.com") != campaign_key("paypal-verify.com")
    assert campaign_key("example.com") == "example.com"


def test_split_never_shares_a_campaign():
    domains = [f"pl-oferta{i}.sbs" for i in range(50)] + [f"site{c}.com" for c in "abcdefghijklmnopqrstuvwxyz"]
    df = pd.DataFrame({"domain": domains, "label": [1] * 50 + [0] * 26})
    train, test = split_by_campaign(df)
    train_keys = set(train["domain"].map(campaign_key))
    test_keys = set(test["domain"].map(campaign_key))
    assert not train_keys & test_keys
    # All 50 numbered kit domains must land on the same side.
    kit_in_test = test["domain"].str.startswith("pl-oferta").sum()
    assert kit_in_test in (0, 50)


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
