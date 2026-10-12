"""
Checks for the train/test grouping in src/preprocess.py. Run with:
    python tests/test_split.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from preprocess import campaign_group, campaign_key, split_by_campaign
from repeat_splits import N_SPLITS, random_splits


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


def test_campaign_group_merges_tld_variants():
    # Kits also register the same name under several cheap TLDs.
    assert campaign_group("pl-oferta48213.cfd") == campaign_group("pl-oferta77120.icu") == campaign_group("pl-oferta1.sbs")
    assert campaign_group("brt-euit.qpon") == campaign_group("brt-euit.bond") == "brt-euit"


def test_campaign_group_drops_multi_part_suffixes():
    assert campaign_group("foo1.co.uk") == campaign_group("foo2.com") == "foo#"
    assert campaign_group("loja.com.br") == "loja"


def test_campaign_group_keeps_ip_hosts_together():
    assert campaign_group("13.88.168.44") == campaign_group("43.134.233.26") == "#.#.#.#"
    # registered_domain returns IPv6 without brackets, which tldextract
    # doesn't recognize as an address.
    assert campaign_group("2001:db8::1") == campaign_key("2001:db8::1")


def test_campaign_group_keeps_different_names_apart():
    assert campaign_group("paypal-login.com") != campaign_group("paypal-verify.com")
    assert campaign_group("example.com") == "example"


def tld_variant_df() -> pd.DataFrame:
    # Forty TLDs, so a split that kept them as separate campaigns would
    # almost surely put some on each side.
    suffixes = "sbs cfd icu shop xyz top lat rest click bond de fr it es nl be at ch se no dk fi cz sk hu ro bg gr pt ie lt lv ee si hr rs ua ru by kz".split()
    domains = [f"pl-oferta{i}.{suffixes[i % 40]}" for i in range(80)] + [f"site{c}.com" for c in "abcdefghijklmnopqrstuvwxyz"]
    return pd.DataFrame({"domain": domains, "label": [1] * 80 + [0] * 26})


def test_split_keeps_tld_variants_together():
    df = tld_variant_df()
    train, test = split_by_campaign(df)
    assert not set(train["domain"].map(campaign_group)) & set(test["domain"].map(campaign_group))
    kit_in_test = test["domain"].str.startswith("pl-oferta").sum()
    assert kit_in_test in (0, 80), kit_in_test


def test_random_splits_keep_tld_variants_together():
    df = tld_variant_df()
    kit = df["domain"].str.startswith("pl-oferta").to_numpy()
    splits = list(random_splits(df))
    assert len(splits) == N_SPLITS
    for _, test_idx in splits:
        assert kit[test_idx].sum() in (0, 80)


def test_random_splits_use_the_given_key():
    # grouping_audit.py passes the old key to get the old splits, which do
    # split the kit.
    df = tld_variant_df()
    kit = df["domain"].str.startswith("pl-oferta").to_numpy()
    assert any(0 < kit[test_idx].sum() < 80 for _, test_idx in random_splits(df, key=campaign_key))


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
