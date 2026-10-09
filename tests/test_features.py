"""
Checks for src/features.py. Run with:
    python tests/test_features.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import extract_features, feature_names, hostname, registered_domain


def test_path_and_query_are_ignored():
    # The original model learned "has a path = phishing". Features must not
    # change when only the path, query, or scheme changes.
    for level in ("domain", "host"):
        a = extract_features("https://www.example.com", level)
        b = extract_features("http://www.example.com/a/b/c.php?x=1&y=2#top", level)
        assert a == b, level


def test_domain_level_ignores_subdomains():
    # Tranco has no subdomains, so at the domain level they must not matter.
    a = extract_features("https://secure.login.example.com/x", "domain")
    b = extract_features("example.com", "domain")
    assert a == b


def test_registered_domain():
    assert registered_domain("https://a.b.paypal.co.uk/x?y=1") == "paypal.co.uk"
    assert registered_domain("http://192.168.1.1/login") == "192.168.1.1"
    assert registered_domain("localhost") == ""
    assert registered_domain("https://user@paypal-login-secure.com:8080/p") == "paypal-login-secure.com"


def test_hostname():
    assert hostname("https://WWW.Example.com:443/path") == "www.example.com"
    assert hostname("http://10.0.0.1/x") == "10.0.0.1"


def test_brand_and_lure_words():
    f = extract_features("https://paypal-login-secure.com", "domain")
    assert f["has_brand_word"] == 1
    assert f["has_lure_word"] == 1

    # A brand's own domain is not a brand lure, in any country TLD.
    assert extract_features("https://www.paypal.com/signin", "domain")["has_brand_word"] == 0
    assert extract_features("paypal.de", "domain")["has_brand_word"] == 0


def test_brand_in_subdomain_only_counts_at_host_level():
    url = "http://paypal.com.account-check.xyz/login"
    assert extract_features(url, "host")["has_brand_word"] == 1
    assert extract_features(url, "domain")["has_brand_word"] == 0


def test_ip_and_punycode():
    f = extract_features("http://192.168.1.1/login", "domain")
    assert f["is_ip"] == 1
    assert f["tld_com"] + f["tld_legacy"] + f["tld_country"] + f["tld_other"] == 0
    assert extract_features("https://xn--pypal-4ve.com", "domain")["is_punycode"] == 1


def test_tld_groups():
    def group(url):
        f = extract_features(url, "domain")
        return [k for k in ("tld_com", "tld_legacy", "tld_country", "tld_other") if f[k] == 1]

    assert group("example.com") == ["tld_com"]
    assert group("example.org") == ["tld_legacy"]
    assert group("example.co.uk") == ["tld_country"]
    assert group("example.xyz") == ["tld_other"]


def test_counts():
    f = extract_features("https://my-shop-24.com", "domain")
    assert f["length"] == len("my-shop-24")
    assert f["num_hyphens"] == 2
    assert f["num_digits"] == 2
    assert abs(f["digit_ratio"] - 2 / 10) < 1e-9


def test_host_level_subdomain_depth():
    assert extract_features("https://a.b.example.com", "host")["subdomain_depth"] == 2
    assert extract_features("https://example.com", "host")["subdomain_depth"] == 0


def test_feature_names_match_extract_order():
    for level in ("domain", "host"):
        assert list(extract_features("example.com", level)) == feature_names(level)
    assert "subdomain_depth" not in feature_names("domain")


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
