"""
features.py

Feature extraction shared by preprocess.py and predict.py, so training and
prediction can't drift apart.

Only the domain name is used. The path, query, and scheme are thrown away for
every URL, because the benign source (Tranco) has none of them and the first
version of this project learned "has a path = phishing" instead of anything
about phishing.

Two levels:
    "domain"  the registered domain only, e.g. "paypal-login-secure.com".
              Subdomains are dropped because Tranco lists no subdomains.
    "host"    the full hostname, e.g. "paypal.com.account-check.xyz".
              Only fair when the benign source also has real hostnames
              (the Umbrella experiment).
"""

import math
from collections import Counter

import tldextract

# Use the public suffix list bundled with tldextract instead of downloading
# the latest one at runtime, so a domain parses the same way at training time
# and at prediction time.
EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

# Commonly impersonated brands, picked by hand before looking at the data
# (not from PhishTank's "target" column, which would leak labels into the
# feature design). Short brands like "ups" or "chase" are left out because
# they show up inside ordinary words ("groups", "purchase").
BRANDS = [
    "paypal", "microsoft", "office365", "outlook", "amazon", "netflix",
    "google", "facebook", "instagram", "whatsapp", "linkedin", "dropbox",
    "docusign", "adobe", "wellsfargo", "bankofamerica", "usps", "fedex",
    "coinbase", "binance", "metamask",
]

LURE_WORDS = [
    "login", "signin", "secure", "verify", "account", "update", "banking",
    "wallet", "support", "billing", "confirm", "unlock", "recover",
]

LEGACY_TLDS = {"net", "org", "edu", "gov", "mil", "int"}

DOMAIN_FEATURES = [
    "length",
    "num_hyphens",
    "num_digits",
    "digit_ratio",
    "entropy",
    "is_ip",
    "is_punycode",
    "has_brand_word",
    "has_lure_word",
    "tld_com",
    "tld_legacy",
    "tld_country",
    "tld_other",
]

HOST_FEATURES = DOMAIN_FEATURES + ["subdomain_depth"]


def feature_names(level: str) -> list:
    if level == "domain":
        return list(DOMAIN_FEATURES)
    if level == "host":
        return list(HOST_FEATURES)
    raise ValueError(f"Unknown level: {level}")


def _parse(url):
    return EXTRACT(str(url).strip().lower())


def hostname(url) -> str:
    """Full hostname, or "" if the URL has no valid public suffix."""
    ext = _parse(url)
    return ext.ipv4 or ext.ipv6 or ext.fqdn


def registered_domain(url) -> str:
    """Registered domain (domain + public suffix), or the IP for IP hosts."""
    ext = _parse(url)
    return ext.ipv4 or ext.ipv6 or ext.top_domain_under_public_suffix


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    return -sum(c / len(text) * math.log2(c / len(text)) for c in counts.values())


def _tld_group(suffix: str) -> str:
    if not suffix:
        return ""
    last = suffix.split(".")[-1]
    if last == "com":
        return "com"
    if last in LEGACY_TLDS:
        return "legacy"
    if len(last) == 2:
        return "country"
    return "other"


def extract_features(url, level: str = "domain") -> dict:
    ext = _parse(url)
    ip = ext.ipv4 or ext.ipv6

    if ip:
        name, suffix, subdomain = ip, "", ""
    else:
        name, suffix, subdomain = ext.domain, ext.suffix, ext.subdomain

    # The text we measure: the part of the name a registrant chooses,
    # without the public suffix.
    if level == "host" and subdomain:
        text = f"{subdomain}.{name}"
    else:
        text = name

    num_digits = sum(ch.isdigit() for ch in text)
    tld = _tld_group(suffix)

    features = {
        "length": len(text),
        "num_hyphens": text.count("-"),
        "num_digits": num_digits,
        "digit_ratio": num_digits / len(text) if text else 0.0,
        "entropy": shannon_entropy(text),
        "is_ip": int(bool(ip)),
        "is_punycode": int("xn--" in text or "xn--" in suffix),
        # A brand name somewhere other than the brand's own domain,
        # e.g. paypal-login-secure.com or paypal.com.account-check.xyz.
        "has_brand_word": int(any(b in text and name != b for b in BRANDS)),
        "has_lure_word": int(any(w in text for w in LURE_WORDS)),
        "tld_com": int(tld == "com"),
        "tld_legacy": int(tld == "legacy"),
        "tld_country": int(tld == "country"),
        "tld_other": int(tld == "other"),
    }

    if level == "host":
        features["subdomain_depth"] = len(subdomain.split(".")) if subdomain else 0
    elif level != "domain":
        raise ValueError(f"Unknown level: {level}")

    return features
