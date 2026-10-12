"""
Checks for src/predict.py. Run with:
    python tests/test_predict.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from features import hostname, registered_domain
from predict import score
from train import NGRAM_MODEL, make_models, model_columns

URLS = [
    ("https://paypal-login.xyz", 1),
    ("https://paypal-verify.xyz", 1),
    ("https://secure-login.top", 1),
    ("https://example.org", 0),
    ("https://examples.org", 0),
    ("https://news.org", 0),
]


def ngram_bundle(level: str) -> dict:
    name_of = registered_domain if level == "domain" else hostname
    train = pd.DataFrame({level: [name_of(url) for url, _ in URLS], "label": [label for _, label in URLS]})
    columns = model_columns(NGRAM_MODEL, level)
    model = make_models(level)[NGRAM_MODEL].fit(train[columns], train["label"])
    return {"model": model, "level": level, "features": columns}


def test_ngram_model_ignores_path_and_query():
    # The n-gram model reads text, so the path must not reach it either.
    for level in ("domain", "host"):
        bundle = ngram_bundle(level)
        a = score(bundle, "http://www.paypal-secure.xyz")
        b = score(bundle, "https://www.paypal-secure.xyz/login.php?id=1#top")
        assert a == b, level
        # A model that gave every name the same score would pass the check
        # above without reading anything.
        assert a != score(bundle, "http://www.example.org"), level


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
