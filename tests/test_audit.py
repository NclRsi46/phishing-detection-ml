"""
Checks for src/ngram_audit.py. Run with:
    python tests/test_audit.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ngram_audit import allegro_metrics, repeated_names


def test_allegro_metrics_need_allegro_rows():
    # Without any, the Allegro numbers are nan, which a mean over splits
    # would silently skip.
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    try:
        allegro_metrics(y, s, np.array([False, False, False, False]))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_allegro_share_is_undefined_when_nothing_is_caught():
    # The top-scored row is benign, so nothing is caught at 1% FPR and the
    # Allegro share of the catches has no value. 0 would pull the mean down.
    y = np.array([0, 0, 1, 1])
    s = np.array([0.95, 0.2, 0.8, 0.9])
    metrics = allegro_metrics(y, s, np.array([False, False, True, False]))
    assert math.isnan(metrics["allegro_share_of_1pct_fpr_catches"])


def test_repeated_names_flags_shared_kit_wording():
    # Names without the suffix, as the audit passes them. Different
    # digits, same kit wording.
    train = ["pl-oferta48213", "example"]
    test = ["pl-oferta77120", "news-site"]
    assert repeated_names(train, test).tolist() == [True, False]


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
