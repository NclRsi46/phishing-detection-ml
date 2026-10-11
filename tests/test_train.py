"""
Checks for the models in src/train.py. Run with:
    python tests/test_train.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from train import make_models


def test_random_forest_scores_are_repeatable():
    # Scores must be bit-for-bit the same on every call. Tiny differences
    # break ties between rows differently, which moves recall at a fixed
    # false positive rate and can flip a row scored exactly 0.5.
    # Discrete features, like the real ones, so identical rows with mixed
    # labels end in impure leaves and trees return fractions, not just 0 or 1.
    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, size=(3000, 3))
    y = (X[:, 0] + rng.normal(size=3000) > 1).astype(int)
    model = make_models()["random_forest"].fit(X[:2000], y[:2000])
    first = model.predict_proba(X[2000:])
    for _ in range(10):
        assert (model.predict_proba(X[2000:]) == first).all()


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
