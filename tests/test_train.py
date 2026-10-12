"""
Checks for the models in src/train.py. Run with:
    python tests/test_train.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from train import NGRAM_MODEL, make_models, model_columns


def test_ngram_model_reads_the_name_for_its_level():
    # Reading the domain column in the host experiment would silently drop
    # every subdomain.
    assert model_columns(NGRAM_MODEL, "domain") == ["domain"]
    assert model_columns(NGRAM_MODEL, "host") == ["host"]


def test_ngram_vocabulary_comes_from_training_rows_only():
    # Fitting TF-IDF on every row before the split would let test names
    # shape the features. The vectorizer is inside the model, so it only
    # learns from the rows the model is trained on.
    train = pd.DataFrame({
        "domain": ["paypal-login.xyz", "paypal-verify.xyz", "example.org", "examples.org"],
        "label": [1, 1, 0, 0],
    })
    model = make_models("domain")[NGRAM_MODEL].fit(train[["domain"]], train["label"])
    vocabulary = model[0].named_transformers_["ngrams"].vocabulary_
    assert "paypa" in vocabulary
    assert "zzq" not in vocabulary
    # Scoring a name must not add it to the vocabulary either.
    model.predict_proba(pd.DataFrame({"domain": ["zzqzzq.top"]}))
    assert "zzq" not in vocabulary


def test_random_forest_scores_are_repeatable():
    # Scores must be bit-for-bit the same on every call. Tiny differences
    # break ties between rows differently, which moves recall at a fixed
    # false positive rate and can flip a row scored exactly 0.5.
    # Discrete features, like the real ones, so identical rows with mixed
    # labels end in impure leaves and trees return fractions, not just 0 or 1.
    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, size=(3000, 3))
    y = (X[:, 0] + rng.normal(size=3000) > 1).astype(int)
    model = make_models("domain")["random_forest"].fit(X[:2000], y[:2000])
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
