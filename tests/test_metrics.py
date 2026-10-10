"""
Checks for the operating-point metric in src/evaluate.py. Run with:
    python tests/test_metrics.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate import recall_at_fpr


def test_perfect_separation():
    assert recall_at_fpr([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.01) == 1.0


def test_never_goes_over_the_target():
    # Benign scores 0.95, 0.3, 0.2, 0.1; phishing 0.9, 0.8, 0.7, 0.05.
    # The top-scored row is benign, so any threshold that catches phishing
    # already has an FPR of 0.25.
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    s = [0.95, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.05]
    assert recall_at_fpr(y, s, 0.2) == 0.0
    # An FPR exactly at the target is allowed.
    assert recall_at_fpr(y, s, 0.25) == 0.75


def test_tied_scores_move_together():
    # The phishing and benign rows tied at 0.5 can't be separated by any
    # threshold, so at a low target only the 0.9 row is caught.
    y = [1, 1, 0, 1, 0]
    s = [0.9, 0.5, 0.5, 0.4, 0.1]
    assert abs(recall_at_fpr(y, s, 0.1) - 1 / 3) < 1e-9


def test_keeps_points_on_a_straight_stretch_of_the_curve():
    # Three tie groups give ROC points (1/3, 1/3), (2/3, 2/3), (1, 1). The
    # middle one lies on a straight line, which sklearn drops by default,
    # and it's the best point under a 0.7 target.
    y = [1, 0, 1, 0, 1, 0]
    s = [0.9, 0.9, 0.6, 0.6, 0.3, 0.3]
    assert abs(recall_at_fpr(y, s, 0.7) - 2 / 3) < 1e-9


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
