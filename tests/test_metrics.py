"""
Checks for the operating-point metric in src/evaluate.py and the tie count
in src/repeat_splits.py. Run with:
    python tests/test_metrics.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from evaluate import recall_at_fpr
from repeat_splits import top_tie


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


def test_never_goes_over_the_target_by_a_small_step():
    # 100 benign rows, so each one is 1% FPR. Catching all phishing costs
    # a second false positive (2% FPR), which is over a 1% target.
    y = [0] * 100 + [1] * 100
    s = [0.995, 0.985] + [0.1] * 98 + [0.99] * 50 + [0.98] * 50
    assert recall_at_fpr(y, s, 0.01) == 0.5


def test_needs_both_classes():
    for y in ([0, 0, 0], [1, 1, 1]):
        try:
            recall_at_fpr(y, [0.2, 0.5, 0.9], 0.01)
        except ValueError:
            continue
        raise AssertionError(f"no error for labels {y}")


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


def test_top_tie_counts_rows_at_the_highest_score():
    # Two benign and three phishing rows share the top score; with four
    # benign rows a 25% FPR allows only one of them, so recall there is 0.
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([1.0, 1.0, 0.2, 0.1, 1.0, 1.0, 1.0, 0.5])
    assert top_tie(y, s) == {"benign": 2, "phishing": 3}
    assert recall_at_fpr(y, s, 0.25) == 0.0
    # Only the highest score counts, not the next one down.
    assert top_tie(np.array([0, 1, 1]), np.array([0.9, 1.0, 0.8])) == {"benign": 0, "phishing": 1}


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
