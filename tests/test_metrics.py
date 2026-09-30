import math

import pytest

from dgalens.metrics import evaluate, roc_auc


def test_perfect_separation() -> None:
    assert roc_auc(benign=[-1.0, -1.5], malicious=[-3.0, -4.0]) == 1.0


def test_inverted_separation() -> None:
    assert roc_auc(benign=[-3.0, -4.0], malicious=[-1.0, -1.5]) == 0.0


def test_ties_count_half() -> None:
    assert roc_auc(benign=[-2.0], malicious=[-2.0]) == 0.5


def test_mixed_case() -> None:
    # Pairs (mal, ben) with mal < ben: (-3,-2), (-3,-1), (-1.5,-1) -> 3 of 4.
    assert math.isclose(roc_auc(benign=[-2.0, -1.0], malicious=[-3.0, -1.5]), 0.75)


def test_roc_auc_needs_both_classes() -> None:
    with pytest.raises(ValueError):
        roc_auc(benign=[], malicious=[-1.0])


def test_evaluate_counts() -> None:
    report = evaluate(benign=[-1.0, -2.5, -1.2], malicious=[-3.0, -2.8, -1.1], threshold=-2.0)
    assert (report.true_positives, report.false_negatives) == (2, 1)
    assert (report.false_positives, report.true_negatives) == (1, 2)
    assert math.isclose(report.precision, 2 / 3)
    assert math.isclose(report.recall, 2 / 3)
    assert math.isclose(report.false_positive_rate, 1 / 3)
