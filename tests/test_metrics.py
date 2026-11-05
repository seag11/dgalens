import math
from itertools import pairwise

import pytest

from dgalens.metrics import evaluate, recall_at_fpr, roc_auc, roc_curve, threshold_at_fpr


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


BENIGN_10 = [-1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8, -5.0]


def test_threshold_at_fpr_flags_at_most_the_target_share() -> None:
    # 10% of 10 benign scores allows one false positive: the -5.0 outlier.
    threshold = threshold_at_fpr(BENIGN_10, 0.1)
    assert sum(s < threshold for s in BENIGN_10) == 1
    assert sum(s < threshold_at_fpr(BENIGN_10, 0.0) for s in BENIGN_10) == 0


def test_recall_at_fpr() -> None:
    malicious = [-6.0, -4.0, -1.75, -0.5]
    assert recall_at_fpr(BENIGN_10, malicious, 0.0) == 0.25  # only below -5.0
    assert recall_at_fpr(BENIGN_10, malicious, 0.1) == 0.5  # below -1.8


def test_infinite_scores_are_never_flagged() -> None:
    benign = [-1.0, math.inf, math.inf]
    assert recall_at_fpr(benign, [math.inf], 0.5) == 0.0


def test_roc_curve_endpoints_and_monotonicity() -> None:
    points = roc_curve(benign=[-1.0, -2.0, -2.0], malicious=[-3.0, -2.0])
    assert points[0] == (0.0, 0.0) and points[-1] == (1.0, 1.0)
    assert points == sorted(points)
    # The tie at -2.0 moves both rates in one step.
    assert (2 / 3, 1.0) in points and (0.0, 0.5) in points


def test_roc_curve_area_matches_roc_auc() -> None:
    benign, malicious = [-1.0, -2.2, -1.5, -3.0], [-3.5, -2.0, -2.5, -0.5]
    points = roc_curve(benign, malicious)
    area = sum((x2 - x1) * (y1 + y2) / 2 for (x1, y1), (x2, y2) in pairwise(points))
    assert math.isclose(area, roc_auc(benign, malicious))
