"""Detection metrics, kept dependency-free."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Report:
    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    roc_auc: float

    @property
    def precision(self) -> float:
        flagged = self.true_positives + self.false_positives
        return self.true_positives / flagged if flagged else 0.0

    @property
    def recall(self) -> float:
        malicious = self.true_positives + self.false_negatives
        return self.true_positives / malicious if malicious else 0.0

    @property
    def false_positive_rate(self) -> float:
        benign = self.false_positives + self.true_negatives
        return self.false_positives / benign if benign else 0.0


def evaluate(benign: list[float], malicious: list[float], threshold: float) -> Report:
    """Summarise detection at ``threshold``, where a score below it is flagged."""
    tp = sum(s < threshold for s in malicious)
    fp = sum(s < threshold for s in benign)
    return Report(
        threshold=threshold,
        true_positives=tp,
        false_positives=fp,
        true_negatives=len(benign) - fp,
        false_negatives=len(malicious) - tp,
        roc_auc=roc_auc(benign, malicious),
    )


def threshold_at_fpr(benign: list[float], false_positive_rate: float) -> float:
    """The threshold that flags at most ``false_positive_rate`` of ``benign``.

    Uses the same rule as ``MarkovModel.calibrate``: scores strictly below the
    returned value are flagged.
    """
    if not benign:
        raise ValueError("need at least one benign score")
    if not 0 <= false_positive_rate < 1:
        raise ValueError("false_positive_rate must be in [0, 1)")
    ordered = sorted(benign)
    return ordered[int(len(ordered) * false_positive_rate)]


def recall_at_fpr(benign: list[float], malicious: list[float], false_positive_rate: float) -> float:
    """Share of ``malicious`` flagged at the threshold from ``threshold_at_fpr``."""
    if not malicious:
        raise ValueError("need at least one malicious score")
    threshold = threshold_at_fpr(benign, false_positive_rate)
    return sum(s < threshold for s in malicious) / len(malicious)


def roc_curve(benign: list[float], malicious: list[float]) -> list[tuple[float, float]]:
    """``(false positive rate, true positive rate)`` points, from (0, 0) to (1, 1).

    Sweeps the threshold upward through every distinct score, flagging each
    score at or below it. Tied scores move together, as one point.
    """
    if not benign or not malicious:
        raise ValueError("need at least one benign and one malicious score")
    events = sorted([(s, False) for s in benign] + [(s, True) for s in malicious])
    points = [(0.0, 0.0)]
    fp = tp = 0
    i = 0
    while i < len(events):
        score = events[i][0]
        while i < len(events) and events[i][0] == score:
            if events[i][1]:
                tp += 1
            else:
                fp += 1
            i += 1
        points.append((fp / len(benign), tp / len(malicious)))
    return points


def roc_auc(benign: list[float], malicious: list[float]) -> float:
    """Probability that a random malicious label scores lower than a random benign one.

    Computed from ranks (Mann-Whitney U), with ties counted as half.
    """
    if not benign or not malicious:
        raise ValueError("need at least one benign and one malicious score")
    ranked = sorted([(s, 1) for s in malicious] + [(s, 0) for s in benign])
    rank_sum = 0.0
    i = 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        average_rank = (i + j + 1) / 2  # 1-based average rank of the tie group
        rank_sum += average_rank * sum(label for _, label in ranked[i:j])
        i = j
    n_mal, n_ben = len(malicious), len(benign)
    # U counts malicious-above-benign pairs; we want malicious-below-benign.
    u = rank_sum - n_mal * (n_mal + 1) / 2
    return 1 - u / (n_mal * n_ben)
