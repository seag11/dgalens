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
