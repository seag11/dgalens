"""Character-level Markov model for scoring how "natural" a domain label looks."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dgalens.metrics import threshold_at_fpr

START = "^"
END = "$"

FORMAT_VERSION = 1


class MarkovModel:
    """An order-n character Markov chain with interpolated smoothing.

    Each label is padded with ``order`` start symbols and one end symbol, so the
    model also learns which characters tend to begin and end a label.

    Probabilities are interpolated recursively from the empty context up to the
    full ``order``-character context: at each level,
    ``P(c | h) = (count(h, c) + alpha * P(c | shorter h)) / (count(h) + alpha)``.
    A context the model has rarely or never seen therefore backs off to what
    shorter contexts predict, instead of to a uniform guess.
    """

    def __init__(self, order: int = 3, alpha: float = 1.0, min_length: int = 6) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if min_length < 1:
            raise ValueError("min_length must be >= 1")
        self.order = order
        self.alpha = alpha
        self.min_length = min_length
        # Keyed by context of every length 0..order ("" holds unigram counts).
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.context_totals: Counter[str] = Counter()
        self.vocab: set[str] = {END}
        self.threshold: float | None = None

    def fit(self, labels: Iterable[str]) -> MarkovModel:
        """Count character transitions over ``labels``. Can be called repeatedly."""
        for label in labels:
            if not label:
                continue
            self.vocab.update(label)
            for context, char in self._transitions(label):
                for k in range(self.order + 1):
                    suffix = context[self.order - k :]
                    self.counts[suffix][char] += 1
                    self.context_totals[suffix] += 1
        return self

    def prob(self, context: str, char: str) -> float:
        """P(``char`` | the last ``order`` characters of ``context``)."""
        context = context[-self.order :] if len(context) >= self.order else context
        p = 1.0 / len(self.vocab)
        for k in range(len(context) + 1):
            suffix = context[len(context) - k :]
            seen = self.counts.get(suffix)
            count = seen[char] if seen else 0
            p = (count + self.alpha * p) / (self.context_totals.get(suffix, 0) + self.alpha)
        return p

    def log_prob(self, label: str) -> float:
        """Total natural-log probability of ``label`` under the model."""
        return sum(math.log(self.prob(context, char)) for context, char in self._transitions(label))

    def score(self, label: str) -> float:
        """Mean log probability per transition. Higher means more natural.

        Normalising by length keeps long and short labels comparable.
        """
        if not label:
            raise ValueError("cannot score an empty label")
        return self.log_prob(label) / (len(label) + 1)

    def is_scoreable(self, label: str) -> bool:
        """Whether ``label`` is long enough to judge. Very short labels (``"x"``,
        ``"k4g"``) carry too little signal and are never flagged."""
        return len(label) >= self.min_length

    def detection_score(self, label: str) -> float:
        """``score(label)``, or +inf for labels too short to judge.

        This is the score that thresholds apply to: no threshold flags +inf,
        so short labels are never flagged anywhere the model is used.
        """
        return self.score(label) if self.is_scoreable(label) else math.inf

    def calibrate(self, labels: Iterable[str], false_positive_rate: float = 0.01) -> float:
        """Set ``threshold`` so that at most ``false_positive_rate`` of ``labels``
        (assumed benign) would be flagged.

        The rate is over all labels, including short ones that are never
        flagged, so it means the same thing here as in ``dgalens eval``.
        """
        if not 0 < false_positive_rate < 1:
            raise ValueError("false_positive_rate must be between 0 and 1")
        scores = [self.detection_score(label) for label in labels if label]
        if not scores:
            raise ValueError("no labels to calibrate on")
        threshold = threshold_at_fpr(scores, false_positive_rate)
        if math.isinf(threshold):
            raise ValueError("too few labels long enough to score; cannot calibrate")
        self.threshold = threshold
        return threshold

    def is_suspicious(self, label: str) -> bool:
        if self.threshold is None:
            raise RuntimeError("model has no threshold; call calibrate() first")
        return self.detection_score(label) < self.threshold

    def _transitions(self, label: str) -> Iterable[tuple[str, str]]:
        padded = START * self.order + label + END
        for i in range(self.order, len(padded)):
            yield padded[i - self.order : i], padded[i]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "order": self.order,
            "alpha": self.alpha,
            "min_length": self.min_length,
            "threshold": self.threshold,
            "vocab": sorted(self.vocab),
            "counts": {ctx: dict(chars) for ctx, chars in self.counts.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MarkovModel:
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported model format: {data.get('format_version')!r}")
        model = cls(order=int(data["order"]), alpha=float(data["alpha"]), min_length=int(data["min_length"]))
        threshold = data.get("threshold")
        model.threshold = None if threshold is None else float(threshold)
        model.vocab = set(data["vocab"])
        for ctx, chars in data["counts"].items():
            model.counts[ctx] = Counter(chars)
            model.context_totals[ctx] = sum(chars.values())
        return model

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> MarkovModel:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
