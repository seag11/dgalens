"""A deliberately simple comparison point for the Markov model."""

from __future__ import annotations

import math
from collections import Counter


def entropy_score(label: str) -> float:
    """Negative Shannon entropy of the label's characters, in bits.

    Uses the same convention as ``MarkovModel.score``: lower means more
    random-looking. It ignores character order entirely, so it is a measure of
    how much the Markov model gains by modelling transitions.
    """
    if not label:
        raise ValueError("cannot score an empty label")
    n = len(label)
    return sum(c / n * math.log2(c / n) for c in Counter(label).values())
