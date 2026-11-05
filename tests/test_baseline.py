import math

import pytest

from dgalens.baseline import entropy_score


def test_repeated_character_has_zero_entropy() -> None:
    assert entropy_score("aaaaaa") == 0


def test_distinct_characters_have_log2_n_bits() -> None:
    assert math.isclose(entropy_score("abcd"), -2.0)


def test_more_varied_labels_score_lower() -> None:
    assert entropy_score("q8f7g2kd9s1m") < entropy_score("google")


def test_empty_label_cannot_be_scored() -> None:
    with pytest.raises(ValueError):
        entropy_score("")
