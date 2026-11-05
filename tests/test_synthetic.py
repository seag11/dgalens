import string

import pytest

from dgalens.synthetic import CONSONANTS, STYLES, VOWELS, WORDS, generate


@pytest.mark.parametrize("style", sorted(STYLES))
def test_generation_is_reproducible(style: str) -> None:
    assert generate(style, 50, seed=1) == generate(style, 50, seed=1)
    assert generate(style, 50, seed=1) != generate(style, 50, seed=2)


@pytest.mark.parametrize("style", sorted(STYLES))
def test_labels_are_long_enough_to_score(style: str) -> None:
    assert all(len(label) >= 8 for label in generate(style, 500))


def test_random_styles_follow_their_alphabets() -> None:
    alnum = set(string.ascii_lowercase + string.digits)
    assert all(set(lb) <= alnum and 8 <= len(lb) <= 16 for lb in generate("random-alnum", 500))
    assert all(lb.isalpha() and 8 <= len(lb) <= 16 for lb in generate("random-alpha", 500))


def test_pronounceable_never_stacks_vowels_or_three_consonants() -> None:
    for label in generate("pronounceable", 500):
        assert len(label) <= 14
        assert label[0] in CONSONANTS
        pattern = "".join("v" if c in VOWELS else "c" for c in label)
        assert "vv" not in pattern and "ccc" not in pattern


def test_dictionary_labels_are_two_words() -> None:
    words = set(WORDS)
    for label in generate("dictionary", 200):
        assert any(label[:i] in words and label[i:] in words for i in range(1, len(label)))


def test_word_list_is_clean() -> None:
    assert all(w.isalpha() and w.islower() and 4 <= len(w) <= 8 for w in WORDS)
    assert len(set(WORDS)) == len(WORDS)


def test_unknown_style_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown style"):
        generate("nope", 1)
