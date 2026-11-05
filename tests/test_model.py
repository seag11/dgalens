import math

import pytest

from dgalens.model import MarkovModel


@pytest.mark.parametrize("context", ["^^a", "abc", "zzz", ""])
def test_probabilities_over_vocab_sum_to_one(context: str) -> None:
    model = MarkovModel(order=3, alpha=0.5).fit(["abcd", "abce", "bcd"])
    assert math.isclose(sum(model.prob(context, c) for c in model.vocab), 1.0)


def test_prob_matches_hand_computed_interpolation() -> None:
    # Corpus "ab", "ab", "b" at order 1, alpha 1. Padded: ^ab$, ^ab$, ^b$.
    # Transitions: ^->a x2, a->b x2, b->$ x3, ^->b x1. Vocab {a, b, $}: uniform 1/3.
    # Unigram counts (context ""): a=2, b=3, $=3, total 8.
    model = MarkovModel(order=1, alpha=1.0).fit(["ab", "ab", "b"])
    p_unigram_b = (3 + 1 / 3) / (8 + 1)  # = 10/27
    p_a_then_b = (2 + p_unigram_b) / (2 + 1)  # context "a": b seen 2 of 2
    assert math.isclose(model.prob("a", "b"), p_a_then_b)
    p_unigram_a = (2 + 1 / 3) / (8 + 1)  # = 7/27
    p_start_then_a = (2 + p_unigram_a) / (3 + 1)  # context "^": a seen 2 of 3
    assert math.isclose(model.prob("^", "a"), p_start_then_a)
    # Unseen context "z" falls straight back to the unigram estimate.
    assert math.isclose(model.prob("z", "a"), p_unigram_a)


def test_calibrate_counts_short_labels_in_the_rate() -> None:
    model = MarkovModel(order=2, min_length=6).fit(["garden", "forest", "meadow"])
    # 10 labels, 3 of them short ("ab", "cd", "garde") and never flagged. At 10%, one may be.
    labels = ["ab", "cd", "garden", "forest", "meadow", "gardens", "forests", "meadows", "qzxjvkw", "garde"]
    model.calibrate(labels, false_positive_rate=0.1)
    assert sum(map(model.is_suspicious, labels)) == 1
    assert model.is_suspicious("qzxjvkw")


def test_calibrate_rejects_mostly_short_labels() -> None:
    model = MarkovModel(order=2).fit(["garden"])
    with pytest.raises(ValueError, match="too few"):
        model.calibrate(["ab", "cd", "ef", "garden"], false_positive_rate=0.5)


def test_unseen_context_backs_off_to_shorter_context() -> None:
    # "xbc" never occurs, but "bc" is always followed by "d".
    model = MarkovModel(order=3, alpha=0.5).fit(["abcd", "abcd", "abcd", "aaaa"])
    assert model.prob("xbc", "d") > model.prob("xbc", "a")


def test_seen_label_scores_higher_than_unseen(benign: list[str]) -> None:
    model = MarkovModel(order=2).fit(benign)
    assert model.score("garden") > model.score("qzxjvk")


def test_random_labels_score_lower_on_average(
    trained: MarkovModel, benign: list[str], random_labels: list[str]
) -> None:
    mean_benign = sum(map(trained.score, benign)) / len(benign)
    mean_random = sum(map(trained.score, random_labels)) / len(random_labels)
    assert mean_random < mean_benign


def test_short_labels_are_never_flagged(trained: MarkovModel) -> None:
    assert not trained.is_scoreable("qzx")
    assert not trained.is_suspicious("qzx")


def test_calibrate_flags_roughly_target_share(benign: list[str]) -> None:
    model = MarkovModel(order=2).fit(benign)
    model.calibrate(benign, false_positive_rate=0.1)
    flagged = sum(model.is_suspicious(label) for label in benign)
    assert flagged <= len(benign) * 0.1


def test_is_suspicious_requires_threshold() -> None:
    with pytest.raises(RuntimeError):
        MarkovModel().fit(["abc"]).is_suspicious("abc")


def test_empty_label_cannot_be_scored() -> None:
    with pytest.raises(ValueError):
        MarkovModel().fit(["abc"]).score("")


@pytest.mark.parametrize("kwargs", [{"order": 0}, {"alpha": 0}, {"min_length": 0}])
def test_rejects_invalid_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        MarkovModel(**kwargs)  # type: ignore[arg-type]


def test_save_and_load_round_trip(tmp_path, trained: MarkovModel) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "model.json"
    trained.save(path)
    loaded = MarkovModel.load(path)
    assert loaded.order == trained.order
    assert loaded.threshold == trained.threshold
    assert loaded.min_length == trained.min_length
    for label in ["garden", "qzxjvk", "a"]:
        assert math.isclose(loaded.score(label), trained.score(label))
