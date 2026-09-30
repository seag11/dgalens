import math

import pytest

from dgalens.model import MarkovModel


@pytest.mark.parametrize("context", ["^^a", "abc", "zzz", ""])
def test_probabilities_over_vocab_sum_to_one(context: str) -> None:
    model = MarkovModel(order=3, alpha=0.5).fit(["abcd", "abce", "bcd"])
    assert math.isclose(sum(model.prob(context, c) for c in model.vocab), 1.0)


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
