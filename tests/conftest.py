import random
import string

import pytest

from dgalens.model import MarkovModel

BENIGN = [
    "google",
    "facebook",
    "amazon",
    "wikipedia",
    "twitter",
    "linkedin",
    "netflix",
    "microsoft",
    "apple",
    "github",
    "stackoverflow",
    "reddit",
    "youtube",
    "instagram",
    "weather",
    "mountain",
    "garden",
    "kitchen",
    "travel",
    "market",
    "coffee",
    "bakery",
    "library",
    "science",
    "energy",
    "network",
    "ocean",
    "forest",
    "rocket",
    "planet",
    "silver",
    "bright",
    "castle",
    "harbor",
    "village",
    "summer",
    "winter",
    "autumn",
    "spring",
    "river",
    "valley",
    "meadow",
    "sunset",
]


@pytest.fixture
def benign() -> list[str]:
    return list(BENIGN)


@pytest.fixture
def random_labels() -> list[str]:
    rng = random.Random(42)
    alphabet = string.ascii_lowercase + string.digits
    return ["".join(rng.choices(alphabet, k=rng.randint(10, 16))) for _ in range(40)]


@pytest.fixture
def trained(benign: list[str]) -> MarkovModel:
    model = MarkovModel(order=2).fit(benign)
    model.calibrate(benign, false_positive_rate=0.05)
    return model
