"""Synthetic generated-domain labels in a few broad styles, for evaluation.

These are style proxies written for this project, not reimplementations of any
real generator. Each spec below was fixed before evaluation results were seen;
changing one changes the published numbers, so treat them as frozen.

- ``random-alnum``:  uniform ``[a-z0-9]``, length uniform in 8-16.
- ``random-alpha``:  uniform ``[a-z]``, length uniform in 8-16.
- ``pronounceable``: consonant-vowel syllables (each consonant uniform over the
  21 consonants, each vowel uniform over ``aeiou``, a closing consonant with
  probability 0.3), appended until the label reaches a target length uniform in
  8-14, then truncated to it.
- ``dictionary``:    two words drawn uniformly from ``WORDS`` and joined.
"""

from __future__ import annotations

import random
import string
from collections.abc import Callable

VOWELS = "aeiou"
CONSONANTS = "".join(c for c in string.ascii_lowercase if c not in VOWELS)

# Common English words, 4-8 letters, written for this project. Kept as one string
# because a one-word-per-line list literal would run to hundreds of lines.
WORDS = (  # noqa: SIM905
    "able acid aged also area army away baby back ball band bank base bath bear beat bell belt best bird "
    "blue boat body bone book boot born boss both bowl bulk burn bush busy cake call calm camp card care "
    "case cash cast cell chat chip city club coal coat code cold cook cool copy core cost crew crop dark "
    "data date dawn deal dear deep desk diet disk door down draw drop drum dust duty earn east easy edge "
    "face fact fair fall farm fast fear feel file fill film find fine fire firm fish flag flat flow food "
    "foot form fort free fuel full fund gain game gate gift girl glad goal gold golf good grey grow half "
    "hall hand hard head hear heat help hero high hill hold home hope host hour huge idea iron item join "
    "jump keep kind king lake land last late lead leaf lift light line link list live load lock long loop "
    "lord loud love luck main make mark mass meal meet mile milk mind mint miss mode moon more move much "
    "name near neck need news next nice node nose note open page pain pair park part pass path peak pick "
    "pine pink plan play plot plus pool port post pure push race rail rain rank rate read real rest rice "
    "rich ride ring rise risk road rock role roof room root rose rule safe sale salt sand save seat seed "
    "self sell send ship shop show side sign silk site size skin slow snow soft soil song sort soul spot "
    "star stay step stop sure tail take talk tall tank tape task team tell tend term test text thin tide "
    "tile time tiny tone tool tour town tree trip true tune turn type unit user vast view vote wait wake "
    "walk wall want warm wash wave weak wear week well west wide wild wind wine wing wire wise wish wood "
    "word work yard year zero zone "
    "anchor basket beacon bridge bronze bubble button candle canyon carbon carpet circle clever cloudy "
    "copper cotton crystal dinner dragon effort engine falcon fabric farmer flower frozen galaxy garden "
    "gentle ginger golden guitar hammer helmet hidden honest island jacket jungle kettle ladder legend "
    "lemon letter little magnet marble meadow mirror modern motion needle orange oxygen paddle pepper "
    "pillow pirate pocket purple puzzle rabbit rocket saddle salmon secret shadow silver simple sketch "
    "spider spring stable summer sunset tablet temple thread ticket timber tunnel turtle velvet violet "
    "wallet window winter wonder yellow anthem balance captain chapter compass dolphin feather freedom "
    "harvest horizon journey kingdom lantern machine mustard network orchard pattern pioneer quantum "
    "rainbow sapphire station thunder trumpet uniform venture voyage whisper"
).split()

Generator = Callable[[random.Random], str]


def random_alnum(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_lowercase + string.digits, k=rng.randint(8, 16)))


def random_alpha(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_lowercase, k=rng.randint(8, 16)))


def pronounceable(rng: random.Random) -> str:
    target = rng.randint(8, 14)
    label = ""
    while len(label) < target:
        label += rng.choice(CONSONANTS) + rng.choice(VOWELS)
        if rng.random() < 0.3:
            label += rng.choice(CONSONANTS)
    return label[:target]


def dictionary(rng: random.Random) -> str:
    return rng.choice(WORDS) + rng.choice(WORDS)


STYLES: dict[str, Generator] = {
    "random-alnum": random_alnum,
    "random-alpha": random_alpha,
    "pronounceable": pronounceable,
    "dictionary": dictionary,
}


def generate(style: str, count: int, seed: int = 0) -> list[str]:
    """Return ``count`` labels in ``style``, reproducibly for a given ``seed``."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; expected one of {sorted(STYLES)}")
    rng = random.Random(f"{style}:{seed}")
    return [STYLES[style](rng) for _ in range(count)]
