"""Reading domain lists from disk."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from dgalens.domains import normalize


def read_domains(path: str | Path) -> Iterator[str]:
    """Yield raw domains from a file with one per line.

    Blank lines and ``#`` comments are skipped. For CSV rows such as the
    Tranco list's ``rank,domain`` format, the last field is used.
    """
    with Path(path).open(encoding="utf-8") as f:
        yield from parse_lines(f)


def parse_lines(lines: Iterable[str]) -> Iterator[str]:
    """Yield the domain from each line, using the rules described in ``read_domains``."""
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        yield line.rsplit(",", 1)[-1].strip()


def read_labels(path: str | Path) -> list[str]:
    """Read a domain list and normalise it to scoreable labels, dropping empties."""
    return [label for label in map(normalize, read_domains(path)) if label]
