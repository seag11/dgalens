"""Download the Tranco top-sites list and split it into benign train/test files.

Usage: python scripts/fetch_tranco.py [--train 100000] [--test 20000] [--out data]

The test slice comes from ranks directly after the training slice, so it is
never seen during training. See https://tranco-list.eu for the list's terms.
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train", type=int, default=100_000, help="Top-ranked domains to train on.")
    parser.add_argument("--test", type=int, default=20_000, help="Following domains to hold out.")
    parser.add_argument("--out", type=Path, default=Path("data"), help="Output directory.")
    args = parser.parse_args()

    print(f"downloading {TRANCO_URL} ...")
    with urllib.request.urlopen(TRANCO_URL, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    lines = archive.read("top-1m.csv").decode("utf-8").splitlines()

    args.out.mkdir(parents=True, exist_ok=True)
    splits = {
        "benign-train.csv": lines[: args.train],
        "benign-test.csv": lines[args.train : args.train + args.test],
    }
    for name, rows in splits.items():
        (args.out / name).write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"wrote {len(rows):>7} domains to {args.out / name}")


if __name__ == "__main__":
    main()
