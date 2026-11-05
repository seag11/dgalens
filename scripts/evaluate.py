"""Evaluate a trained model per generator style, against an entropy baseline.

Usage: python scripts/evaluate.py [--model models/dgalens.json] [--benign data/benign-test.csv]
                                  [--count 5000] [--seed 0] [--plot docs/roc.png]

Prints a Markdown table for the README. With --plot, also draws ROC curves
(needs the optional ``eval`` dependency group: ``uv run --group eval ...``).

Labels shorter than the model's ``min_length`` can never be flagged, by either
method, so they enter the benign scores as +inf. Generated labels are all
8+ characters, so they are always scored.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from pathlib import Path

from dgalens.baseline import entropy_score
from dgalens.data import read_labels
from dgalens.metrics import recall_at_fpr, roc_auc, roc_curve
from dgalens.model import MarkovModel
from dgalens.synthetic import STYLES, generate

FPRS = (0.01, 0.001)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, default=Path("models/dgalens.json"))
    parser.add_argument("--benign", type=Path, default=Path("data/benign-test.csv"))
    parser.add_argument("--count", type=int, default=5000, help="Labels generated per style.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", type=Path, help="Write ROC curves to this image file.")
    args = parser.parse_args()

    model = MarkovModel.load(args.model)
    if model.threshold is None:
        parser.error("model has no threshold")

    def entropy_detection_score(label: str) -> float:
        # Same short-label rule as the model, so both methods see the same benign set.
        return entropy_score(label) if model.is_scoreable(label) else math.inf

    scorers: dict[str, Callable[[str], float]] = {
        "Markov": model.detection_score,
        "Entropy": entropy_detection_score,
    }
    benign_labels = read_labels(args.benign)
    benign = {name: list(map(score, benign_labels)) for name, score in scorers.items()}
    generated = {style: generate(style, args.count, args.seed) for style in STYLES}

    calibrated_fpr = sum(s < model.threshold for s in benign["Markov"]) / len(benign_labels)
    print(
        f"{len(benign_labels):,} benign test labels; {args.count:,} generated per style (seed {args.seed}).\n"
        f"Calibrated Markov threshold {model.threshold:.4f} flags {calibrated_fpr:.2%} of benign.\n"
    )
    header = ["Style", "Method", "ROC AUC", *(f"Recall @ {f:.1%} FPR" for f in FPRS), "Recall @ threshold"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join([" --- "] * 2 + [" ---: "] * (len(header) - 2)) + "|")

    curves: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for style, labels in generated.items():
        for name, score in scorers.items():
            malicious = list(map(score, labels))
            row = [style, name, f"{roc_auc(benign[name], malicious):.3f}"]
            row += [f"{recall_at_fpr(benign[name], malicious, f):.1%}" for f in FPRS]
            at_threshold = sum(s < model.threshold for s in malicious) / len(malicious)
            row.append(f"{at_threshold:.1%}" if name == "Markov" else "n/a")
            print("| " + " | ".join(row) + " |")
            curves.setdefault(style, {})[name] = roc_curve(benign[name], malicious)

    if args.plot:
        plot(curves, args.plot, floor=1 / len(benign_labels))
        print(f"\nwrote {args.plot}")


def plot(curves: dict[str, dict[str, list[tuple[float, float]]]], path: Path, floor: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    colors = {"Markov": "#2a78d6", "Entropy": "#eb6834"}
    styles = {"Markov": "-", "Entropy": "--"}

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "axes.labelcolor": muted})
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True, facecolor=surface)
    for ax, (style, by_method) in zip(axes.flat, curves.items(), strict=True):
        ax.set_facecolor(surface)
        for fpr in FPRS:
            ax.axvline(fpr, color=grid, linewidth=1, zorder=0)
        for name, points in by_method.items():
            # Points at FPR 0 can't sit on a log axis; start the curve at one benign label.
            xs = [max(x, floor) for x, _ in points]
            ys = [y for _, y in points]
            ax.plot(xs, ys, styles[name], color=colors[name], linewidth=2, label=name)
        ax.set_xscale("log")
        ax.set_xlim(floor, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(style, color=ink, loc="left", fontsize=11)
        ax.tick_params(colors=muted, length=0)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("False positive rate on benign test labels (log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Recall")
    axes[0][0].legend(frameon=False, loc="upper left", labelcolor=ink)
    fig.suptitle("ROC by generator style: Markov model vs. entropy baseline", color=ink, x=0.06, ha="left")
    fig.text(0.06, 0.925, "Vertical lines mark 1% and 0.1% FPR.", color=muted, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=surface)


if __name__ == "__main__":
    main()
