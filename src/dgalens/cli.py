"""Command-line interface: train, score and evaluate."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Annotated

import typer

from dgalens.data import parse_lines, read_labels
from dgalens.domains import normalize
from dgalens.metrics import evaluate
from dgalens.model import MarkovModel

DEFAULT_MODEL = Path(os.environ.get("DGALENS_MODEL", "models/dgalens.json"))

app = typer.Typer(help="Flag algorithmically generated (DGA) domains with a Markov model.")

ModelOption = Annotated[Path, typer.Option("--model", "-m", help="Model file.")]


@app.command()
def train(
    benign: Annotated[Path, typer.Argument(help="File of benign domains, one per line.")],
    model_path: ModelOption = DEFAULT_MODEL,
    order: Annotated[int, typer.Option(help="Characters of context per prediction.")] = 3,
    alpha: Annotated[float, typer.Option(help="Smoothing strength; higher backs off more.")] = 1.0,
    min_length: Annotated[int, typer.Option(help="Labels shorter than this are never flagged.")] = 6,
    fpr: Annotated[float, typer.Option(help="Target false-positive rate for the threshold.")] = 0.01,
    holdout: Annotated[float, typer.Option(help="Share of labels held out to calibrate.")] = 0.1,
    seed: Annotated[int, typer.Option(help="Seed for the train/calibration split.")] = 0,
) -> None:
    """Train a model on benign domains and calibrate its threshold on a held-out slice."""
    labels = read_labels(benign)
    if len(labels) < 2:
        raise typer.BadParameter(f"need at least 2 domains, found {len(labels)}", param_hint="BENIGN")
    random.Random(seed).shuffle(labels)
    split = max(1, int(len(labels) * holdout))
    calibration, training = labels[:split], labels[split:]

    model = MarkovModel(order=order, alpha=alpha, min_length=min_length).fit(training)
    threshold = model.calibrate(calibration, false_positive_rate=fpr)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    typer.echo(
        f"trained on {len(training)} labels, calibrated on {len(calibration)}; "
        f"threshold={threshold:.4f}; saved to {model_path}"
    )


@app.command()
def score(
    domains: Annotated[
        list[str] | None, typer.Argument(help="Domains to score; reads stdin if omitted.")
    ] = None,
    model_path: ModelOption = DEFAULT_MODEL,
    as_json: Annotated[bool, typer.Option("--json", help="Emit one JSON object per line.")] = False,
    only_suspicious: Annotated[
        bool, typer.Option("--suspicious", help="Print only flagged domains.")
    ] = False,
) -> None:
    """Score domains. Lower scores look more machine-generated."""
    model = MarkovModel.load(model_path)
    source = domains if domains else parse_lines(sys.stdin)
    for raw in source:
        label = normalize(raw)
        if not label:
            continue
        value = model.score(label)
        suspicious = model.is_suspicious(label)
        if only_suspicious and not suspicious:
            continue
        if as_json:
            typer.echo(json.dumps({"domain": raw, "label": label, "score": value, "suspicious": suspicious}))
        else:
            verdict = "SUSPICIOUS" if suspicious else "ok"
            typer.echo(f"{value:8.4f}  {verdict:<10}  {raw}")


@app.command("eval")
def eval_command(
    benign: Annotated[Path, typer.Argument(help="Benign domains not used in training.")],
    malicious: Annotated[Path, typer.Argument(help="Known DGA domains.")],
    model_path: ModelOption = DEFAULT_MODEL,
) -> None:
    """Report precision, recall and ROC AUC on labelled test data."""
    model = MarkovModel.load(model_path)
    if model.threshold is None:
        raise typer.BadParameter("model has no threshold", param_hint="--model")
    report = evaluate(
        benign=[model.score(label) for label in read_labels(benign)],
        malicious=[model.score(label) for label in read_labels(malicious)],
        threshold=model.threshold,
    )
    typer.echo(f"threshold            {report.threshold:.4f}")
    typer.echo(f"precision            {report.precision:.4f}")
    typer.echo(f"recall               {report.recall:.4f}")
    typer.echo(f"false positive rate  {report.false_positive_rate:.4f}")
    typer.echo(f"ROC AUC              {report.roc_auc:.4f}")


if __name__ == "__main__":
    app()
