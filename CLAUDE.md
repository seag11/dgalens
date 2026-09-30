# dgalens

Character-level Markov-chain DGA domain detector.

## Decisions already made

- **The Markov model is hand-written, with no Markov library.** It's the core of the portfolio value, and libraries like markovify don't do smoothed character-level scoring.
- **Never commit data or models.** Download them with scripts (`data/` and `models/` are gitignored). Check each source's license before adding it.
- **Keep the project neutral.** Don't name specific malware families or tools in code, docs or comments; describe generator _styles_ instead (random alphanumeric, pronounceable, dictionary-based).

## Layout

- `src/dgalens/model.py`: `MarkovModel` (interpolated smoothing, calibration, JSON save/load). No third-party imports.
- `src/dgalens/domains.py`: `normalize()` turns a URL or FQDN into its registrable label (tldextract, offline snapshot).
- `src/dgalens/data.py`: reads domain lists (one per line, `#` comments, `rank,domain` CSV).
- `src/dgalens/metrics.py`: precision, recall, FPR and rank-based ROC AUC, dependency-free.
- `src/dgalens/cli.py`: Typer app with `train`, `score` and `eval`.
- `src/dgalens/api.py`: FastAPI `create_app(model)`. Tests inject a model; otherwise it's loaded from `$DGALENS_MODEL`.
- `scripts/fetch_tranco.py`: benign data (Tranco top-1M, split into train and test by rank).

## Findings so far

- Plain add-α smoothing failed at order ≥ 3: unseen contexts back off to uniform, so random labels weren't penalised (about 5% recall at 1% FPR). Interpolated smoothing fixed it (93% recall on random alphanumeric labels, about 55% on letters-only). Defaults: order=3, alpha=1.0, min_length=6.
- Labels shorter than 6 characters dominated benign false positives, so they're never flagged.
- The Tranco tail contains real DGA-looking domains, so benign FPR is slightly pessimistic.

## Next steps

1. Get labelled DGA-style data for `dgalens eval`. Prefer synthetic generators written for this repo (random alphanumeric, pronounceable, dictionary-based styles) over reimplementing real malware algorithms. A public feed is an option only if its license permits it.
2. Report per-style recall and a ROC curve in the README (scikit-learn and matplotlib could go in an optional `eval` dependency group).
3. Optional `generate` command.

## Commands

```sh
uv sync
uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run mypy src
uv run python scripts/fetch_tranco.py && uv run dgalens train data/benign-train.csv
```
