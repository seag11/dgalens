# dgalens

Flag algorithmically generated domains (DGA domains) with a character-level Markov model.

Malware often finds its command-and-control server by generating hundreds of pseudo-random domains a day, such as `q8f7g2kd9s1m.net`, and trying each one. Human-chosen names follow the letter patterns of real words. Generated names usually don't. `dgalens` learns those patterns from popular legitimate domains and flags names that don't fit.

```console
$ dgalens score google.com paypal-login-secure.com q8f7g2kd9s1m.net kmbvkhwrcxqjlz.info
 -0.9671  ok          google.com
 -1.9605  ok          paypal-login-secure.com
 -8.8686  SUSPICIOUS  q8f7g2kd9s1m.net
 -6.2705  SUSPICIOUS  kmbvkhwrcxqjlz.info
```

`dgalens` ships as a CLI for training, scoring and evaluation, plus a small FastAPI service. The core model is about 150 lines of dependency-free Python.

## How it works

1. **Normalise.** `https://mail.example.co.uk/login` becomes `example`. DGAs randomise the registrable label, so that's the part that gets scored. The public suffix list is handled by `tldextract`, using its bundled offline snapshot.
2. **Model.** An order-3 character Markov chain estimates `P(next char | previous 3 chars)`, padded with start and end markers so it also learns how labels begin and end.
3. **Smoothing.** The model interpolates from the full 3-character context down through shorter contexts:

   `P(c | h) = (count(h, c) + α · P(c | shorter h)) / (count(h) + α)`

   An unseen context therefore backs off to what shorter contexts predict, instead of to a uniform guess. This mattered a lot in practice. With plain add-α smoothing, an order-3 model caught only about 5% of random labels at a 1% false-positive rate, because every unseen context looked equally plausible. With interpolation it catches 93% (see [Results](#results)).

4. **Score.** The score is the mean log-probability per character, so long and short labels are comparable. Lower means less word-like.
5. **Threshold.** `train` holds out 10% of the benign data and sets the threshold so that about 1% of it would be flagged (`--fpr`).
6. **Short labels.** Labels under 6 characters (`x7`, `k4g`) never get flagged. They carry too little signal, and legitimate short domains are common.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run python scripts/fetch_tranco.py      # benign data -> data/
uv run dgalens train data/benign-train.csv # model -> models/dgalens.json

uv run dgalens score example.com xjq9v2kp7z.com
cat domains.txt | uv run dgalens score --suspicious --json
```

### API

```sh
uv run uvicorn dgalens.api:app
curl -s localhost:8000/score -H 'content-type: application/json' -d '{"domain": "xjq9v2kp7z.com"}'
# {"domain":"xjq9v2kp7z.com","label":"xjq9v2kp7z","score":-5.49,"suspicious":true}
```

| Method | Path           | Body                                     |
| ------ | -------------- | ---------------------------------------- |
| `POST` | `/score`       | `{"domain": "..."}`                      |
| `POST` | `/score/batch` | `{"domains": ["...", ...]}` (up to 1000) |
| `GET`  | `/health`      |                                          |

Interactive docs are served at `/docs`.

### Docker

The image downloads the Tranco list and trains the model at build time:

```sh
docker build -t dgalens . && docker run -p 8000:8000 dgalens
```

## Results

Training data: the top 100,000 domains of the [Tranco list](https://tranco-list.eu), 90% for training and 10% for calibration. Target false-positive rate: 1%.

| Test set                                                   | Flagged                |
| ---------------------------------------------------------- | ---------------------- |
| Held-out legitimate domains (Tranco ranks 100,001–120,000) | 1.5% (false positives) |
| Random `[a-z0-9]` labels, 8–16 chars (synthetic)           | 93%                    |
| Random `[a-z]` labels, 8–16 chars (synthetic)              | ~55%                   |

The random labels are a sanity check, not a benchmark. Evaluating against realistic generated-domain datasets is the next milestone (see the roadmap).

Many of the "false positives" in the Tranco tail look machine-generated themselves (`itrnznre.top`, `afxiocmvsd.monster`). A popularity list isn't a clean list of legitimate sites.

## Limitations

- **Dictionary DGAs aren't caught.** Some generators build names by joining real words (`mantleshake.net`). They look natural character by character, so a character-level model can't separate them.
- **Letters-only random labels are harder than alphanumeric ones.** Digits mixed with letters are a strong signal. Pronounceable random strings often pass.
- **The model is biased toward English.** Legitimate domains in other languages or transliterations score lower. Training on a regionally appropriate list helps.
- **Each domain is scored alone.** Real DGA detection also uses NXDOMAIN rates, query timing and many-domains-per-host patterns. `dgalens` is one signal, not a verdict.

## Roadmap

- [ ] Evaluate against labelled DGA-style datasets (random, pronounceable and dictionary-based). Publish per-style recall and a ROC curve.
- [ ] Compare order and smoothing settings on that data.
- [ ] Add a `generate` command that samples plausible-looking labels from the chain, e.g. for typosquat and lookalike testing.

## Development

```sh
uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run mypy src
```

## License

MIT
