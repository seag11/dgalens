# Builds the API image. The model is trained at build time from the Tranco list,
# so the image is self-contained:  docker build -t dgalens . && docker run -p 8000:8000 dgalens
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY README.md ./
COPY src ./src
COPY scripts ./scripts
RUN uv sync --frozen \
    && uv run python scripts/fetch_tranco.py --test 0 \
    && uv run dgalens train data/benign-train.csv --model models/dgalens.json \
    && rm -rf data

ENV DGALENS_MODEL=/app/models/dgalens.json
EXPOSE 8000
USER nobody
CMD ["/app/.venv/bin/uvicorn", "dgalens.api:app", "--host", "0.0.0.0", "--port", "8000"]
