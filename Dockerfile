FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev


FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/models \
    LLAMA_INDEX_CACHE_DIR=/models/llama_index \
    QDRANT_HOST=qdrant \
    OLLAMA_HOST=http://host.docker.internal:11434

RUN groupadd --system app \
    && useradd --system --gid app --home /app app \
    && mkdir -p /models \
    && chown -R app:app /models

WORKDIR /app

COPY --from=builder --chown=app:app /app /app

USER app

VOLUME ["/models"]

ENTRYPOINT ["raptor-qdrant"]
CMD ["--help"]
