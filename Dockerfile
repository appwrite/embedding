# syntax=docker/dockerfile:1.7

FROM rust:1.95-slim-trixie AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libssl-dev \
        g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release --bin embedding --bin warmup && \
    cp target/release/embedding /usr/local/bin/embedding && \
    cp target/release/warmup /usr/local/bin/warmup

FROM debian:trixie-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3t64 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 embedder
RUN mkdir -p /home/embedder/models && chown embedder:embedder /home/embedder/models
ENV EMBEDDING_CACHE_DIR=/home/embedder/models
USER embedder
WORKDIR /home/embedder

COPY --from=builder /usr/local/bin/embedding /usr/local/bin/embedding
COPY --from=builder /usr/local/bin/warmup /usr/local/bin/warmup

# Download models at build time so the image ships with them cached. Override
# the model set with `--build-arg EMBEDDING_MODELS=...` (docker compose passes
# this from .env). Pool size is forced to 1 to keep the build's memory low —
# it only affects the warmup, not the runtime pool.
ARG EMBEDDING_MODELS=nomic,bge-small
RUN EMBEDDING_MODELS="${EMBEDDING_MODELS}" EMBEDDING_POOL_SIZE=1 /usr/local/bin/warmup

EXPOSE 3000

ENTRYPOINT ["/usr/local/bin/embedding"]
