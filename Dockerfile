# syntax=docker/dockerfile:1.7

FROM rust:1.95-slim-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libssl-dev \
        g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release --bin embedding && \
    cp target/release/embedding /usr/local/bin/embedding

FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 embedder
RUN mkdir -p /home/embedder/models && chown embedder:embedder /home/embedder/models
ENV EMBEDDING_CACHE_DIR=/home/embedder/models
USER embedder
WORKDIR /home/embedder

COPY --from=builder /usr/local/bin/embedding /usr/local/bin/embedding

EXPOSE 3000

ENTRYPOINT ["/usr/local/bin/embedding"]
