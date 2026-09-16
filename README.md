# embedding

A small Rust HTTP service for generating vector embeddings.

## Quick start

```bash
docker compose up --build
```

First request triggers the model download into `./models` (bind-mounted into the container); subsequent restarts reuse it.

`GET /health` is liveness: it stays `200` while unused so idle RSS is not pinned. After a failed model load it returns `503` with the error so a broken cache is visible without calling `/embed`. Failed loads are retried at most once every 30 seconds.

```bash
curl -X POST http://localhost:3000/embed \
  -H 'content-type: application/json' \
  -d '{"texts":["hello world","another piece of text"]}'
```

## Configuration

Configured via environment variables (set them in `.env`):

| Variable | Default | Description |
| --- | --- | --- |
| `EMBEDDING_PORT` | `3000` | Port the service listens on. |
| `EMBEDDING_MODELS` | `nomic` | Comma-separated list of models allowed to load. ONNX sessions are created on first `/embed`, not at process start. Image/compose defaults are nomic only; `bge-small` and other aliases still work if you add them here. |
| `EMBEDDING_CACHE_DIR` | _(default cache)_ | Directory for downloaded model files. |
| `EMBEDDING_POOL_SIZE` | `1` | Number of ONNX sessions per model while it is loaded, then capped by available RAM. Concurrent `/embed` calls round-robin across sessions. Raise this for parallel HTTP throughput; each extra session keeps another copy of the weights resident until idle unload. |
| `EMBEDDING_INTRA_THREADS` | CPU count | ONNX Runtime intra-op threads per session. The default uses the whole machine on the single default session. When `EMBEDDING_POOL_SIZE` is greater than one, threads are split across sessions (`nproc / pool_size`, still capped by this value) so concurrent embeds do not oversubscribe the host. |
| `EMBEDDING_IDLE_UNLOAD_SECS` | `300` | Drop a model's sessions this many seconds after last use (`0` disables). The next `/embed` reloads the same checkpoint from `EMBEDDING_CACHE_DIR`. |

## API

### `POST /embed`

Request:

```json
{ "texts": ["string", "..."] }
```

Response:

```json
{
  "model": "NomicEmbedTextV15",
  "embeddings": [[0.012, -0.034, ...], ...],
  "tokens": 17
}
```

Errors:

- `400 Bad Request` — `texts` is empty or the model alias is not in `EMBEDDING_MODELS`.
- `500 Internal Server Error` — embedding or tokenizer failure (message in `error` field).
- `503 Service Unavailable` — `GET /health` after a model load has failed (message in `error` field).
