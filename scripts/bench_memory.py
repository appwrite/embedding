#!/usr/bin/env python3
"""Compare the memory of two embedding server binaries under the same load.

Starts each binary, sends batches of long texts of varying length to /embed,
samples RSS, and prints a markdown table of peak and retained RSS.

  python3 scripts/bench_memory.py --base target/base --head target/head \
      --cache-dir models --out bench.md
"""

import argparse
import json
import os
import random
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

WORDS = (
    "function return class public private static array string value index "
    "query request response database"
).split()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def rss_bytes(pid):
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True
    ).stdout.strip()
    return int(out) * 1024 if out else 0


def workload(seed, batches, batch_size):
    # Varying lengths so ONNX Runtime sees many input shapes.
    rng = random.Random(seed)
    return [
        [" ".join(rng.choices(WORDS, k=rng.randint(100, 1800))) for _ in range(batch_size)]
        for _ in range(batches)
    ]


def run(binary, args, concurrency):
    port = free_port()
    env = dict(
        os.environ,
        EMBEDDING_PORT=str(port),
        EMBEDDING_MODELS=args.model,
        EMBEDDING_POOL_SIZE=str(args.pool_size),
        EMBEDDING_IDLE_UNLOAD_SECS="0",
        EMBEDDING_CACHE_DIR=args.cache_dir,
        RUST_LOG="warn",
    )
    proc = subprocess.Popen([binary], env=env, stdout=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    peak = 0
    stop = threading.Event()

    def sample():
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, rss_bytes(proc.pid))
            time.sleep(0.1)

    def post(texts):
        body = json.dumps({"model": args.model, "texts": texts}).encode()
        req = urllib.request.Request(
            url + "/embed", body, {"content-type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=900) as resp:
            json.load(resp)

    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(url + "/health", timeout=1).close()
                break
            except Exception:
                if proc.poll() is not None:
                    sys.exit(f"{binary} exited with {proc.returncode}")
                time.sleep(0.1)
        else:
            sys.exit(f"{binary} never became healthy")

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        errors = []

        def guarded(texts):
            try:
                post(texts)
            except Exception as e:
                errors.append(e)

        batches = workload(args.seed, args.batches, args.batch_size)
        started = time.monotonic()
        for i in range(0, len(batches), concurrency):
            threads = [
                threading.Thread(target=guarded, args=(b,))
                for b in batches[i : i + concurrency]
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        elapsed = time.monotonic() - started
        if errors:
            sys.exit(f"{binary}: /embed failed: {errors[0]}")

        time.sleep(args.settle)
        retained = rss_bytes(proc.pid)
        stop.set()
        sampler.join()
        return peak, retained, elapsed
    finally:
        stop.set()
        proc.terminate()
        proc.wait()


def mb(value):
    return f"{value / 2**20:,.0f} MB"


def delta(base, head):
    if base == 0:
        return "n/a"
    return f"{(head - base) / base * 100:+.0f}%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="baseline server binary")
    p.add_argument("--head", required=True, help="candidate server binary")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--model", default="nomic")
    p.add_argument("--pool-size", type=int, default=1)
    p.add_argument("--concurrency", default="1,4", help="comma-separated scenarios")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--batches", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--settle", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", help="also write the markdown table here")
    args = p.parse_args()

    rows = []
    for conc in [int(c) for c in args.concurrency.split(",")]:
        results = {}
        for label, binary in (("base", args.base), ("head", args.head)):
            reps = []
            for rep in range(args.reps):
                peak, retained, elapsed = run(binary, args, conc)
                print(
                    f"{label} conc={conc} rep={rep + 1}: peak={mb(peak)} "
                    f"retained={mb(retained)} time={elapsed:.1f}s",
                    file=sys.stderr,
                )
                reps.append((peak, retained, elapsed))
            results[label] = [statistics.median(col) for col in zip(*reps)]
        b, h = results["base"], results["head"]
        rows.append(
            f"| {conc} | {mb(b[0])} | {mb(h[0])} | {delta(b[0], h[0])} "
            f"| {mb(b[1])} | {mb(h[1])} | {delta(b[1], h[1])} "
            f"| {b[2]:.1f}s | {h[2]:.1f}s |"
        )

    table = "\n".join(
        [
            f"Model `{args.model}`, `EMBEDDING_POOL_SIZE={args.pool_size}`, "
            f"{args.batches} requests of {args.batch_size} texts (100 to 1,800 words), "
            f"median of {args.reps} runs. Retained is RSS {args.settle:g}s after the last request.",
            "",
            "| Concurrent requests | Peak (base) | Peak (head) | Δ | Retained (base) "
            "| Retained (head) | Δ | Time (base) | Time (head) |",
            "|---|---|---|---|---|---|---|---|---|",
            *rows,
        ]
    )
    print(table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
