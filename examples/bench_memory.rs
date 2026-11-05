//! Compare the memory of two embedding server binaries under the same load.
//!
//! Starts each binary, sends batches of long texts of varying length to
//! /embed, samples RSS, and prints a markdown table of peak and retained RSS.
//!
//! cargo run --release --example bench_memory -- \
//!     --base bin/base --head bin/head --cache-dir models --out bench.md

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

const WORDS: &[&str] = &[
    "function", "return", "class", "public", "private", "static", "array", "string", "value",
    "index", "query", "request", "response", "database",
];

struct Args {
    base: String,
    head: String,
    cache_dir: String,
    model: String,
    pool_size: usize,
    concurrency: Vec<usize>,
    reps: usize,
    batches: usize,
    batch_size: usize,
    settle_secs: u64,
    seed: u64,
    out: Option<String>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = Args {
            base: String::new(),
            head: String::new(),
            cache_dir: String::new(),
            model: "nomic".into(),
            pool_size: 1,
            concurrency: vec![1, 4],
            reps: 2,
            batches: 12,
            batch_size: 16,
            settle_secs: 3,
            seed: 42,
            out: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let value = it.next().ok_or(format!("{flag} needs a value"))?;
            let num = |v: &str| v.parse::<u64>().map_err(|e| format!("{flag}: {e}"));
            match flag.as_str() {
                "--base" => args.base = value,
                "--head" => args.head = value,
                "--cache-dir" => args.cache_dir = value,
                "--model" => args.model = value,
                "--pool-size" => args.pool_size = num(&value)? as usize,
                "--concurrency" => {
                    args.concurrency = value
                        .split(',')
                        .map(|c| num(c.trim()).map(|n| n.max(1) as usize))
                        .collect::<Result<_, _>>()?
                }
                "--reps" => args.reps = num(&value)?.max(1) as usize,
                "--batches" => args.batches = num(&value)? as usize,
                "--batch-size" => args.batch_size = num(&value)? as usize,
                "--settle" => args.settle_secs = num(&value)?,
                "--seed" => args.seed = num(&value)?,
                "--out" => args.out = Some(value),
                _ => return Err(format!("unknown flag {flag}")),
            }
        }
        if args.base.is_empty() || args.head.is_empty() || args.cache_dir.is_empty() {
            return Err("--base, --head and --cache-dir are required".into());
        }
        Ok(args)
    }
}

struct Sample {
    peak: u64,
    retained: u64,
    elapsed: Duration,
}

/// xorshift64 so the workload is identical for base and head without a rand dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next() % (hi - lo + 1) as u64) as usize
    }
}

// Varying lengths so ONNX Runtime sees many input shapes.
fn workload(args: &Args) -> Vec<Vec<String>> {
    let mut rng = Rng(args.seed.max(1));
    (0..args.batches)
        .map(|_| {
            (0..args.batch_size)
                .map(|_| {
                    let len = rng.range(100, 1800);
                    (0..len)
                        .map(|_| WORDS[rng.range(0, WORDS.len() - 1)])
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .collect()
        })
        .collect()
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .expect("bind a free port")
}

fn rss(sys: &mut System, pid: Pid) -> u64 {
    sys.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_memory(),
    );
    sys.process(pid).map(|p| p.memory()).unwrap_or(0)
}

struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn run(binary: &str, args: &Args, batches: &[Vec<String>], concurrency: usize) -> Sample {
    let port = free_port();
    let mut server = Server(
        Command::new(binary)
            .env("EMBEDDING_PORT", port.to_string())
            .env("EMBEDDING_MODELS", &args.model)
            .env("EMBEDDING_POOL_SIZE", args.pool_size.to_string())
            .env("EMBEDDING_IDLE_UNLOAD_SECS", "0")
            .env("EMBEDDING_CACHE_DIR", &args.cache_dir)
            .env("RUST_LOG", "warn")
            .stdout(Stdio::null())
            .spawn()
            .unwrap_or_else(|e| panic!("start {binary}: {e}")),
    );
    let pid = Pid::from_u32(server.0.id());
    let url = format!("http://127.0.0.1:{port}");

    let deadline = Instant::now() + Duration::from_secs(30);
    while ureq::get(format!("{url}/health")).call().is_err() {
        if let Ok(Some(status)) = server.0.try_wait() {
            panic!("{binary} exited with {status}");
        }
        assert!(Instant::now() < deadline, "{binary} never became healthy");
        thread::sleep(Duration::from_millis(100));
    }

    let peak = AtomicU64::new(0);
    let done = AtomicBool::new(false);
    let errors = Mutex::new(Vec::new());
    let mut elapsed = Duration::ZERO;

    thread::scope(|s| {
        s.spawn(|| {
            let mut sys = System::new();
            while !done.load(Ordering::Relaxed) {
                peak.fetch_max(rss(&mut sys, pid), Ordering::Relaxed);
                thread::sleep(Duration::from_millis(100));
            }
        });

        let started = Instant::now();
        let (url, errors) = (&url, &errors);
        for chunk in batches.chunks(concurrency) {
            thread::scope(|inner| {
                for texts in chunk {
                    inner.spawn(move || {
                        let body = serde_json::json!({ "model": args.model, "texts": texts });
                        let result = ureq::post(format!("{url}/embed"))
                            .header("content-type", "application/json")
                            .send(body.to_string())
                            .and_then(|mut r| r.body_mut().read_to_string());
                        if let Err(e) = result {
                            errors.lock().unwrap().push(e.to_string());
                        }
                    });
                }
            });
        }
        elapsed = started.elapsed();
        done.store(true, Ordering::Relaxed);
    });

    if let Some(e) = errors.lock().unwrap().first() {
        panic!("{binary}: /embed failed: {e}");
    }

    thread::sleep(Duration::from_secs(args.settle_secs));
    let retained = rss(&mut System::new(), pid);
    Sample {
        peak: peak.into_inner().max(retained),
        retained,
        elapsed,
    }
}

fn median<T: Copy + PartialOrd>(mut values: Vec<T>) -> T {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

fn mb(bytes: u64) -> String {
    format!("{} MB", bytes / (1024 * 1024))
}

fn delta(base: u64, head: u64) -> String {
    if base == 0 {
        return "n/a".into();
    }
    format!("{:+.0}%", (head as f64 - base as f64) / base as f64 * 100.0)
}

fn main() {
    let args = Args::parse().unwrap_or_else(|e| {
        eprintln!("bench_memory: {e}");
        std::process::exit(2);
    });
    let batches = workload(&args);

    let mut rows = Vec::new();
    for &concurrency in &args.concurrency {
        let mut medians = Vec::new();
        for (label, binary) in [("base", &args.base), ("head", &args.head)] {
            let samples: Vec<Sample> = (1..=args.reps)
                .map(|rep| {
                    let s = run(binary, &args, &batches, concurrency);
                    eprintln!(
                        "{label} concurrency={concurrency} rep={rep}: peak={} retained={} time={:.1}s",
                        mb(s.peak),
                        mb(s.retained),
                        s.elapsed.as_secs_f64()
                    );
                    s
                })
                .collect();
            medians.push((
                median(samples.iter().map(|s| s.peak).collect()),
                median(samples.iter().map(|s| s.retained).collect()),
                median(samples.iter().map(|s| s.elapsed.as_secs_f64()).collect()),
            ));
        }
        let (b, h) = (medians[0], medians[1]);
        rows.push(format!(
            "| {concurrency} | {} | {} | {} | {} | {} | {} | {:.1}s | {:.1}s |",
            mb(b.0),
            mb(h.0),
            delta(b.0, h.0),
            mb(b.1),
            mb(h.1),
            delta(b.1, h.1),
            b.2,
            h.2
        ));
    }

    let table = format!(
        "Model `{}`, `EMBEDDING_POOL_SIZE={}`, {} requests of {} texts (100 to 1,800 words), \
         median of {} runs. Retained is RSS {}s after the last request.\n\n\
         | Concurrent requests | Peak (base) | Peak (head) | Δ | Retained (base) \
         | Retained (head) | Δ | Time (base) | Time (head) |\n\
         |---|---|---|---|---|---|---|---|---|\n{}\n",
        args.model,
        args.pool_size,
        args.batches,
        args.batch_size,
        args.reps,
        args.settle_secs,
        rows.join("\n")
    );
    print!("{table}");
    if let Some(out) = &args.out {
        std::fs::write(out, &table).unwrap_or_else(|e| panic!("write {out}: {e}"));
    }
}
