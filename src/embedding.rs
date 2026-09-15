use crate::error::EmbedError;
use crate::model::{self, EmbeddingModel};
use fastembed::{ExecutionProviderDispatch, InitOptions, TextEmbedding, get_cache_dir};
use hf_hub::api::sync::ApiBuilder;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};
use tokenizers::Tokenizer;

/// Unload a model this many seconds after last use. `0` disables unloading.
pub const DEFAULT_IDLE_UNLOAD_SECS: u64 = 300;

#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    pub model: String,
    pub embeddings: Vec<Vec<f32>>,
    pub tokens: usize,
    pub total_duration: u64,
}

#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub models: Vec<EmbeddingModel>,
    pub show_download_progress: bool,
    pub cache_dir: Option<String>,
    pub pool_size: usize,
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub sub_batch_size: usize,
    pub intra_threads: usize,
    pub idle_unload_secs: u64,
}

fn available_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
}

fn default_pool_size() -> usize {
    available_cpus()
}

fn default_intra_threads() -> usize {
    available_cpus().min(4)
}

fn memory_budget(host_available: u64, cgroup_free: Option<u64>) -> u64 {
    match cgroup_free {
        Some(cgroup_free) => host_available.min(cgroup_free),
        None => host_available,
    }
}

fn next_index(counter: &AtomicUsize, len: usize) -> usize {
    counter.fetch_add(1, Ordering::Relaxed) % len
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Whether an idle model slot should drop its ONNX sessions.
pub(crate) fn should_unload(last_access: u64, now: u64, idle_unload_secs: u64) -> bool {
    idle_unload_secs > 0 && last_access > 0 && now.saturating_sub(last_access) >= idle_unload_secs
}

fn parse_usize_env(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|size| size.parse::<usize>().ok())
}

impl EmbeddingConfig {
    pub fn from_env() -> Self {
        let models: Vec<EmbeddingModel> = std::env::var("EMBEDDING_MODELS")
            .ok()
            .map(|model| {
                model
                    .split(",")
                    .map(|model| {
                        let name = model.trim();
                        model::from_name(name)
                            .unwrap_or_else(|| panic!("{} model not available", name))
                    })
                    .collect()
            })
            .unwrap_or_else(|| vec![EmbeddingModel::NomicEmbedTextV15]);

        let cache_dir = std::env::var("EMBEDDING_CACHE_DIR").ok();

        let pool_size = parse_usize_env("EMBEDDING_POOL_SIZE")
            .filter(|&n| n >= 1)
            .unwrap_or_else(default_pool_size);

        let intra_threads = parse_usize_env("EMBEDDING_INTRA_THREADS")
            .filter(|&n| n >= 1)
            .unwrap_or_else(default_intra_threads);

        let idle_unload_secs = std::env::var("EMBEDDING_IDLE_UNLOAD_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(DEFAULT_IDLE_UNLOAD_SECS);

        Self {
            models,
            show_download_progress: true,
            cache_dir,
            pool_size,
            execution_providers: Vec::new(),
            sub_batch_size: 0,
            intra_threads,
            idle_unload_secs,
        }
    }
}

pub struct EmbeddingClient {
    models: HashMap<String, LoadedModel>,
    config: EmbeddingConfig,
    sub_batch_override: usize,
    gpu: bool,
}

struct LoadedModel {
    spec: EmbeddingModel,
    model_name: String,
    next: AtomicUsize,
    load: Mutex<()>,
    inner: Mutex<ModelSlot>,
    dimension: usize,
    last_access: AtomicU64,
    in_flight: AtomicUsize,
}

struct ModelSlot {
    pool: Option<Vec<Arc<Mutex<TextEmbedding>>>>,
    tokenizer: Option<Arc<Tokenizer>>,
}

struct InFlightGuard<'a> {
    counter: &'a AtomicUsize,
}

impl Drop for InFlightGuard<'_> {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::SeqCst);
    }
}

struct BuiltModel {
    pool: Vec<Arc<Mutex<TextEmbedding>>>,
    tokenizer: Arc<Tokenizer>,
}

impl EmbeddingClient {
    /// Register configured models without loading ONNX weights. Sessions are
    /// created on the first `/embed` (or [`Self::preload`]).
    pub fn new(config: EmbeddingConfig) -> Result<Self, String> {
        let mut models = HashMap::new();
        for model in &config.models {
            let model_name = format!("{:?}", model);
            models.insert(
                model_name.clone(),
                LoadedModel {
                    spec: model.clone(),
                    model_name,
                    next: AtomicUsize::new(0),
                    load: Mutex::new(()),
                    inner: Mutex::new(ModelSlot {
                        pool: None,
                        tokenizer: None,
                    }),
                    dimension: model::dimension(model),
                    last_access: AtomicU64::new(0),
                    in_flight: AtomicUsize::new(0),
                },
            );
        }

        let sub_batch_override = config.sub_batch_size;
        let gpu = !config.execution_providers.is_empty();
        Ok(Self {
            models,
            config,
            sub_batch_override,
            gpu,
        })
    }

    /// Load every configured model now. Used by the warmup binary so image
    /// builds still populate the ONNX cache.
    pub fn preload(&self) -> Result<(), String> {
        for loaded in self.models.values() {
            self.ensure_loaded(loaded)?;
            loaded.last_access.store(unix_now(), Ordering::Relaxed);
        }
        Ok(())
    }

    /// Drop ONNX sessions that have been unused for `idle_unload_secs`.
    pub fn unload_idle(&self) {
        if self.config.idle_unload_secs == 0 {
            return;
        }
        let now = unix_now();
        for loaded in self.models.values() {
            let last = loaded.last_access.load(Ordering::Relaxed);
            if !should_unload(last, now, self.config.idle_unload_secs) {
                continue;
            }
            let mut slot = match loaded.inner.lock() {
                Ok(slot) => slot,
                Err(poisoned) => poisoned.into_inner(),
            };
            // Recheck under the pool lock so we never drop sessions that an
            // in-flight embed already acquired (which would reload a second pool).
            if loaded.in_flight.load(Ordering::SeqCst) > 0 {
                continue;
            }
            if slot.pool.is_some() || slot.tokenizer.is_some() {
                slot.pool = None;
                slot.tokenizer = None;
                tracing::info!(
                    model = loaded.model_name.as_str(),
                    idle_secs = now.saturating_sub(last),
                    "unloaded idle embedding model"
                );
            }
        }
    }

    pub fn idle_unload_secs(&self) -> u64 {
        self.config.idle_unload_secs
    }

    fn ensure_loaded(&self, loaded: &LoadedModel) -> Result<(), String> {
        {
            let slot = loaded
                .inner
                .lock()
                .map_err(|e| format!("Embedding model lock poisoned: {}", e))?;
            if slot.pool.is_some() && slot.tokenizer.is_some() {
                return Ok(());
            }
        }

        let _load = loaded
            .load
            .lock()
            .map_err(|e| format!("Embedding model lock poisoned: {}", e))?;
        {
            let slot = loaded
                .inner
                .lock()
                .map_err(|e| format!("Embedding model lock poisoned: {}", e))?;
            if slot.pool.is_some() && slot.tokenizer.is_some() {
                return Ok(());
            }
        }

        let built = Self::load_model(&loaded.spec, &loaded.model_name, &self.config)?;

        let mut slot = loaded
            .inner
            .lock()
            .map_err(|e| format!("Embedding model lock poisoned: {}", e))?;
        slot.pool = Some(built.pool);
        slot.tokenizer = Some(built.tokenizer);
        Ok(())
    }

    fn load_model(
        model: &EmbeddingModel,
        model_name: &str,
        config: &EmbeddingConfig,
    ) -> Result<BuiltModel, String> {
        let desired_pool_size = config.pool_size.max(1);
        let dimension = model::dimension(model);
        let has_gpu_providers = !config.execution_providers.is_empty();

        let mem_before_loading_model = if desired_pool_size > 1 {
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            Some(memory_budget(
                sys.available_memory(),
                sys.cgroup_limits().map(|limits| limits.free_memory),
            ))
        } else {
            None
        };

        let mut first_model = Self::init_model(model, config)?;

        // Tokenizer is fetched from the same cache dir fastembed just populated,
        // so this is a cache hit (no network) after the first model load.
        let tokenizer = Arc::new(Self::load_tokenizer(model, config)?);

        // Always run one inference so Docker warmup (pool_size=1) still proves
        // the session can execute, and so extra pool slots are sized from a
        // post-arena RSS delta when desired_pool_size > 1.
        first_model
            .embed(vec!["warmup"], None)
            .map_err(|e| format!("warmup inference failed for {}: {}", model_name, e))?;

        let pool_size = if let Some(mem_before_loading_model) = mem_before_loading_model {
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            let memory_after_loading_model = memory_budget(
                sys.available_memory(),
                sys.cgroup_limits().map(|limits| limits.free_memory),
            );
            let per_instance_loaded =
                mem_before_loading_model.saturating_sub(memory_after_loading_model);

            // ONNX Runtime uses arena allocation that grows with
            // batch_size × sequence_length² (attention matrices) and is never
            // released.  The warmup above only allocates a minimal arena for a
            // single short text.  Apply a 3× multiplier to account for realistic
            // inference workloads (batch=8-32 texts of 1000-2000 tokens each).
            let per_instance_bytes = per_instance_loaded.saturating_mul(3);

            let nproc = available_cpus();
            let budget = mem_before_loading_model * 6 / 10;
            if let Some(max_memory) = budget.checked_div(per_instance_bytes) {
                if max_memory == 0 {
                    tracing::warn!(
                        estimated_with_arena_mb = per_instance_bytes / (1024 * 1024),
                        budget_mb = budget / (1024 * 1024),
                        "A single {} instance is estimated to exceed the memory budget; \
                         running with pool_size=1 but the process may be OOM-killed under load. \
                         Raise the container memory limit or pick a smaller model.",
                        model_name
                    );
                }
                let max_memory = (max_memory as usize).max(1);
                let capped = max_memory.min(desired_pool_size);
                tracing::info!(
                    per_instance_mb = per_instance_loaded / (1024 * 1024),
                    estimated_with_arena_mb = per_instance_bytes / (1024 * 1024),
                    available_mb = mem_before_loading_model / (1024 * 1024),
                    budget_mb = budget / (1024 * 1024),
                    nproc = nproc,
                    desired = desired_pool_size,
                    max_from_memory = max_memory,
                    capped = capped,
                    "Measured ONNX model memory footprint"
                );
                capped
            } else {
                desired_pool_size
            }
        } else {
            tracing::info!(
                model = model_name,
                "Using pool_size=1; extra sessions are not created"
            );
            1
        };

        let mut extra_config = config.clone();
        extra_config.intra_threads = (available_cpus() / pool_size)
            .max(1)
            .min(config.intra_threads);

        let mut pool = Vec::with_capacity(pool_size);
        pool.push(Arc::new(Mutex::new(first_model)));

        for _ in 1..pool_size {
            let inst = Self::init_model(model, &extra_config)?;
            pool.push(Arc::new(Mutex::new(inst)));
        }

        let ep_label = if has_gpu_providers {
            "GPU (CUDA)"
        } else {
            "CPU"
        };
        tracing::info!(
            "Initialized embedding model: {} ({}d, pool_size={}, intra_threads={}, execution_provider={})",
            model_name,
            dimension,
            pool_size,
            config.intra_threads,
            ep_label,
        );

        Ok(BuiltModel { pool, tokenizer })
    }

    fn load_tokenizer(
        model: &EmbeddingModel,
        config: &EmbeddingConfig,
    ) -> Result<Tokenizer, String> {
        let info =
            TextEmbedding::get_model_info(model).map_err(|e| format!("get_model_info: {}", e))?;
        let cache_dir: PathBuf = config
            .cache_dir
            .clone()
            .map(PathBuf::from)
            .unwrap_or_else(|| get_cache_dir().into());
        let api = ApiBuilder::new()
            .with_cache_dir(cache_dir)
            .with_progress(false)
            .build()
            .map_err(|e| format!("hf-hub init: {}", e))?;
        let repo = api.model(info.model_code.clone());
        let path = repo
            .get("tokenizer.json")
            .map_err(|e| format!("fetch tokenizer.json: {}", e))?;
        Tokenizer::from_file(&path).map_err(|e| format!("parse tokenizer.json: {}", e))
    }

    fn init_model(
        model: &EmbeddingModel,
        config: &EmbeddingConfig,
    ) -> Result<TextEmbedding, String> {
        let mut init_options = InitOptions::new(model.clone())
            .with_show_download_progress(config.show_download_progress)
            .with_intra_threads(config.intra_threads);

        if let Some(cache_dir) = &config.cache_dir {
            init_options = init_options.with_cache_dir(cache_dir.into());
        }

        if !config.execution_providers.is_empty() {
            init_options =
                init_options.with_execution_providers(config.execution_providers.clone());
        }

        TextEmbedding::try_new(init_options)
            .map_err(|e| format!("Failed to initialize embedding model: {}", e))
    }

    fn acquire_instance(&self, loaded: &LoadedModel) -> Result<Arc<Mutex<TextEmbedding>>, String> {
        for _ in 0..2 {
            self.ensure_loaded(loaded)?;
            let slot = loaded
                .inner
                .lock()
                .map_err(|e| format!("Embedding model lock poisoned: {}", e))?;
            if let Some(pool) = slot.pool.as_ref() {
                let idx = next_index(&loaded.next, pool.len());
                return Ok(pool[idx].clone());
            }
        }
        Err(format!(
            "embedding model {} unloaded during acquire",
            loaded.model_name
        ))
    }

    fn tokenizer(&self, loaded: &LoadedModel) -> Result<Arc<Tokenizer>, String> {
        self.ensure_loaded(loaded)?;
        let slot = loaded
            .inner
            .lock()
            .map_err(|e| format!("Embedding model lock poisoned: {}", e))?;
        slot.tokenizer
            .clone()
            .ok_or_else(|| format!("tokenizer missing for {}", loaded.model_name))
    }

    pub async fn embed(
        &self,
        model_name: &str,
        texts: &[&str],
    ) -> Result<EmbeddingResult, EmbedError> {
        let started = std::time::Instant::now();
        let resolved = model::from_name(model_name).ok_or_else(|| {
            EmbedError::UnknownModel(format!("unknown model alias: {}", model_name))
        })?;
        let canonical = format!("{:?}", resolved);
        let loaded = self.models.get(&canonical).ok_or_else(|| {
            EmbedError::UnknownModel(format!("model not allowed: {}", model_name))
        })?;

        loaded.last_access.store(unix_now(), Ordering::Relaxed);
        loaded.in_flight.fetch_add(1, Ordering::SeqCst);
        let _in_flight = InFlightGuard {
            counter: &loaded.in_flight,
        };

        let sub_batch = if self.sub_batch_override > 0 {
            self.sub_batch_override
        } else {
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            let available_mb = memory_budget(
                sys.available_memory(),
                sys.cgroup_limits().map(|limits| limits.free_memory),
            ) / (1024 * 1024);
            Self::compute_sub_batch(available_mb, loaded.dimension, self.gpu)
        };

        let mut handles = Vec::new();
        for chunk in texts.chunks(sub_batch) {
            let inst = self.acquire_instance(loaded)?;
            let chunked_texts: Vec<String> = chunk.iter().map(|t| (*t).to_owned()).collect();

            handles.push(tokio::task::spawn_blocking(move || {
                let mut m = inst
                    .lock()
                    .map_err(|e| format!("Embedding model lock poisoned: {}", e))?;
                m.embed(chunked_texts, None)
                    .map_err(|e| format!("Failed to generate embeddings: {}", e))
            }));
        }

        let mut embeddings = Vec::with_capacity(texts.len());
        for handle in handles {
            let mut batch_result = handle
                .await
                .map_err(|e| format!("Failed to join embedding task: {}", e))??;
            embeddings.append(&mut batch_result);
        }

        let tokenizer = self.tokenizer(loaded)?;
        let owned_texts: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
        let tokens = tokio::task::spawn_blocking(move || -> Result<usize, String> {
            let encodings = tokenizer
                .encode_batch(owned_texts, true)
                .map_err(|e| format!("tokenize: {}", e))?;
            Ok(encodings.iter().map(|e| e.get_ids().len()).sum())
        })
        .await
        .map_err(|e| format!("Failed to join tokenizer task: {}", e))??;

        loaded.last_access.store(unix_now(), Ordering::Relaxed);

        Ok(EmbeddingResult {
            model: loaded.model_name.clone(),
            embeddings,
            tokens,
            total_duration: started.elapsed().as_nanos() as u64,
        })
    }

    /// Compute sub-batch size based on available system memory.
    ///
    /// Uses 50% of available RAM as a budget.
    /// When GPU is enabled, the upper clamp is raised to 256 (GPU VRAM can
    /// handle much larger batches than CPU).
    /// TODO: add here the config batch size
    fn compute_sub_batch(available_mb: u64, dimension: usize, gpu: bool) -> usize {
        // Per-text memory estimate for ONNX inference.  Attention matrices
        // dominate: heads × seq² × 4 bytes.  For 768-dim BERT-like models
        // (12 heads) processing ~1000-2000 token code chunks, attention alone
        // is 50-200 MB per text.  The estimate below is conservative so the
        // sub-batch stays small enough to prevent arena over-allocation.
        let mb_per_text: u64 = if dimension >= 768 { 100 } else { 40 };
        let budget_mb = available_mb / 2;
        let max_batch = if gpu { 256 } else { 16 };
        (budget_mb / mb_per_text).clamp(1, max_batch) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    const ENV_KEYS: [&str; 5] = [
        "EMBEDDING_MODELS",
        "EMBEDDING_CACHE_DIR",
        "EMBEDDING_POOL_SIZE",
        "EMBEDDING_INTRA_THREADS",
        "EMBEDDING_IDLE_UNLOAD_SECS",
    ];

    /// Holds the env mutex and restores the original values on drop. Tests that
    /// touch `std::env` must hold one of these — env state is process-global, so
    /// parallel cargo-test threads would otherwise stomp each other.
    struct EnvGuard {
        saved: Vec<(&'static str, Option<String>)>,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (k, v) in &self.saved {
                unsafe {
                    match v {
                        Some(val) => std::env::set_var(k, val),
                        None => std::env::remove_var(k),
                    }
                }
            }
        }
    }

    fn isolate_env() -> EnvGuard {
        let lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved: Vec<(&'static str, Option<String>)> = ENV_KEYS
            .iter()
            .map(|k| (*k, std::env::var(k).ok()))
            .collect();
        for k in ENV_KEYS {
            unsafe {
                std::env::remove_var(k);
            }
        }
        EnvGuard { saved, _lock: lock }
    }

    fn set(k: &str, v: &str) {
        unsafe {
            std::env::set_var(k, v);
        }
    }

    #[test]
    fn memory_budget_is_capped_by_cgroup_limit() {
        const GIB: u64 = 1024 * 1024 * 1024;
        assert_eq!(memory_budget(32 * GIB, None), 32 * GIB);
        assert_eq!(memory_budget(32 * GIB, Some(GIB)), GIB);
        assert_eq!(memory_budget(2 * GIB, Some(32 * GIB)), 2 * GIB);
    }

    #[test]
    fn from_env_uses_nomic_when_unset() {
        let _g = isolate_env();
        let cfg = EmbeddingConfig::from_env();
        assert!(matches!(cfg.models[0], EmbeddingModel::NomicEmbedTextV15));
        assert_eq!(cfg.cache_dir, None);
        assert_eq!(cfg.pool_size, available_cpus());
        assert!(cfg.show_download_progress);
        assert!(cfg.execution_providers.is_empty());
        assert_eq!(cfg.sub_batch_size, 0);
        assert!(cfg.intra_threads >= 1);
        assert!(cfg.intra_threads <= 4);
        assert_eq!(cfg.idle_unload_secs, DEFAULT_IDLE_UNLOAD_SECS);
    }

    #[test]
    fn from_env_parses_gemma_alias() {
        let _g = isolate_env();
        set("EMBEDDING_MODELS", "embedding-gemma");
        let cfg = EmbeddingConfig::from_env();
        assert!(matches!(cfg.models[0], EmbeddingModel::EmbeddingGemma300M));
    }

    #[test]
    fn from_env_parses_nomic_aliases() {
        for alias in ["nomic-embed-text", "nomic", "NOMIC"] {
            let _g = isolate_env();
            set("EMBEDDING_MODELS", alias);
            let cfg = EmbeddingConfig::from_env();
            assert!(
                matches!(cfg.models[0], EmbeddingModel::NomicEmbedTextV15),
                "alias `{}` should map to nomic",
                alias
            );
        }
    }

    #[test]
    fn from_env_parses_minilm_aliases() {
        for alias in ["all-minilm", "minilm", "MiniLM"] {
            let _g = isolate_env();
            set("EMBEDDING_MODELS", alias);
            let cfg = EmbeddingConfig::from_env();
            assert!(
                matches!(cfg.models[0], EmbeddingModel::AllMiniLML6V2),
                "alias `{}` should map to minilm",
                alias
            );
        }
    }

    #[test]
    fn from_env_parses_bge_aliases() {
        for alias in ["bge-small", "bge", "BGE"] {
            let _g = isolate_env();
            set("EMBEDDING_MODELS", alias);
            let cfg = EmbeddingConfig::from_env();
            assert!(
                matches!(cfg.models[0], EmbeddingModel::BGESmallENV15),
                "alias `{}` should map to bge",
                alias
            );
        }
    }

    #[test]
    #[should_panic(expected = "model not available")]
    fn from_env_unknown_model_panics() {
        let _g = isolate_env();
        set("EMBEDDING_MODELS", "completely-made-up");
        let _ = EmbeddingConfig::from_env();
    }

    #[test]
    fn from_env_parses_comma_separated_list() {
        let _g = isolate_env();
        set("EMBEDDING_MODELS", "nomic, minilm , bge");
        let cfg = EmbeddingConfig::from_env();
        assert_eq!(
            cfg.models,
            vec![
                EmbeddingModel::NomicEmbedTextV15,
                EmbeddingModel::AllMiniLML6V2,
                EmbeddingModel::BGESmallENV15,
            ]
        );
    }

    #[test]
    fn from_env_reads_cache_dir() {
        let _g = isolate_env();
        set("EMBEDDING_CACHE_DIR", "/tmp/embedding-cache");
        let cfg = EmbeddingConfig::from_env();
        assert_eq!(cfg.cache_dir.as_deref(), Some("/tmp/embedding-cache"));
    }

    #[test]
    fn from_env_parses_pool_size() {
        let _g = isolate_env();
        set("EMBEDDING_POOL_SIZE", "7");
        let cfg = EmbeddingConfig::from_env();
        assert_eq!(cfg.pool_size, 7);
    }

    #[test]
    fn from_env_rejects_zero_pool_size() {
        let _g = isolate_env();
        set("EMBEDDING_POOL_SIZE", "0");
        let cfg = EmbeddingConfig::from_env();
        assert_eq!(cfg.pool_size, available_cpus());
    }

    #[test]
    fn from_env_rejects_non_numeric_pool_size() {
        let _g = isolate_env();
        set("EMBEDDING_POOL_SIZE", "not-a-number");
        let cfg = EmbeddingConfig::from_env();
        assert_eq!(cfg.pool_size, available_cpus());
    }

    #[test]
    fn from_env_parses_intra_threads() {
        let _g = isolate_env();
        set("EMBEDDING_INTRA_THREADS", "2");
        let cfg = EmbeddingConfig::from_env();
        assert_eq!(cfg.intra_threads, 2);
    }

    #[test]
    fn from_env_parses_idle_unload_secs() {
        let _g = isolate_env();
        set("EMBEDDING_IDLE_UNLOAD_SECS", "0");
        let cfg = EmbeddingConfig::from_env();
        assert_eq!(cfg.idle_unload_secs, 0);
        set("EMBEDDING_IDLE_UNLOAD_SECS", "60");
        let cfg = EmbeddingConfig::from_env();
        assert_eq!(cfg.idle_unload_secs, 60);
    }

    #[test]
    fn new_does_not_load_onnx_sessions() {
        let cfg = EmbeddingConfig {
            models: vec![EmbeddingModel::AllMiniLML6V2],
            show_download_progress: false,
            cache_dir: None,
            pool_size: 1,
            execution_providers: Vec::new(),
            sub_batch_size: 0,
            intra_threads: 1,
            idle_unload_secs: 0,
        };
        let client = EmbeddingClient::new(cfg).expect("lazy construct");
        assert_eq!(client.idle_unload_secs(), 0);
        assert_eq!(client.models.len(), 1);
    }

    #[test]
    fn should_unload_requires_prior_use_and_timeout() {
        assert!(!should_unload(0, 1_000, 300));
        assert!(!should_unload(900, 1_000, 0));
        assert!(!should_unload(800, 1_000, 300));
        assert!(should_unload(700, 1_000, 300));
    }

    #[test]
    fn compute_sub_batch_shrinks_to_fit_small_budgets() {
        assert_eq!(EmbeddingClient::compute_sub_batch(0, 768, false), 1);
        // 200 MiB free -> 100 MiB budget -> one 100 MiB text, not four.
        assert_eq!(EmbeddingClient::compute_sub_batch(200, 768, false), 1);
        assert_eq!(
            EmbeddingClient::compute_sub_batch(64 * 1024, 768, false),
            16
        );
        assert_eq!(
            EmbeddingClient::compute_sub_batch(64 * 1024, 768, true),
            256
        );
    }

    #[test]
    fn next_index_round_robins_through_pool() {
        let counter = AtomicUsize::new(0);
        let observed: Vec<usize> = (0..7).map(|_| next_index(&counter, 3)).collect();
        assert_eq!(observed, vec![0, 1, 2, 0, 1, 2, 0]);
    }

    #[test]
    fn next_index_pool_of_one_always_zero() {
        let counter = AtomicUsize::new(0);
        for _ in 0..5 {
            assert_eq!(next_index(&counter, 1), 0);
        }
    }

    #[test]
    fn next_index_continues_from_existing_counter_value() {
        let counter = AtomicUsize::new(5);
        // 5 % 3 = 2, then 6 % 3 = 0, then 7 % 3 = 1
        assert_eq!(next_index(&counter, 3), 2);
        assert_eq!(next_index(&counter, 3), 0);
        assert_eq!(next_index(&counter, 3), 1);
    }

    #[test]
    fn default_pool_size_follows_cpu_count() {
        assert_eq!(default_pool_size(), available_cpus());
        assert!(default_pool_size() >= 1);
    }

    #[test]
    fn default_intra_threads_is_capped() {
        assert!(default_intra_threads() >= 1);
        assert!(default_intra_threads() <= 4);
        assert!(available_cpus() >= 1);
    }
}
