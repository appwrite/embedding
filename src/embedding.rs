use crate::model::{self, EmbeddingModel};
use fastembed::{ExecutionProviderDispatch, InitOptions, TextEmbedding, get_cache_dir};
use hf_hub::api::sync::ApiBuilder;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use tokenizers::Tokenizer;

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
}

fn default_pool_size() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
}

fn next_index(counter: &AtomicUsize, len: usize) -> usize {
    counter.fetch_add(1, Ordering::Relaxed) % len
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

        let pool_size = std::env::var("EMBEDDING_POOL_SIZE")
            .ok()
            .and_then(|size| size.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or_else(default_pool_size);

        Self {
            models,
            show_download_progress: true,
            cache_dir,
            pool_size,
            execution_providers: Vec::new(),
            sub_batch_size: 0,
        }
    }
}

pub struct EmbeddingClient {
    models: HashMap<String, LoadedModel>,
    sub_batch_override: usize,
    gpu: bool,
}

struct LoadedModel {
    model_name: String,
    next: AtomicUsize,
    pool: Vec<Arc<Mutex<TextEmbedding>>>,
    dimension: usize,
    tokenizer: Arc<Tokenizer>,
}

impl EmbeddingClient {
    pub fn new(config: EmbeddingConfig) -> Result<Self, String> {
        // ONNX model loading memory != actual inference memory.
        //
        // `TextEmbedding::try_new()` mainly loads:
        // - model weights
        // - tokenizer
        // - ONNX graph/session
        //
        // However, ONNX Runtime lazily allocates most execution memory
        // (attention buffers, tensor arenas, activations, kernel workspaces)
        // only during the first real inference call.
        //
        // We therefore run a warmup inference before measuring memory usage,
        // otherwise pool sizing would severely underestimate the true runtime
        // footprint and may cause OOMs under load.
        let mut models = HashMap::new();
        for model in &config.models {
            let model_name = format!("{:?}", model);
            let loaded_model = Self::load_model(model, &model_name, &config)?;
            models.insert(model_name, loaded_model);
        }

        let sub_batch_override = config.sub_batch_size;
        let gpu = !config.execution_providers.is_empty();
        Ok(Self {
            models,
            sub_batch_override,
            gpu,
        })
    }

    fn load_model(
        model: &EmbeddingModel,
        model_name: &str,
        config: &EmbeddingConfig,
    ) -> Result<LoadedModel, String> {
        let desired_pool_size = config.pool_size.max(1);
        let dimension = model::dimension(model);
        // loading instance and measuring memory footprint
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        let mem_before_loading_model = sys.available_memory();

        let has_gpu_providers = !config.execution_providers.is_empty();

        let mut first_model = Self::init_model(model, config)?;

        // Tokenizer is fetched from the same cache dir fastembed just populated,
        // so this is a cache hit (no network) after the first model load.
        let tokenizer = Arc::new(Self::load_tokenizer(model, config)?);

        // Run a warmup inference so the ONNX Runtime arena is allocated before
        // we measure memory.  Without this, per_instance only captures model
        // weights and misses the arena buffers.
        let _ = first_model.embed(vec!["warmup"], None);

        sys.refresh_memory();
        let memory_after_loading_model = sys.available_memory();
        let per_instance_loaded =
            mem_before_loading_model.saturating_sub(memory_after_loading_model);

        // ONNX Runtime uses arena allocation that grows with
        // batch_size × sequence_length² (attention matrices) and is never
        // released.  The warmup above only allocates a minimal arena for a
        // single short text.  Apply a 3× multiplier to account for realistic
        // inference workloads (batch=8-32 texts of 1000-2000 tokens each).
        let per_instance_bytes = per_instance_loaded.saturating_mul(3);

        // determining pool size based on ram and capacity provided
        let nproc = default_pool_size();
        // 60% of memory that was available before loading first model
        let budget = mem_before_loading_model * 6 / 10;
        let pool_size = if let Some(max_memory) = budget.checked_div(per_instance_bytes) {
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
        };

        // loading remaining instances in the pool along with the first model
        let mut pool = Vec::with_capacity(pool_size);
        pool.push(Arc::new(Mutex::new(first_model)));

        for _ in 1..pool_size {
            let inst = Self::init_model(model, config)?;
            pool.push(Arc::new(Mutex::new(inst)));
        }

        let ep_label = if has_gpu_providers {
            "GPU (CUDA)"
        } else {
            "CPU"
        };
        tracing::info!(
            "Initialized embedding model: {} ({}d, pool_size={}, execution_provider={})",
            model_name,
            dimension,
            pool_size,
            ep_label,
        );

        Ok(LoadedModel {
            dimension,
            model_name: model_name.to_string(),
            next: AtomicUsize::new(0),
            pool,
            tokenizer,
        })
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
            .with_show_download_progress(config.show_download_progress);

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
    pub async fn embed(&self, model_name: &str, texts: &[&str]) -> Result<EmbeddingResult, String> {
        let started = std::time::Instant::now();
        // Accept any alias the user might type ("minilm", "all-minilm", "MiniLM"…)
        // by resolving to the canonical Debug name we used as the HashMap key.
        let resolved = model::from_name(model_name)
            .ok_or_else(|| format!("unknown model alias: {}", model_name))?;
        let canonical = format!("{:?}", resolved);
        let loaded = self
            .models
            .get(&canonical)
            .ok_or_else(|| format!("model not allowed: {}", model_name))?;

        let sub_batch = if self.sub_batch_override > 0 {
            self.sub_batch_override
        } else {
            Self::compute_sub_batch(loaded.dimension, self.gpu)
        };

        let mut handles = Vec::new();
        for chunk in texts.chunks(sub_batch) {
            let inst = Self::acquire(loaded);
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

        let tokenizer = loaded.tokenizer.clone();
        let owned_texts: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
        let tokens = tokio::task::spawn_blocking(move || -> Result<usize, String> {
            let encodings = tokenizer
                .encode_batch(owned_texts, true)
                .map_err(|e| format!("tokenize: {}", e))?;
            Ok(encodings.iter().map(|e| e.get_ids().len()).sum())
        })
        .await
        .map_err(|e| format!("Failed to join tokenizer task: {}", e))??;

        Ok(EmbeddingResult {
            model: loaded.model_name.clone(),
            embeddings,
            tokens,
            total_duration: started.elapsed().as_nanos() as u64,
        })
    }

    /// Round-robin acquire one instance from a model's pool.
    fn acquire(loaded: &LoadedModel) -> Arc<Mutex<TextEmbedding>> {
        let idx = next_index(&loaded.next, loaded.pool.len());
        loaded.pool[idx].clone()
    }

    /// Compute sub-batch size based on available system memory.
    ///
    /// Uses 50% of available RAM as a budget.  Falls back to 32 if sysinfo
    /// reports 0.
    /// When GPU is enabled, the upper clamp is raised to 256 (GPU VRAM can
    /// handle much larger batches than CPU).
    /// TODO: add here the config batch size
    fn compute_sub_batch(dimension: usize, gpu: bool) -> usize {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        let available_mb = sys.available_memory() / (1024 * 1024);

        if available_mb == 0 {
            return 32;
        }

        // Per-text memory estimate for ONNX inference.  Attention matrices
        // dominate: heads × seq² × 4 bytes.  For 768-dim BERT-like models
        // (12 heads) processing ~1000-2000 token code chunks, attention alone
        // is 50-200 MB per text.  The estimate below is conservative so the
        // sub-batch stays small enough to prevent arena over-allocation.
        let mb_per_text: u64 = if dimension >= 768 { 100 } else { 40 };
        let budget_mb = available_mb / 2;
        let max_batch = if gpu { 256 } else { 16 };
        (budget_mb / mb_per_text).clamp(4, max_batch) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    const ENV_KEYS: [&str; 3] = [
        "EMBEDDING_MODELS",
        "EMBEDDING_CACHE_DIR",
        "EMBEDDING_POOL_SIZE",
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
    fn from_env_uses_nomic_when_unset() {
        let _g = isolate_env();
        let cfg = EmbeddingConfig::from_env();
        assert!(matches!(cfg.models[0], EmbeddingModel::NomicEmbedTextV15));
        assert_eq!(cfg.cache_dir, None);
        assert!(cfg.pool_size >= 1);
        assert!(cfg.show_download_progress);
        assert!(cfg.execution_providers.is_empty());
        assert_eq!(cfg.sub_batch_size, 0);
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
        assert_eq!(cfg.pool_size, default_pool_size());
    }

    #[test]
    fn from_env_rejects_non_numeric_pool_size() {
        let _g = isolate_env();
        set("EMBEDDING_POOL_SIZE", "not-a-number");
        let cfg = EmbeddingConfig::from_env();
        assert_eq!(cfg.pool_size, default_pool_size());
    }

    #[test]
    fn compute_sub_batch_cpu_768_within_clamp() {
        let result = EmbeddingClient::compute_sub_batch(768, false);
        // 32 is the sysinfo-zero fallback; otherwise the CPU clamp is [4, 16].
        assert!(
            result == 32 || (4..=16).contains(&result),
            "got {} for cpu/768",
            result
        );
    }

    #[test]
    fn compute_sub_batch_cpu_384_within_clamp() {
        let result = EmbeddingClient::compute_sub_batch(384, false);
        assert!(
            result == 32 || (4..=16).contains(&result),
            "got {} for cpu/384",
            result
        );
    }

    #[test]
    fn compute_sub_batch_gpu_raises_ceiling() {
        let result = EmbeddingClient::compute_sub_batch(768, true);
        assert!(
            result == 32 || (4..=256).contains(&result),
            "got {} for gpu/768",
            result
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
    fn default_pool_size_is_positive() {
        assert!(default_pool_size() >= 1);
    }
}
