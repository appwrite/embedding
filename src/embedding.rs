use fastembed::{EmbeddingModel, ExecutionProviderDispatch, InitOptions, TextEmbedding};
use futures::lock::Mutex;
use std::sync::{Arc, atomic::AtomicUsize};

#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub model: EmbeddingModel,
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

impl EmbeddingConfig {
    pub fn from_env() -> Self {
        let model = std::env::var("EMBEDDING_DEFAULT_MODEL")
            .ok()
            .and_then(|m| match m.to_lowercase().as_str() {
                "embedding-gemma" => Some(EmbeddingModel::EmbeddingGemma300M),
                "nomic-embed-text" | "nomic" => Some(EmbeddingModel::NomicEmbedTextV15),
                "all-minilm" | "minilm" => Some(EmbeddingModel::AllMiniLML6V2),
                "bge-small" | "bge" => Some(EmbeddingModel::BGESmallENV15),
                _ => None,
            })
            .unwrap_or(EmbeddingModel::NomicEmbedTextV15);

        let cache_dir = std::env::var("EMBEDDING_CACHE_DIR").ok();

        let pool_size = std::env::var("EMBEDDING_POOL_SIZE")
            .ok()
            .and_then(|size| size.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or_else(default_pool_size);

        Self {
            model,
            show_download_progress: true,
            cache_dir,
            pool_size,
            execution_providers: Vec::new(),
            sub_batch_size: 0,
        }
    }
}

pub struct EmbeddingClient {
    pool: Vec<Arc<Mutex<TextEmbedding>>>,
    next: AtomicUsize,
    dimension: usize,
    model_name: String,
    sub_batch_override: usize,
    gpu: bool,
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

        let dimension = match config.model {
            EmbeddingModel::NomicEmbedTextV15 => 768,
            EmbeddingModel::NomicEmbedTextV1 => 768,
            EmbeddingModel::AllMiniLML6V2 => 384,
            EmbeddingModel::BGESmallENV15 => 384,
            EmbeddingModel::BGEBaseENV15 => 768,
            EmbeddingModel::BGELargeENV15 => 1024,
            _ => 768,
        };

        let model_name = format!("{:?}", config.model);
        let desired_pool_size = config.pool_size.max(1);

        // loading instance and measuring memory footprint
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        let mem_before_loading_model = sys.available_memory();

        let has_gpu_providers = !config.execution_providers.is_empty();

        let mut first_model = Self::init_model(&config)?;

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
        let pool_size = if per_instance_bytes > 0 {
            // 60% of memory that was available before loading first model
            let budget = mem_before_loading_model * 6 / 10;
            let max_memory = (budget / per_instance_bytes).max(1) as usize;
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
            let mut model = Self::init_model(&config)?;
            pool.push(Arc::new(Mutex::new(model)));
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

        let sub_batch_override = config.sub_batch_size;

        Ok(Self {
            pool,
            next: AtomicUsize::new(0),
            dimension,
            model_name,
            sub_batch_override,
            gpu: has_gpu_providers,
        })
    }

    fn init_model(config: &EmbeddingConfig) -> Result<TextEmbedding, String> {
        let mut init_options = InitOptions::new(config.model.clone())
            .with_show_download_progress(config.show_download_progress);

        if let Some(cache_dir) = &config.cache_dir {
            init_options = init_options.with_cache_dir(cache_dir.into());
        }

        if !config.execution_providers.is_empty() {
            init_options =
                init_options.with_execution_providers(config.execution_providers.clone());
        }

        Ok(TextEmbedding::try_new(init_options)
            .map_err(|e| format!("Failed to initialize embedding model: {}", e.to_string()))?)
    }
}
