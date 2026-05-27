use embedding::{EmbeddingClient, EmbeddingConfig};

/// Build-time warmup: resolves the configured models from the environment and
/// constructs the client, which downloads each model into the cache dir. Run
/// during `docker build` so the image ships with models already cached instead
/// of fetching them on first request.
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = EmbeddingConfig::from_env();
    tracing::info!("warmup: downloading and initializing {} model(s)", config.models.len());
    let _ = EmbeddingClient::new(config)?;
    tracing::info!("warmup: models cached and ready");

    Ok(())
}
