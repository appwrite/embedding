use embedding::{EmbeddingClient, EmbeddingConfig};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = EmbeddingConfig::from_env();
    tracing::info!(
        "warmup: downloading and initializing {} model(s)",
        config.models.len()
    );
    let _ = EmbeddingClient::new(config)?;
    tracing::info!("warmup: models cached and ready");

    Ok(())
}
