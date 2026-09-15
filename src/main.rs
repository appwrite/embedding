use std::sync::Arc;
use std::time::Duration;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};

use embedding::{EmbedError, EmbeddingClient, EmbeddingConfig};

#[derive(Clone)]
struct AppState {
    client: Arc<EmbeddingClient>,
}

#[derive(Deserialize)]
struct EmbedRequest {
    model: String,
    texts: Vec<String>,
}

#[derive(Serialize)]
struct EmbedResponse {
    model: String,
    embeddings: Vec<Vec<f32>>,
    tokens: usize,
    total_duration: u64,
}

struct AppError(StatusCode, String);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.0, Json(serde_json::json!({ "error": self.1 }))).into_response()
    }
}

async fn health() -> impl IntoResponse {
    StatusCode::OK
}

async fn embed(
    State(state): State<AppState>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, AppError> {
    if req.texts.is_empty() {
        return Err(AppError(
            StatusCode::BAD_REQUEST,
            "texts must not be empty".to_string(),
        ));
    }

    let refs: Vec<&str> = req.texts.iter().map(|s| s.as_str()).collect();
    let result = state
        .client
        .embed(&req.model, &refs)
        .await
        .map_err(|e| match e {
            EmbedError::UnknownModel(msg) => AppError(StatusCode::BAD_REQUEST, msg),
            EmbedError::Internal(msg) => AppError(StatusCode::INTERNAL_SERVER_ERROR, msg),
        })?;

    Ok(Json(EmbedResponse {
        model: result.model,
        embeddings: result.embeddings,
        tokens: result.tokens,
        total_duration: result.total_duration,
    }))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = EmbeddingConfig::from_env();
    let idle_unload_secs = config.idle_unload_secs;
    let client = Arc::new(EmbeddingClient::new(config)?);

    if idle_unload_secs > 0 {
        let client_bg = client.clone();
        let tick_secs = (idle_unload_secs / 6).clamp(10, 30);
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(tick_secs));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            ticker.tick().await; // skip the immediate first tick
            loop {
                ticker.tick().await;
                client_bg.unload_idle();
            }
        });
        tracing::info!(idle_unload_secs, tick_secs, "idle model unload enabled");
    }

    let state = AppState { client };

    let app = Router::new()
        .route("/health", get(health))
        .route("/embed", post(embed))
        .with_state(state);

    let port_str = std::env::var("EMBEDDING_PORT").unwrap_or_else(|_| "3000".to_string());
    let port: u16 = port_str.parse().map_err(|_| {
        format!(
            "EMBEDDING_PORT '{}' is not a valid port number (1-65535)",
            port_str
        )
    })?;
    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("listening on {}", addr);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("shutdown signal received, draining connections");
}
