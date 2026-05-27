use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
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
    let client = Arc::new(EmbeddingClient::new(config)?);
    let state = AppState { client };

    let app = Router::new().route("/embed", post(embed)).with_state(state);

    let addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:3000".to_string());
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
