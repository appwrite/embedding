mod embedding;
mod error;
mod model;

pub use embedding::{EmbeddingClient, EmbeddingConfig, EmbeddingResult};
pub use error::EmbedError;
pub use fastembed::ExecutionProviderDispatch;
pub use model::EmbeddingModel;
