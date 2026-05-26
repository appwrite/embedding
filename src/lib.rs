mod embedding;
mod model;

pub use embedding::{EmbeddingClient, EmbeddingConfig, EmbeddingResult};
pub use fastembed::ExecutionProviderDispatch;
pub use model::EmbeddingModel;
