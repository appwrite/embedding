pub use fastembed::EmbeddingModel;

pub fn from_name(name: &str) -> Option<EmbeddingModel> {
    match name.trim().to_lowercase().as_str() {
        "embedding-gemma" => Some(EmbeddingModel::EmbeddingGemma300M),
        "nomic-embed-text" | "nomic" => Some(EmbeddingModel::NomicEmbedTextV15),
        "all-minilm" | "minilm" => Some(EmbeddingModel::AllMiniLML6V2),
        "bge-small" | "bge" => Some(EmbeddingModel::BGESmallENV15),
        _ => None,
    }
}

/// Embedding vector dimension for a given model.
pub fn dimension(model: &EmbeddingModel) -> usize {
    match model {
        EmbeddingModel::NomicEmbedTextV15 | EmbeddingModel::NomicEmbedTextV1 => 768,
        EmbeddingModel::AllMiniLML6V2 => 384,
        EmbeddingModel::BGESmallENV15 => 384,
        EmbeddingModel::BGEBaseENV15 => 768,
        EmbeddingModel::BGELargeENV15 => 1024,
        _ => 768,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_name_recognizes_aliases() {
        assert_eq!(
            from_name("nomic-embed-text"),
            Some(EmbeddingModel::NomicEmbedTextV15)
        );
        assert_eq!(from_name("NOMIC"), Some(EmbeddingModel::NomicEmbedTextV15));
        assert_eq!(from_name(" MiniLM "), Some(EmbeddingModel::AllMiniLML6V2));
        assert_eq!(from_name("bge"), Some(EmbeddingModel::BGESmallENV15));
        assert_eq!(from_name("not-a-model"), None);
    }

    #[test]
    fn dimension_known_models() {
        assert_eq!(dimension(&EmbeddingModel::AllMiniLML6V2), 384);
        assert_eq!(dimension(&EmbeddingModel::NomicEmbedTextV15), 768);
        assert_eq!(dimension(&EmbeddingModel::BGELargeENV15), 1024);
    }
}
