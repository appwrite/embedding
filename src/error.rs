#[derive(Debug)]
pub enum EmbedError {
    UnknownModel(String),
    TooLong {
        model: String,
        index: usize,
        tokens: usize,
        limit: usize,
    },
    Internal(String),
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbedError::UnknownModel(m) | EmbedError::Internal(m) => write!(f, "{}", m),
            EmbedError::TooLong {
                model,
                index,
                tokens,
                limit,
            } => write!(
                f,
                "text at index {index} has {tokens} tokens, the limit for {model} is {limit}"
            ),
        }
    }
}

impl std::error::Error for EmbedError {}

impl From<String> for EmbedError {
    fn from(s: String) -> Self {
        EmbedError::Internal(s)
    }
}
