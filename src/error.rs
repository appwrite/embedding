#[derive(Debug)]
pub enum EmbedError {
    UnknownModel(String),
    Internal(String),
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbedError::UnknownModel(m) | EmbedError::Internal(m) => write!(f, "{}", m),
        }
    }
}

impl std::error::Error for EmbedError {}

impl From<String> for EmbedError {
    fn from(s: String) -> Self {
        EmbedError::Internal(s)
    }
}
