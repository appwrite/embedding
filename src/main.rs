use crate::embedding::{EmbeddingClient, EmbeddingConfig};

mod embedding;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = EmbeddingConfig::from_env();
    let client = EmbeddingClient::new(config)?;

    let texts = vec!["hello world", "another piece of text"];
    let outcome = client.embed(&texts).await?;

    for (text, embedding) in texts.iter().zip(outcome.embeddings.iter()) {
        let preview = &embedding[..5.min(embedding.len())];
        println!("{text:?} -> dim={} first5={preview:?}", embedding.len());
    }
    println!(
        "model={} total_tokens={} batch_size={}",
        outcome.model,
        outcome.tokens,
        outcome.embeddings.len()
    );

    Ok(())
}
