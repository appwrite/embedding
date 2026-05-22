use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
mod embedding;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model
    let mut model = TextEmbedding::try_new(InitOptions::new(EmbeddingModel::AllMiniLML6V2))?;

    // Input texts
    let documents = vec!["Rust is fast", "Embeddings are useful"];

    // Generate embeddings
    let embeddings = model.embed(documents, None)?;

    println!("Number of embeddings: {}", embeddings.len());

    // Each embedding is Vec<f32>
    println!("Embedding dimension: {}", embeddings[0].len());

    // Print first few values
    println!("{:?}", &embeddings[0][0..5]);

    Ok(())
}
