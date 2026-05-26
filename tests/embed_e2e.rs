use embedding::{EmbeddingClient, EmbeddingConfig, EmbeddingModel};

fn small_model_config(pool_size: usize) -> EmbeddingConfig {
    EmbeddingConfig {
        models: vec![EmbeddingModel::AllMiniLML6V2],
        show_download_progress: false,
        cache_dir: None,
        pool_size,
        execution_providers: Vec::new(),
        sub_batch_size: 0,
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "downloads the AllMiniLML6V2 ONNX model on first run"]
async fn embed_single_text_returns_expected_dimension() {
    let client = EmbeddingClient::new(small_model_config(1)).expect("client init");
    let result = client
        .embed("minilm", &["hello world"])
        .await
        .expect("embed should succeed");
    assert_eq!(result.embeddings.len(), 1);
    assert_eq!(result.embeddings[0].len(), 384);
    assert!(result.tokens > 0);
    assert!(result.model.contains("MiniLM"));
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "downloads the AllMiniLML6V2 ONNX model on first run"]
async fn embed_batch_produces_one_vector_per_input() {
    let client = EmbeddingClient::new(small_model_config(2)).expect("client init");
    let texts = vec!["alpha", "beta", "gamma", "delta"];
    let result = client
        .embed("minilm", &texts)
        .await
        .expect("embed should succeed");
    assert_eq!(result.embeddings.len(), texts.len());
    for embedding in &result.embeddings {
        assert_eq!(embedding.len(), 384);
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "downloads the AllMiniLML6V2 ONNX model on first run"]
async fn embed_distinct_inputs_produce_distinct_vectors() {
    let client = EmbeddingClient::new(small_model_config(1)).expect("client init");
    let result = client
        .embed(
            "minilm",
            &["the cat sat on the mat", "rust is a systems language"],
        )
        .await
        .expect("embed should succeed");
    assert_eq!(result.embeddings.len(), 2);
    assert_ne!(result.embeddings[0], result.embeddings[1]);
}

#[tokio::test(flavor = "multi_thread")]
async fn embed_unknown_alias_returns_error() {
    // No model loading on this path — fails the alias check first.
    let client = EmbeddingClient::new(small_model_config(1)).ok();
    if let Some(client) = client {
        let err = client.embed("not-a-real-model", &["x"]).await.unwrap_err();
        assert!(err.contains("unknown model alias"), "got: {}", err);
    }
}
