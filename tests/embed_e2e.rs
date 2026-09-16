use embedding::{EmbeddingClient, EmbeddingConfig, EmbeddingModel};
use std::sync::Arc;
use std::time::Duration;

fn small_model_config(pool_size: usize) -> EmbeddingConfig {
    EmbeddingConfig {
        models: vec![EmbeddingModel::AllMiniLML6V2],
        show_download_progress: false,
        cache_dir: None,
        pool_size,
        execution_providers: Vec::new(),
        sub_batch_size: 0,
        intra_threads: 1,
        idle_unload_secs: 0,
    }
}

fn nomic_config() -> EmbeddingConfig {
    EmbeddingConfig {
        models: vec![EmbeddingModel::NomicEmbedTextV15],
        show_download_progress: false,
        cache_dir: None,
        pool_size: 1,
        execution_providers: Vec::new(),
        sub_batch_size: 0,
        intra_threads: 1,
        idle_unload_secs: 0,
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
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
#[ignore = "downloads the AllMiniLML6V2 ONNX model on first run"]
async fn embed_after_idle_unload_reloads() {
    let mut cfg = small_model_config(1);
    cfg.idle_unload_secs = 1;
    let client = EmbeddingClient::new(cfg).expect("client init");

    let first = client
        .embed("minilm", &["hello world"])
        .await
        .expect("first embed should succeed");
    assert_eq!(first.embeddings.len(), 1);
    assert_eq!(first.embeddings[0].len(), 384);

    tokio::time::sleep(Duration::from_secs(2)).await;
    client.unload_idle();

    let second = client
        .embed("minilm", &["hello world"])
        .await
        .expect("embed after idle unload should reload");
    assert_eq!(second.embeddings.len(), 1);
    assert_eq!(second.embeddings[0].len(), 384);
    assert!(second.tokens > 0);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "downloads the AllMiniLML6V2 ONNX model on first run"]
async fn in_flight_embed_survives_idle_unload() {
    let mut cfg = small_model_config(1);
    cfg.idle_unload_secs = 1;
    let client = Arc::new(EmbeddingClient::new(cfg).expect("client init"));

    let embed_client = client.clone();
    let embed_task = tokio::spawn(async move {
        embed_client
            .embed(
                "minilm",
                &[
                    "the cat sat on the mat",
                    "rust is a systems language",
                    "idle unload must not drop in-flight sessions",
                ],
            )
            .await
    });

    for _ in 0..50 {
        client.unload_idle();
        tokio::time::sleep(Duration::from_millis(2)).await;
    }

    let result = embed_task
        .await
        .expect("embed task should join")
        .expect("in-flight embed should succeed while unload runs");
    assert_eq!(result.embeddings.len(), 3);
    for embedding in &result.embeddings {
        assert_eq!(embedding.len(), 384);
    }

    tokio::time::sleep(Duration::from_secs(2)).await;
    client.unload_idle();
    let after = client
        .embed("minilm", &["hello"])
        .await
        .expect("embed after in-flight request should succeed");
    assert_eq!(after.embeddings[0].len(), 384);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "downloads the NomicEmbedTextV15 ONNX model on first run"]
async fn nomic_embed_is_768d_and_ranks_similar_text_higher() {
    let client = EmbeddingClient::new(nomic_config()).expect("client init");
    let result = client
        .embed(
            "nomic",
            &[
                "The cat sat on the mat",
                "A kitten rested on the rug",
                "Rust is a systems programming language",
            ],
        )
        .await
        .expect("nomic embed should succeed");

    assert!(
        result.model.to_lowercase().contains("nomic"),
        "public model identity should be nomic, got {}",
        result.model
    );
    assert_eq!(result.embeddings.len(), 3);
    for embedding in &result.embeddings {
        assert_eq!(embedding.len(), 768);
    }
    assert!(result.tokens > 0);

    let similar = cosine_similarity(&result.embeddings[0], &result.embeddings[1]);
    let dissimilar = cosine_similarity(&result.embeddings[0], &result.embeddings[2]);
    assert!(
        similar > dissimilar,
        "related sentences should rank above an unrelated one: similar={similar} dissimilar={dissimilar}"
    );
}
