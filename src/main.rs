mod qdrant;

use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::post, Json, Router};
use qdrant::RAGSystem;
use qdrant_client::client::QdrantClient;
use serde::Deserialize;

#[derive(Deserialize)]
struct Prompt {
    prompt: String,
}

async fn prompt(
    State(state): State<RAGSystem>,
    Json(prompt): Json<Prompt>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let search_result = match state.search(&prompt.prompt).await {
        Ok(res) => res,
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("An error occurred while prompting: {e}"),
            ))
        }
    };

    match state.prompt(&prompt.prompt, &search_result).await {
        Ok(prompt_result) => Ok((StatusCode::OK, prompt_result)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Something went wrong while prompting: {e}"),
        )),
    }
}
#[shuttle_runtime::main]
async fn main(#[shuttle_qdrant::Qdrant] qdrant: QdrantClient) -> shuttle_axum::ShuttleAxum {
    let rag = RAGSystem::new(qdrant);

    let setup_required = true;

    if setup_required {
        rag.create_regular_collection().await?;
        rag.create_cache_collection().await?;
    }

    let rtr = Router::new().route("/prompt", post(prompt)).with_state(rag);

    Ok(rtr.into())
}
