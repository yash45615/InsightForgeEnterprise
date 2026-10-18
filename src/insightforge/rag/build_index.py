from __future__ import annotations

import json

import numpy as np
from fastembed import TextEmbedding

from insightforge.rag.ingest import ARTIFACT_DIR, build_chunks

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main() -> None:
    chunks = build_chunks()
    if not chunks:
        raise ValueError("No document chunks were generated.")

    model = TextEmbedding(model_name=MODEL_NAME)
    texts = [str(chunk["text"]) for chunk in chunks]
    embeddings = np.asarray(
        list(model.embed(texts)),
        dtype=np.float32,
    )

    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError("Embedding output shape does not match chunk count.")

    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms == 0):
        raise ValueError("The embedding model produced a zero vector.")

    embeddings = embeddings / norms[:, None]

    np.save(ARTIFACT_DIR / "embeddings.npy", embeddings)
    (ARTIFACT_DIR / "index_metadata.json").write_text(
        json.dumps(
            {
                "model_name": MODEL_NAME,
                "chunk_count": len(chunks),
                "embedding_dimension": int(embeddings.shape[1]),
                "normalized": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Embedding model: {MODEL_NAME}")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"Index saved to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
