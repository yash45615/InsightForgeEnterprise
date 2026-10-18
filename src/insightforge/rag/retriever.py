from __future__ import annotations

import json
import re

import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

from insightforge.rag.build_index import MODEL_NAME
from insightforge.rag.ingest import ARTIFACT_DIR

TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class HybridRetriever:
    def __init__(self) -> None:
        chunks_path = ARTIFACT_DIR / "chunks.json"
        embeddings_path = ARTIFACT_DIR / "embeddings.npy"

        if not chunks_path.exists() or not embeddings_path.exists():
            raise FileNotFoundError(
                "RAG index missing. Run: python -m insightforge.rag.build_index"
            )

        self.chunks: list[dict[str, object]] = json.loads(
            chunks_path.read_text(encoding="utf-8")
        )
        self.embeddings = np.load(embeddings_path).astype(np.float32)

        if len(self.chunks) != self.embeddings.shape[0]:
            raise ValueError("Chunk count and embedding count do not match.")

        self.model = TextEmbedding(model_name=MODEL_NAME)
        tokenized_chunks = [tokenize(str(chunk["text"])) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def search(self, query: str, top_k: int = 4) -> list[dict[str, object]]:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_vector = np.asarray(
            list(self.model.query_embed([query])),
            dtype=np.float32,
        )[0]

        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            raise ValueError("The query embedding is a zero vector.")

        query_vector /= query_norm
        dense_scores = self.embeddings @ query_vector
        keyword_scores = np.asarray(
            self.bm25.get_scores(tokenize(query)), dtype=np.float32
        )

        # Reciprocal Rank Fusion combines the two rankings.
        dense_order = np.argsort(-dense_scores)
        keyword_order = np.argsort(-keyword_scores)
        fused_scores = np.zeros(len(self.chunks), dtype=np.float32)
        rank_constant = 60

        for ranking in (dense_order, keyword_order):
            for rank, index in enumerate(ranking, start=1):
                fused_scores[index] += 1.0 / (rank_constant + rank)

        result_count = min(top_k, len(self.chunks))
        best_indices = np.argsort(-fused_scores)[:result_count]

        results: list[dict[str, object]] = []
        for index in best_indices:
            chunk = self.chunks[int(index)]
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "score": float(fused_scores[index]),
                    "dense_score": float(dense_scores[index]),
                    "keyword_score": float(keyword_scores[index]),
                }
            )

        return results
