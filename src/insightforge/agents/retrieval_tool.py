from __future__ import annotations

from typing import Any

from insightforge.rag.answer import answer_with_evidence
from insightforge.rag.retriever import HybridRetriever


class KnowledgeTool:
    """Search approved local knowledge and return evidence-grounded answers."""

    def __init__(self) -> None:
        self._retriever: HybridRetriever | None = None

    def _get_retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever()
        return self._retriever

    def search(self, query: str, top_k: int = 3) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("Search query must not be empty.")

        if top_k < 1 or top_k > 5:
            raise ValueError("top_k must be between 1 and 5.")

        return answer_with_evidence(
            query=query,
            retriever=self._get_retriever(),
            top_k=top_k,
        )
