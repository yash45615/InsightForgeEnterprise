from __future__ import annotations

import pytest

from insightforge.rag.answer import answer_with_evidence


class FakeRetriever:
    def __init__(self, results: list[dict[str, object]]) -> None:
        self.results = results

    def search(
        self,
        query: str,
        top_k: int = 4,
    ) -> list[dict[str, object]]:
        return self.results[:top_k]


def make_result(
    text: str,
    source: str = "campaign_metrics.md",
    chunk_id: str = "chunk-1",
    chunk_index: int = 0,
) -> dict[str, object]:
    return {
        "text": text,
        "source": source,
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "score": 0.03,
        "dense_score": 0.8,
        "keyword_score": 2.0,
    }


def test_answer_uses_retrieved_sentence_and_cites_source() -> None:
    retriever = FakeRetriever(
        [
            make_result(
                "Campaign conversion rates measure the share of users "
                "who complete a desired action. "
                "Customer feedback can reveal usability problems."
            )
        ]
    )

    result = answer_with_evidence(
        "What do campaign conversion rates measure?",
        retriever,  # type: ignore[arg-type]
    )

    assert result["status"] == "evidence_found"
    assert "Campaign conversion rates measure" in result["answer"]
    assert "[campaign_metrics.md#chunk-0]" in result["answer"]
    assert result["citations"][0]["source"] == "campaign_metrics.md"


def test_abstains_when_retriever_returns_no_results() -> None:
    result = answer_with_evidence(
        "What determines satellite launch windows?",
        FakeRetriever([]),  # type: ignore[arg-type]
    )

    assert result["status"] == "insufficient_evidence"
    assert result["citations"] == []
    assert "enough relevant evidence" in result["answer"]


def test_abstains_when_retrieved_text_has_no_query_term_overlap() -> None:
    retriever = FakeRetriever(
        [
            make_result(
                "Customer feedback describes satisfaction with support "
                "and the quality of service."
            )
        ]
    )

    result = answer_with_evidence(
        "What determines satellite launch windows?",
        retriever,  # type: ignore[arg-type]
    )

    assert result["status"] == "insufficient_evidence"
    assert result["citations"] == []
    assert result["evidence"] == []


def test_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="Query must not be empty"):
        answer_with_evidence(
            "   ",
            FakeRetriever([]),  # type: ignore[arg-type]
        )


def test_rejects_invalid_sentence_limit() -> None:
    with pytest.raises(ValueError, match="max_sentences"):
        answer_with_evidence(
            "campaign conversion rates",
            FakeRetriever([]),  # type: ignore[arg-type]
            max_sentences=0,
        )


def test_answer_citations_match_returned_evidence() -> None:
    retriever = FakeRetriever(
        [
            make_result(
                "Campaign conversion rates measure completed actions.",
                source="campaign_metrics.md",
                chunk_id="campaign-1",
                chunk_index=2,
            )
        ]
    )

    result = answer_with_evidence(
        "What do campaign conversion rates measure?",
        retriever,  # type: ignore[arg-type]
    )

    assert result["status"] == "evidence_found"
    assert len(result["citations"]) == len(result["evidence"])
    assert result["citations"][0]["chunk_id"] == "campaign-1"
    assert "[campaign_metrics.md#chunk-2]" in result["answer"]
