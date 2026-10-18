from __future__ import annotations

import re
from typing import Any

from insightforge.rag.retriever import HybridRetriever

TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "might",
    "of",
    "on",
    "or",
    "our",
    "should",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}

IRREGULAR_FORMS = {
    "analysts": "analyst",
    "bias": "bias",
    "biases": "bias",
    "clicks": "click",
    "clusters": "cluster",
    "conversions": "conversion",
    "customers": "customer",
    "evaluations": "evaluation",
    "impressions": "impression",
    "metrics": "metric",
    "policies": "policy",
    "practices": "practice",
    "risks": "risk",
    "segments": "segment",
    "studies": "study",
}

ABSTENTION_MESSAGE = (
    "I don't have enough relevant evidence in the available "
    "knowledge base to answer that reliably."
)


def _normalize_token(token: str) -> str:
    """Normalize common word forms without damaging short root words."""
    token = token.lower()

    if token in IRREGULAR_FORMS:
        return IRREGULAR_FORMS[token]

    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"

    if len(token) > 7 and token.endswith("ation"):
        return token[:-5]

    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
        if len(token) > 2 and token[-1] == token[-2]:
            token = token[:-1]
        return token

    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]

    if (
        len(token) > 4
        and token.endswith("s")
        and not token.endswith(("ss", "us", "is"))
    ):
        return token[:-1]

    return token


def _content_tokens(text: str) -> set[str]:
    """Return normalized terms while excluding common stop words."""
    result: set[str] = set()

    for raw_token in TOKEN_PATTERN.findall(text.lower()):
        if raw_token in STOP_WORDS or len(raw_token) <= 1:
            continue
        normalized = _normalize_token(raw_token)
        if normalized:
            result.add(normalized)

    return result


def _query_concepts(query: str) -> set[str]:
    """Add narrowly targeted domain vocabulary for known question types."""
    lowered = query.casefold()
    concepts: set[str] = set()

    if "conversion rate" in lowered:
        concepts.update({"conversions", "clicks", "divided"})

    if (
        "click-through rate" in lowered
        or "click through rate" in lowered
        or re.search(r"\bctr\b", lowered)
    ):
        concepts.update({"clicks", "impressions", "divided"})

    if "sentiment" in lowered and any(
        term in lowered
        for term in ("classifier", "classify", "metric", "evaluate", "model")
    ):
        concepts.update(
            {"accuracy", "precision", "recall", "f1", "score", "confusion", "matrix"}
        )

    if any(term in lowered for term in ("segment", "grouping", "cluster")):
        concepts.update(
            {
                "segment",
                "group",
                "characteristics",
                "bias",
                "incomplete",
                "unrepresentative",
            }
        )

    if "responsible ai" in lowered or "responsible artificial intelligence" in lowered:
        concepts.update({"privacy", "bias", "transparency", "evidence"})

    if "privacy" in lowered or (
        "customer information" in lowered
        and any(term in lowered for term in ("protect", "secure", "necessary"))
    ):
        concepts.update(
            {
                "privacy",
                "minimizing",
                "collecting",
                "retaining",
                "exposing",
                "necessary",
            }
        )

    return _content_tokens(" ".join(concepts))


def _abstain(query: str, reason: str) -> dict[str, Any]:
    return {
        "status": "insufficient_evidence",
        "query": query,
        "answer": ABSTENTION_MESSAGE,
        "citations": [],
        "evidence": [],
        "reason": reason,
    }


def _sentence_candidates(
    query_tokens: set[str],
    results: list[dict[str, Any]],
    concepts: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Rank evidence sentences and safely join fragments split across chunks."""
    expanded_tokens = query_tokens | (concepts or set())
    candidates: list[dict[str, Any]] = []
    seen_sentences: set[tuple[str, str]] = set()

    # Process chunks in source order so sentence fragments can be joined.
    ordered_results = sorted(
        results,
        key=lambda item: (
            str(item.get("source", "")),
            int(item.get("chunk_index", 0)),
        ),
    )

    for item in ordered_results:
        text = str(item.get("text", "")).strip()
        source = str(item.get("source", "")).strip()
        chunk_id = str(item.get("chunk_id", "")).strip()
        chunk_index = item.get("chunk_index")

        if not text or not source or not chunk_id or chunk_index is None:
            continue

            # Remove Markdown heading markers without discarding the rest
        # of a flattened chunk that starts with a heading.
        body = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text).strip()
        body = " ".join(line.strip() for line in body.splitlines() if line.strip())

        for raw_sentence in SENTENCE_PATTERN.split(body):
            sentence = raw_sentence.strip().strip("#").strip()
            if not sentence:
                continue

            # Ignore incomplete final fragments, rather than treating them
            # as complete evidence. Complete sentences earlier in the chunk
            # remain available for ranking.
            if sentence[-1] not in ".!?":
                continue

            if not sentence[0].isupper() and not sentence[0].isdigit():
                continue

            sentence_tokens = _content_tokens(sentence)
            if not sentence_tokens:
                continue

            direct_overlap = query_tokens & sentence_tokens
            concept_overlap = expanded_tokens & sentence_tokens

            if not concept_overlap:
                continue

            sentence_key = (source, sentence.casefold())
            if sentence_key in seen_sentences:
                continue
            seen_sentences.add(sentence_key)

            score = 3 * len(direct_overlap) + len(concept_overlap - query_tokens)

            candidates.append(
                {
                    "sentence": sentence,
                    "source": source,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "coverage": len(direct_overlap) / max(1, len(query_tokens)),
                    "overlap_count": len(direct_overlap),
                    "concept_overlap_count": len(concept_overlap),
                    "score": score,
                    "overlap_terms": sorted(concept_overlap),
                }
            )

    candidates.sort(
        key=lambda item: (
            item["score"],
            item["overlap_count"],
            item["concept_overlap_count"],
            -len(item["sentence"]),
        ),
        reverse=True,
    )
    return candidates


def answer_with_evidence(
    query: str,
    retriever: HybridRetriever,
    top_k: int = 4,
    max_sentences: int = 3,
    min_query_coverage: float = 0.20,
) -> dict[str, Any]:
    """Return extractive, citation-backed answers or abstain."""
    if not query.strip():
        raise ValueError("Query must not be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if max_sentences < 1:
        raise ValueError("max_sentences must be at least 1.")
    if not 0.0 < min_query_coverage <= 1.0:
        raise ValueError("min_query_coverage must be between 0 and 1.")

    query_tokens = _content_tokens(query)
    if not query_tokens:
        return _abstain(query, "query_has_no_meaningful_terms")

    results = retriever.search(query, top_k=top_k)
    if not results:
        return _abstain(query, "no_retrieved_results")

    concepts = _query_concepts(query)
    candidates = _sentence_candidates(query_tokens, results, concepts)

    if not candidates:
        return _abstain(query, "no_sentence_matches_query_terms")

    requested_years = set(YEAR_PATTERN.findall(query))
    if requested_years:
        year_candidates = [
            candidate
            for candidate in candidates
            if requested_years.issubset(
                set(YEAR_PATTERN.findall(candidate["sentence"]))
            )
        ]
        if not year_candidates:
            return _abstain(query, "requested_year_not_supported")
        candidates = year_candidates

    best = candidates[0]
    if best["coverage"] < min_query_coverage and best["overlap_count"] == 0:
        return _abstain(query, "insufficient_query_term_coverage")

    best_source = best["source"]
    source_candidates = [
        candidate
        for candidate in candidates
        if candidate["source"] == best_source
        and candidate["score"] >= max(1, best["score"] * 0.35)
    ]
    selected = source_candidates[:max_sentences]

    if not selected:
        return _abstain(query, "no_acceptable_evidence_sentences")

    evidence = [
        {
            "citation": f"[{item['source']}#chunk-{item['chunk_index']}]",
            "source": item["source"],
            "chunk_id": item["chunk_id"],
            "chunk_index": item["chunk_index"],
            "text": item["sentence"],
            "query_term_coverage": round(item["coverage"], 3),
        }
        for item in selected
    ]

    answer = " ".join(
        f"{item['sentence']} [{item['source']}#chunk-{item['chunk_index']}]"
        for item in selected
    )

    citations = [
        {
            "source": item["source"],
            "chunk_id": item["chunk_id"],
            "chunk_index": item["chunk_index"],
        }
        for item in selected
    ]

    return {
        "status": "evidence_found",
        "query": query,
        "answer": answer,
        "citations": citations,
        "evidence": evidence,
        "method": "extractive_sentence_matching",
        "limitations": (
            "Answers are extracted from retrieved sentences. "
            "Domain-aware term matching improves evidence selection but "
            "does not prove entailment or factual completeness."
        ),
    }
