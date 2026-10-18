from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from insightforge.evaluation.rag_evaluator import load_cases
from insightforge.rag.answer import answer_with_evidence
from insightforge.rag.retriever import HybridRetriever

ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = ROOT / "reports" / "rag" / "answer_quality_report.json"

TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
CITATION_PATTERN = re.compile(r"\[([^\[\]]+\.md)#chunk-[^\[\]]+\]")


def normalize_terms(text: str) -> set[str]:
    """Extract normalized terms for a simple lexical evaluation."""
    return set(TOKEN_PATTERN.findall(text.lower()))


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize answer-quality and evidence-grounding checks."""
    total = len(results)

    if total == 0:
        return {
            "total_cases": 0,
            "answerability_accuracy": 0.0,
            "answerable_cases": 0,
            "unanswerable_cases": 0,
            "citation_grounding_rate": None,
            "expected_term_coverage": None,
        }

    answerable = [item for item in results if item["expected_answerable"]]
    unanswerable = [item for item in results if not item["expected_answerable"]]
    citation_cases = [item for item in results if item["citation_grounding_applicable"]]
    coverage_values = [
        item["expected_term_coverage"]
        for item in answerable
        if item["expected_term_coverage"] is not None
    ]

    return {
        "total_cases": total,
        "answerability_accuracy": sum(item["answerability_correct"] for item in results)
        / total,
        "answerable_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
        "answer_rate_on_answerable": (
            sum(item["answered"] for item in answerable) / len(answerable)
            if answerable
            else None
        ),
        "abstention_recall_on_unanswerable": (
            sum(item["abstained"] for item in unanswerable) / len(unanswerable)
            if unanswerable
            else None
        ),
        "citation_grounding_rate": (
            sum(
                item["citation_sources_are_grounded"] is True for item in citation_cases
            )
            / len(citation_cases)
            if citation_cases
            else None
        ),
        "expected_term_coverage": (
            sum(coverage_values) / len(coverage_values) if coverage_values else None
        ),
    }


def evaluate_answers() -> dict[str, Any]:
    """Evaluate answerability, abstention, citations, and term coverage."""
    cases = load_cases()
    retriever = HybridRetriever()
    results: list[dict[str, Any]] = []

    for case in cases:
        response = answer_with_evidence(
            query=case["question"],
            retriever=retriever,
            top_k=3,
        )

        expected_answerable = case.get(
            "answerable",
            case.get("expected_source") is not None,
        )
        answered = response["status"] == "evidence_found"
        abstained = response["status"] == "insufficient_evidence"

        answer = str(response.get("answer", ""))
        evidence = response.get("evidence", [])

        evidence_sources = {
            str(item["source"]) for item in evidence if item.get("source")
        }
        cited_sources = set(CITATION_PATTERN.findall(answer))
        expected_source = case.get("expected_source")

        expected_terms = {str(term).lower() for term in case.get("expected_terms", [])}
        answer_terms = normalize_terms(answer)
        matched_terms = sorted(expected_terms & answer_terms)

        citation_grounding_applicable = bool(cited_sources)
        citation_sources_are_grounded = (
            cited_sources.issubset(evidence_sources)
            if citation_grounding_applicable
            else None
        )

        expected_source_cited = (
            expected_source is not None and expected_source in cited_sources
        )

        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "expected_answerable": expected_answerable,
                "status": response["status"],
                "answered": answered,
                "abstained": abstained,
                "answerability_correct": (
                    answered if expected_answerable else abstained
                ),
                "expected_source": expected_source,
                "cited_sources": sorted(cited_sources),
                "evidence_sources": sorted(evidence_sources),
                "citation_grounding_applicable": (citation_grounding_applicable),
                "citation_sources_are_grounded": (citation_sources_are_grounded),
                "expected_source_cited": expected_source_cited,
                "expected_terms": sorted(expected_terms),
                "matched_expected_terms": matched_terms,
                "expected_term_coverage": (
                    len(matched_terms) / len(expected_terms) if expected_terms else None
                ),
                "answer": answer,
                "abstention_reason": response.get("reason"),
            }
        )

    answerable = [result for result in results if result["expected_answerable"]]
    unanswerable = [result for result in results if not result["expected_answerable"]]

    answerable_count = len(answerable)
    unanswerable_count = len(unanswerable)
    total_count = len(results)

    term_coverage_results = [
        result["expected_term_coverage"]
        for result in answerable
        if result["expected_term_coverage"] is not None
    ]

    grounded_results = [
        result["citation_sources_are_grounded"]
        for result in results
        if result["citation_grounding_applicable"]
    ]

    report: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_size": total_count,
        "answerable_questions": answerable_count,
        "unanswerable_questions": unanswerable_count,
        "answerability_accuracy": (
            sum(result["answerability_correct"] for result in results) / total_count
            if total_count
            else 0.0
        ),
        "answer_rate_on_answerable": (
            sum(result["answered"] for result in answerable) / answerable_count
            if answerable_count
            else 0.0
        ),
        "abstention_recall_on_unanswerable": (
            sum(result["abstained"] for result in unanswerable) / unanswerable_count
            if unanswerable_count
            else 0.0
        ),
        "expected_source_citation_rate": (
            sum(result["expected_source_cited"] for result in answerable)
            / answerable_count
            if answerable_count
            else 0.0
        ),
        "citation_presence_rate_on_answerable": (
            sum(bool(result["cited_sources"]) for result in answerable)
            / answerable_count
            if answerable_count
            else 0.0
        ),
        "grounded_citation_rate": (
            sum(bool(value) for value in grounded_results) / len(grounded_results)
            if grounded_results
            else 0.0
        ),
        "grounded_citation_cases": len(grounded_results),
        "mean_expected_term_coverage": (
            sum(term_coverage_results) / len(term_coverage_results)
            if term_coverage_results
            else 0.0
        ),
        "results": results,
        "limitations": [
            "The dataset is small and hand-authored.",
            "Lexical term coverage is not semantic answer correctness.",
            "A citation matching retrieved evidence does not prove entailment.",
            "Grounded citation rate only includes answers with parsed citations.",
            "Answerability labels are evaluation assumptions, not universal truth.",
            "Metrics should be reviewed per case before changing thresholds.",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return report


if __name__ == "__main__":
    print(json.dumps(evaluate_answers(), indent=2, ensure_ascii=False))
