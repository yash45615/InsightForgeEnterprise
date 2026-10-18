from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
EVAL_PATH = ROOT / "tests" / "fixtures" / "rag_eval.json"
REPORT_PATH = ROOT / "reports" / "rag" / "quality_report.json"


def load_cases(path: Path = EVAL_PATH) -> list[dict[str, Any]]:
    """Load and validate the retrieval evaluation dataset."""
    cases = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation dataset must be a non-empty list.")

    required = {
        "id",
        "question",
        "expected_source",
        "expected_terms",
        "category",
    }

    for case in cases:
        if not isinstance(case, dict) or not required.issubset(case):
            raise ValueError("An evaluation case is missing required fields.")
        if not case["question"].strip():
            raise ValueError("Evaluation questions must not be empty.")
        if not isinstance(case["expected_terms"], list):
            raise ValueError("expected_terms must be a list.")

    return cases


def evaluate_retrieval(top_k: int = 3) -> dict[str, Any]:
    """Evaluate expected-source retrieval and unanswerable cases."""
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10.")

    from insightforge.rag.retriever import HybridRetriever

    cases = load_cases()
    retriever = HybridRetriever()
    results: list[dict[str, Any]] = []

    for case in cases:
        retrieved = retriever.search(case["question"], top_k=top_k)
        sources = [item["source"] for item in retrieved]
        expected = case.get("expected_source")
        answerable = case.get("answerable", expected is not None)

        rank = None
        if expected is not None:
            rank = next(
                (
                    index
                    for index, source in enumerate(sources, start=1)
                    if source == expected
                ),
                None,
            )

        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "answerable": answerable,
                "expected_source": expected,
                "retrieved_sources": sources,
                "expected_source_rank": rank,
                "hit": rank is not None,
            }
        )

    answerable_results = [result for result in results if result["answerable"]]
    unanswerable_results = [result for result in results if not result["answerable"]]

    answerable_total = len(answerable_results)
    reciprocal_rank = (
        sum(
            1 / result["expected_source_rank"]
            if result["expected_source_rank"] is not None
            else 0
            for result in answerable_results
        )
        / answerable_total
        if answerable_total
        else 0.0
    )

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_size": len(results),
        "top_k": top_k,
        "answerable_questions": answerable_total,
        "unanswerable_questions": len(unanswerable_results),
        "hit_rate_at_k": (
            sum(result["hit"] for result in answerable_results) / answerable_total
            if answerable_total
            else 0.0
        ),
        "mean_reciprocal_rank": reciprocal_rank,
        "unanswerable_cases": [
            {
                "id": result["id"],
                "retrieved_sources": result["retrieved_sources"],
                "retrieved_any_source": bool(result["retrieved_sources"]),
                "note": (
                    "Retrieval may return related documents even when "
                    "the requested answer is unavailable."
                ),
            }
            for result in unanswerable_results
        ],
        "results": results,
        "limitations": [
            "Small, hand-authored evaluation dataset.",
            "Retrieval of a source does not prove answer correctness.",
            "Unanswerable cases require answer-level abstention tests.",
            "Synthetic data and local documents do not establish "
            "production performance.",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return report


if __name__ == "__main__":
    print(json.dumps(evaluate_retrieval(), indent=2))
