from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RELEASE_THRESHOLDS: dict[str, float] = {
    "hit_rate_at_k": 0.90,
    "mean_reciprocal_rank": 0.90,
    "answerability_accuracy": 0.80,
    "abstention_recall_on_unanswerable": 0.90,
    "expected_source_citation_rate": 0.80,
    "mean_expected_term_coverage": 0.50,
}

LIMITATIONS = [
    "Evaluation uses a small, hand-authored dataset.",
    "Retrieval quality does not establish answer correctness.",
    "Lexical expected-term coverage is not semantic correctness.",
    "Citation grounding does not prove that cited sources entail every claim.",
    "Release thresholds are configurable engineering policies, not industry standards.",
    "Passing this gate does not establish production readiness.",
]


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Required report is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Report contains invalid JSON: {path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Report must contain a JSON object: {path}")

    return data


def _require_number(
    data: dict[str, Any],
    key: str,
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """Read a finite numeric metric within its expected range."""
    value = data.get(key)

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Metric '{key}' must be numeric.")

    number = float(value)

    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"Metric '{key}' must be between {minimum} and {maximum}.")

    return number


def _validate_dataset(
    report: dict[str, Any],
    *,
    report_name: str,
) -> None:
    """Validate evaluation counts and per-case results."""
    dataset_size = report.get("dataset_size")
    answerable = report.get("answerable_questions")
    unanswerable = report.get("unanswerable_questions")
    results = report.get("results")

    for key, value in (
        ("dataset_size", dataset_size),
        ("answerable_questions", answerable),
        ("unanswerable_questions", unanswerable),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{report_name}: '{key}' must be a non-negative integer.")

    if dataset_size == 0:
        raise ValueError(f"{report_name}: dataset_size must be greater than zero.")

    if answerable + unanswerable != dataset_size:
        raise ValueError(
            f"{report_name}: answerable and unanswerable counts "
            "must sum to dataset_size."
        )

    if not isinstance(results, list) or len(results) != dataset_size:
        raise ValueError(f"{report_name}: results length must equal dataset_size.")

    if any(not isinstance(item, dict) for item in results):
        raise ValueError(f"{report_name}: every result must be a JSON object.")


def _check(
    name: str,
    actual: float,
    threshold: float,
) -> dict[str, Any]:
    """Create a deterministic release-threshold check."""
    return {
        "metric": name,
        "actual": round(actual, 6),
        "threshold": threshold,
        "passed": actual >= threshold,
    }


def build_release_report(
    retrieval_path: Path,
    answer_path: Path,
) -> dict[str, Any]:
    """Combine retrieval and answer reports into a release decision."""
    retrieval = _load_json(retrieval_path)
    answers = _load_json(answer_path)

    _validate_dataset(retrieval, report_name="retrieval report")
    _validate_dataset(answers, report_name="answer quality report")

    if retrieval["dataset_size"] != answers["dataset_size"]:
        raise ValueError("Retrieval and answer reports use different dataset sizes.")

    if (
        retrieval["answerable_questions"] != answers["answerable_questions"]
        or retrieval["unanswerable_questions"] != answers["unanswerable_questions"]
    ):
        raise ValueError(
            "Retrieval and answer reports have inconsistent answerability counts."
        )

    retrieval_metrics = {
        "hit_rate_at_k": _require_number(retrieval, "hit_rate_at_k"),
        "mean_reciprocal_rank": _require_number(retrieval, "mean_reciprocal_rank"),
    }

    answer_metrics = {
        "answerability_accuracy": _require_number(answers, "answerability_accuracy"),
        "answer_rate_on_answerable": _require_number(
            answers, "answer_rate_on_answerable"
        ),
        "abstention_recall_on_unanswerable": _require_number(
            answers, "abstention_recall_on_unanswerable"
        ),
        "expected_source_citation_rate": _require_number(
            answers, "expected_source_citation_rate"
        ),
        "citation_presence_rate_on_answerable": _require_number(
            answers, "citation_presence_rate_on_answerable"
        ),
        "grounded_citation_rate": _require_number(answers, "grounded_citation_rate"),
        "mean_expected_term_coverage": _require_number(
            answers, "mean_expected_term_coverage"
        ),
    }

    checks = [
        _check(name, value, RELEASE_THRESHOLDS[name])
        for name, value in {
            **retrieval_metrics,
            "answerability_accuracy": answer_metrics["answerability_accuracy"],
            "abstention_recall_on_unanswerable": (
                answer_metrics["abstention_recall_on_unanswerable"]
            ),
            "expected_source_citation_rate": (
                answer_metrics["expected_source_citation_rate"]
            ),
            "mean_expected_term_coverage": (
                answer_metrics["mean_expected_term_coverage"]
            ),
        }.items()
    ]

    failed_checks = [check["metric"] for check in checks if not check["passed"]]

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "PASS" if not failed_checks else "FAIL",
        "release_allowed": not failed_checks,
        "evaluation": {
            "dataset_size": retrieval["dataset_size"],
            "answerable_questions": retrieval["answerable_questions"],
            "unanswerable_questions": retrieval["unanswerable_questions"],
            "retrieval_top_k": retrieval.get("top_k"),
        },
        "retrieval_metrics": retrieval_metrics,
        "answer_metrics": answer_metrics,
        "thresholds": RELEASE_THRESHOLDS,
        "checks": checks,
        "failed_checks": failed_checks,
        "source_reports": {
            "retrieval": str(retrieval_path),
            "answer_quality": str(answer_path),
        },
        "limitations": LIMITATIONS,
    }


def generate_release_report(
    project_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Generate and save the unified release report."""
    root = project_root or Path(__file__).resolve().parents[3]

    retrieval_path = root / "reports" / "rag" / "quality_report.json"
    answer_path = root / "reports" / "rag" / "answer_quality_report.json"
    output_path = root / "reports" / "release" / "unified_evaluation_report.json"

    report = build_release_report(retrieval_path, answer_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return output_path, report
