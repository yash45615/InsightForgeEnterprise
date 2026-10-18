from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from insightforge.evaluation.release_report import build_release_report


def _write_json(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _retrieval_report() -> dict[str, Any]:
    return {
        "dataset_size": 2,
        "top_k": 3,
        "answerable_questions": 1,
        "unanswerable_questions": 1,
        "hit_rate_at_k": 1.0,
        "mean_reciprocal_rank": 1.0,
        "results": [{"id": "eval-1"}, {"id": "eval-2"}],
    }


def _answer_report() -> dict[str, Any]:
    return {
        "dataset_size": 2,
        "answerable_questions": 1,
        "unanswerable_questions": 1,
        "answerability_accuracy": 1.0,
        "answer_rate_on_answerable": 1.0,
        "abstention_recall_on_unanswerable": 1.0,
        "expected_source_citation_rate": 1.0,
        "citation_presence_rate_on_answerable": 1.0,
        "grounded_citation_rate": 1.0,
        "mean_expected_term_coverage": 1.0,
        "results": [{"id": "eval-1"}, {"id": "eval-2"}],
    }


def test_release_report_passes_when_all_thresholds_are_met(
    tmp_path: Path,
) -> None:
    retrieval_path = _write_json(
        tmp_path / "retrieval.json",
        _retrieval_report(),
    )
    answer_path = _write_json(
        tmp_path / "answers.json",
        _answer_report(),
    )

    report = build_release_report(retrieval_path, answer_path)

    assert report["status"] == "PASS"
    assert report["release_allowed"] is True
    assert report["failed_checks"] == []
    assert all(check["passed"] for check in report["checks"])


def test_release_report_fails_when_abstention_is_too_low(
    tmp_path: Path,
) -> None:
    answers = _answer_report()
    answers["abstention_recall_on_unanswerable"] = 0.5

    retrieval_path = _write_json(
        tmp_path / "retrieval.json",
        _retrieval_report(),
    )
    answer_path = _write_json(tmp_path / "answers.json", answers)

    report = build_release_report(retrieval_path, answer_path)

    assert report["status"] == "FAIL"
    assert report["release_allowed"] is False
    assert "abstention_recall_on_unanswerable" in report["failed_checks"]


def test_release_report_rejects_inconsistent_dataset_sizes(
    tmp_path: Path,
) -> None:
    retrieval = _retrieval_report()
    retrieval["dataset_size"] = 3
    retrieval["answerable_questions"] = 2
    retrieval["results"].append({"id": "eval-3"})

    retrieval_path = _write_json(tmp_path / "retrieval.json", retrieval)
    answer_path = _write_json(
        tmp_path / "answers.json",
        _answer_report(),
    )

    with pytest.raises(ValueError, match="different dataset sizes"):
        build_release_report(retrieval_path, answer_path)


def test_release_report_rejects_invalid_metric_values(
    tmp_path: Path,
) -> None:
    retrieval = _retrieval_report()
    retrieval["hit_rate_at_k"] = 1.5

    retrieval_path = _write_json(tmp_path / "retrieval.json", retrieval)
    answer_path = _write_json(
        tmp_path / "answers.json",
        _answer_report(),
    )

    with pytest.raises(ValueError, match="hit_rate_at_k"):
        build_release_report(retrieval_path, answer_path)


def test_release_report_rejects_missing_input_file(
    tmp_path: Path,
) -> None:
    answer_path = _write_json(
        tmp_path / "answers.json",
        _answer_report(),
    )

    with pytest.raises(ValueError, match="Required report is missing"):
        build_release_report(tmp_path / "missing.json", answer_path)
