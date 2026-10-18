import json

import pytest

from insightforge.evaluation.rag_evaluator import load_cases


def test_evaluation_dataset_has_required_fields() -> None:
    cases = load_cases()

    assert len(cases) >= 4
    assert all(case["id"] for case in cases)
    assert all(
        isinstance(case["question"], str) and case["question"].strip() for case in cases
    )
    assert all(isinstance(case["expected_terms"], list) for case in cases)

    answerable_cases = [case for case in cases if case.get("answerable", True)]
    unanswerable_cases = [case for case in cases if not case.get("answerable", True)]

    assert all(case["expected_source"] for case in answerable_cases)
    assert all(case["expected_source"] is None for case in unanswerable_cases)


def test_evaluation_dataset_contains_distinct_ids() -> None:
    cases = load_cases()
    ids = [case["id"] for case in cases]

    assert len(ids) == len(set(ids))


def test_empty_evaluation_dataset_is_rejected(tmp_path) -> None:
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError):
        load_cases(path)


def test_invalid_evaluation_case_is_rejected(tmp_path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps([{"id": "eval-bad"}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_cases(path)
