from insightforge.evaluation.answer_evaluator import summarize_results


def test_summary_handles_empty_results() -> None:
    summary = summarize_results([])

    assert summary["total_cases"] == 0
    assert summary["citation_grounding_rate"] is None


def test_summary_calculates_answerability_accuracy() -> None:
    results = [
        {
            "expected_answerable": True,
            "answerability_correct": True,
            "answered": True,
            "abstained": False,
            "citation_grounding_applicable": True,
            "citation_sources_are_grounded": True,
            "expected_term_coverage": 1.0,
        },
        {
            "expected_answerable": False,
            "answerability_correct": True,
            "answered": False,
            "abstained": True,
            "citation_grounding_applicable": False,
            "citation_sources_are_grounded": None,
            "expected_term_coverage": None,
        },
    ]

    summary = summarize_results(results)

    assert summary["total_cases"] == 2
    assert summary["answerability_accuracy"] == 1.0
    assert summary["answer_rate_on_answerable"] == 1.0
    assert summary["abstention_recall_on_unanswerable"] == 1.0
    assert summary["citation_grounding_rate"] == 1.0
    assert summary["expected_term_coverage"] == 1.0
