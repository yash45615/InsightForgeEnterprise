from scripts.generate_quality_summary import build_markdown

from insightforge.evaluation.rag_evaluator import load_cases


def test_quality_summary_contains_key_sections() -> None:
    report = {
        "created_at": "2026-10-17T00:00:00+00:00",
        "dataset_size": 1,
        "top_k": 3,
        "hit_rate_at_k": 1.0,
        "mean_reciprocal_rank": 1.0,
        "results": [
            {
                "id": "eval-001",
                "category": "business_metrics",
                "expected_source": "campaign_metrics.md",
                "expected_source_rank": 1,
                "hit": True,
            }
        ],
        "limitations": ["Small evaluation dataset."],
    }

    markdown = build_markdown(report)

    assert "# InsightForge Enterprise" in markdown
    assert "Hit rate@3" in markdown
    assert "Mean Reciprocal Rank" in markdown
    assert "campaign_metrics.md" in markdown
    assert "Small evaluation dataset." in markdown


def test_evaluation_dataset_is_available() -> None:
    assert len(load_cases()) >= 4
