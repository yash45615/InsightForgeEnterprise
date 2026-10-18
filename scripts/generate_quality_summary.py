from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "reports" / "rag" / "quality_report.json"
OUTPUT_PATH = ROOT / "reports" / "rag" / "quality_summary.md"


def load_report() -> dict[str, Any]:
    """Load the latest RAG quality report."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation report not found: {INPUT_PATH}. Run the RAG evaluator first."
        )

    report = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    required = {
        "created_at",
        "dataset_size",
        "top_k",
        "hit_rate_at_k",
        "mean_reciprocal_rank",
        "results",
        "limitations",
    }

    missing = required.difference(report)
    if missing:
        raise ValueError(f"Evaluation report is missing fields: {sorted(missing)}")

    return report


def build_markdown(report: dict[str, Any]) -> str:
    """Format retrieval evaluation results as a Markdown report."""
    hit_rate = float(report["hit_rate_at_k"])
    mrr = float(report["mean_reciprocal_rank"])

    lines = [
        "# InsightForge Enterprise - RAG Quality Report",
        "",
        f"- **Generated at:** {report['created_at']}",
        f"- **Evaluation questions:** {report['dataset_size']}",
        f"- **Retrieval cutoff:** top {report['top_k']}",
        "",
        "## Summary metrics",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Hit rate@{report['top_k']} | {hit_rate:.2%} |",
        f"| Mean Reciprocal Rank (MRR) | {mrr:.4f} |",
        "",
        "## Per-question results",
        "",
        "| ID | Category | Expected source | Rank | Retrieved? |",
        "|---|---|---|---:|---|",
    ]

    for result in report["results"]:
        rank = result["expected_source_rank"]
        rank_text = str(rank) if rank is not None else "Not found"
        retrieved = "Yes" if result["hit"] else "No"

        lines.append(
            f"| {result['id']} "
            f"| {result['category']} "
            f"| `{result['expected_source']}` "
            f"| {rank_text} "
            f"| {retrieved} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Hit rate measures whether the expected source appears "
            "within the selected top-k results.",
            "- MRR rewards expected sources that appear nearer the top.",
            "- These metrics evaluate retrieval, not generated-answer "
            "factuality or business impact.",
            "",
            "## Limitations",
            "",
        ]
    )

    lines.extend(f"- {item}" for item in report["limitations"])

    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```powershell",
            "python -m insightforge.rag.build_index",
            "python -m insightforge.evaluation.rag_evaluator",
            "python scripts/generate_quality_summary.py",
            "```",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    report = load_report()
    markdown = build_markdown(report)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(markdown, encoding="utf-8")

    print(f"Quality summary saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
