from __future__ import annotations

import json

from insightforge.evaluation.rag_evaluator import evaluate_retrieval


def main() -> None:
    report = evaluate_retrieval(top_k=3)

    legacy_report = {
        "evaluation_cases": report["dataset_size"],
        "answerable_questions": report["answerable_questions"],
        "unanswerable_questions": report["unanswerable_questions"],
        "hit_rate_at_3": report["hit_rate_at_k"],
        "mean_reciprocal_rank": report["mean_reciprocal_rank"],
        "note": (
            "Small, hand-authored evaluation set. "
            "Retrieval metrics do not establish answer correctness."
        ),
        "details": report["results"],
    }

    print(json.dumps(legacy_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
