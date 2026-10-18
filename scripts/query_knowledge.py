from __future__ import annotations

import argparse
import json

from insightforge.rag.answer import answer_with_evidence
from insightforge.rag.retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search the InsightForge knowledge base."
    )
    parser.add_argument("query", help="Question or search query")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    retriever = HybridRetriever()
    result = answer_with_evidence(
        args.query,
        retriever,
        top_k=args.top_k,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
