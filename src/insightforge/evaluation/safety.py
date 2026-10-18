from __future__ import annotations

import re
from typing import Any

INJECTION_PATTERNS = [
    re.compile(
        r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(reveal|print|show|expose)\s+"
        r"(the\s+)?(system prompt|hidden instructions)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(jailbreak|bypass)\s+(the\s+)?"
        r"(safety|security|guardrails|restrictions)\b",
        re.IGNORECASE,
    ),
]

DESTRUCTIVE_SQL_PATTERNS = [
    re.compile(r"\bDROP\s+(TABLE|DATABASE|SCHEMA)\b", re.IGNORECASE),
    re.compile(r"\bTRUNCATE\s+TABLE\b", re.IGNORECASE),
    re.compile(r"\bDELETE\s+FROM\b", re.IGNORECASE),
    re.compile(r"\bUPDATE\s+\w+\s+SET\b", re.IGNORECASE),
    re.compile(r"\bINSERT\s+INTO\b", re.IGNORECASE),
    re.compile(r"\bEXEC(UTE)?\s*\(", re.IGNORECASE),
]


def inspect_user_input(text: str) -> dict[str, Any]:
    """Flag common injection attempts and potentially unsafe SQL requests."""
    if not text or not text.strip():
        return {
            "allowed": False,
            "flags": ["empty_input"],
            "risk_level": "high",
        }

    flags: list[str] = []

    if any(pattern.search(text) for pattern in INJECTION_PATTERNS):
        flags.append("possible_prompt_injection")

    if any(pattern.search(text) for pattern in DESTRUCTIVE_SQL_PATTERNS):
        flags.append("potentially_unsafe_sql")

    return {
        "allowed": not flags,
        "flags": flags,
        "risk_level": "high" if flags else "low",
    }


def validate_citations(
    answer: str,
    retrieved_sources: list[str],
) -> dict[str, Any]:
    """Check whether an answer cites at least one retrieved source."""
    citations = re.findall(
        r"\[([^\[\]]+\.md)(?:#[^\[\]]+)?\]",
        answer,
    )
    valid_citations = sorted(
        set(citation for citation in citations if citation in retrieved_sources)
    )
    invalid_citations = sorted(
        set(citation for citation in citations if citation not in retrieved_sources)
    )

    has_sources = bool(retrieved_sources)
    has_valid_citation = bool(valid_citations)

    return {
        "citation_present": bool(citations),
        "citation_valid": has_valid_citation,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "citation_coverage": (
            len(valid_citations) / len(set(retrieved_sources)) if has_sources else 0.0
        ),
    }
