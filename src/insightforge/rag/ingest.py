from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOCUMENT_DIR = ROOT / "data" / "documents"
ARTIFACT_DIR = ROOT / "artifacts" / "rag"

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


def _split_long_sentence(sentence: str, chunk_size: int) -> list[str]:
    """Split an unusually long sentence at word boundaries when possible."""
    if len(sentence) <= chunk_size:
        return [sentence]

    words = sentence.split()
    pieces: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join([*current, word])

        if current and len(candidate) > chunk_size:
            pieces.append(" ".join(current))
            current = []

        # Handle a single token longer than the configured limit.
        if len(word) > chunk_size:
            if current:
                pieces.append(" ".join(current))
                current = []

            pieces.extend(
                word[start : start + chunk_size]
                for start in range(0, len(word), chunk_size)
            )
        else:
            current.append(word)

    if current:
        pieces.append(" ".join(current))

    return pieces


def chunk_document(
    text: str,
    chunk_size: int = 700,
    overlap: int = 100,
) -> list[str]:
    """Create sentence-aware chunks with character overlap as a fallback."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size.")

    normalized = " ".join(text.split())

    if not normalized:
        return []

    # Preserve exact character overlap for unpunctuated text, such as
    # identifiers, generated strings, or text without sentence boundaries.
    sentence_parts = SENTENCE_PATTERN.split(normalized)
    if len(sentence_parts) == 1 and len(normalized) > chunk_size:
        step = chunk_size - overlap
        return [
            normalized[start : start + chunk_size]
            for start in range(0, len(normalized), step)
        ]

    sentences = [part.strip() for part in sentence_parts if part.strip()]

    units: list[str] = []
    for sentence in sentences:
        units.extend(_split_long_sentence(sentence, chunk_size))

    chunks: list[str] = []
    current: list[str] = []

    for unit in units:
        candidate = " ".join([*current, unit])

        if not current or len(candidate) <= chunk_size:
            current.append(unit)
            continue

        chunks.append(" ".join(current))

        suffix: list[str] = []
        for previous in reversed(current):
            proposed = " ".join([previous, *suffix])
            if len(proposed) > overlap:
                break
            suffix.insert(0, previous)

        if suffix and len(" ".join([*suffix, unit])) > chunk_size:
            suffix = []

        current = [*suffix, unit]

    if current:
        chunks.append(" ".join(current))

    return chunks


def build_chunks() -> list[dict[str, object]]:
    """Read local documents, create chunks, and persist their metadata."""
    if not DOCUMENT_DIR.exists():
        raise FileNotFoundError(f"Document directory not found: {DOCUMENT_DIR}")

    documents = sorted(
        [
            *DOCUMENT_DIR.glob("*.md"),
            *DOCUMENT_DIR.glob("*.txt"),
        ]
    )

    if not documents:
        raise ValueError(f"No Markdown or text documents found in {DOCUMENT_DIR}")

    records: list[dict[str, object]] = []

    for path in documents:
        content = path.read_text(encoding="utf-8")

        if not content.strip():
            print(f"Skipping empty document: {path.name}")
            continue

        chunks = chunk_document(content)

        for index, chunk in enumerate(chunks):
            chunk_id = hashlib.sha256(
                f"{path.name}:{index}:{chunk}".encode()
            ).hexdigest()[:16]

            records.append(
                {
                    "chunk_id": chunk_id,
                    "source": path.name,
                    "chunk_index": index,
                    "text": chunk,
                }
            )

    if not records:
        raise ValueError("No non-empty document chunks were generated.")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = ARTIFACT_DIR / "chunks.json"
    output_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Documents processed: {len(documents)}")
    print(f"Chunks created: {len(records)}")
    print(f"Saved chunks to: {output_path}")

    return records


if __name__ == "__main__":
    build_chunks()
