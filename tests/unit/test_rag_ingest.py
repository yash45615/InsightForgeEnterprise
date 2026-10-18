import pytest

from insightforge.rag.ingest import chunk_document


def test_empty_document_returns_no_chunks():
    assert chunk_document("   ") == []


def test_short_document_returns_one_chunk():
    assert chunk_document("Campaign metrics guide.") == ["Campaign metrics guide."]


def test_chunks_respect_maximum_size():
    text = " ".join(["analytics"] * 250)
    chunks = chunk_document(text, chunk_size=100, overlap=20)

    assert chunks
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_chunks_have_expected_overlap():
    text = "abcdefghijklmnopqrstuvwxyz" * 4
    chunks = chunk_document(text, chunk_size=40, overlap=10)

    assert len(chunks) > 1
    assert chunks[0][-10:] == chunks[1][:10]


def test_invalid_chunk_settings_raise_error():
    with pytest.raises(ValueError):
        chunk_document("example", chunk_size=10, overlap=10)
