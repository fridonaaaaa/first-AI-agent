from app.ingestion.chunkers import chunk_fixed_token, chunk_sentence_window, chunk_document
from app.ingestion.parsers import ParsedDocument


def _doc(text: str) -> ParsedDocument:
    return ParsedDocument(title="Test", text=text, source="test://doc", format="txt")


def test_fixed_token_basic():
    doc = _doc(" ".join(f"word{i}" for i in range(100)))
    chunks = chunk_fixed_token(doc, window=20, overlap=5)
    assert len(chunks) > 1
    for c in chunks:
        assert c.strategy == "fixed_token"
        assert len(c.text.split()) <= 20


def test_fixed_token_overlap_content():
    words = [f"w{i}" for i in range(30)]
    doc = _doc(" ".join(words))
    chunks = chunk_fixed_token(doc, window=10, overlap=3)
    # last words of chunk 0 should appear at start of chunk 1
    last_words_c0 = set(chunks[0].text.split()[-3:])
    first_words_c1 = set(chunks[1].text.split()[:3])
    assert last_words_c0 & first_words_c1


def test_fixed_token_short_doc():
    doc = _doc("short text")
    chunks = chunk_fixed_token(doc, window=256, overlap=64)
    assert len(chunks) == 1
    assert chunks[0].text == "short text"


def test_fixed_token_positions_sequential():
    doc = _doc(" ".join(f"w{i}" for i in range(60)))
    chunks = chunk_fixed_token(doc, window=20, overlap=5)
    positions = [c.position for c in chunks]
    assert positions == list(range(len(chunks)))


def test_sentence_window_basic():
    doc = _doc("First sentence. Second sentence. Third sentence. Fourth sentence.")
    chunks = chunk_sentence_window(doc, window=2, stride=1)
    assert len(chunks) > 1
    for c in chunks:
        assert c.strategy == "sentence_window"


def test_sentence_window_single_sentence():
    doc = _doc("Only one sentence here.")
    chunks = chunk_sentence_window(doc, window=3, stride=1)
    assert len(chunks) == 1


def test_sentence_window_metadata_carries_title():
    doc = _doc("Sentence one. Sentence two.")
    chunks = chunk_sentence_window(doc, window=2, stride=1)
    for c in chunks:
        assert c.metadata["title"] == "Test"
        assert c.metadata["source"] == "test://doc"


def test_chunk_document_dispatches():
    doc = _doc("Word one two three four five six seven eight nine ten.")
    assert chunk_document(doc, strategy="fixed_token")[0].strategy == "fixed_token"
    assert chunk_document(doc, strategy="sentence_window")[0].strategy == "sentence_window"


def test_chunk_document_unknown_strategy():
    import pytest
    doc = _doc("some text")
    with pytest.raises(ValueError):
        chunk_document(doc, strategy="unknown")
