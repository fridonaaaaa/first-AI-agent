import json
import tempfile
from pathlib import Path

from app.ingestion.parsers import (
    ParsedDocument,
    parse_markdown,
    parse_text,
    parse_hotpotqa_record,
    parse_jsonl,
)


def _write(tmp: Path, name: str, content: str) -> Path:
    p = tmp / name
    p.write_text(content, encoding="utf-8")
    return p


def test_parse_markdown_extracts_title(tmp_path):
    p = _write(tmp_path, "doc.md", "# My Title\n\nSome text here.")
    doc = parse_markdown(p)
    assert doc.title == "My Title"
    assert "Some text here" in doc.text
    assert doc.format == "md"


def test_parse_markdown_strips_code_blocks(tmp_path):
    p = _write(tmp_path, "doc.md", "# Title\n\n```python\nprint('hi')\n```\n\nReal text.")
    doc = parse_markdown(p)
    assert "print" not in doc.text
    assert "Real text" in doc.text


def test_parse_text_uses_first_line_as_title(tmp_path):
    p = _write(tmp_path, "doc.txt", "First Line Title.\nSecond line content.")
    doc = parse_text(p)
    assert "First Line" in doc.title
    assert doc.format == "txt"


def test_parse_hotpotqa_record():
    record = {
        "question_id": "q1",
        "title": "Albert Einstein",
        "text": "Einstein was born in 1879.",
        "answer": "1879",
        "type": "bridge",
        "level": "easy",
    }
    doc = parse_hotpotqa_record(record)
    assert doc.title == "Albert Einstein"
    assert "1879" in doc.text
    assert doc.format == "jsonl"
    assert doc.source.startswith("hotpotqa/")
    assert doc.metadata["question_id"] == "q1"


def test_parse_jsonl_multiple_records(tmp_path):
    records = [
        {"question_id": "q1", "title": "Doc A", "text": "Text A.", "answer": "A", "type": "bridge", "level": "easy"},
        {"question_id": "q2", "title": "Doc B", "text": "Text B.", "answer": "B", "type": "comparison", "level": "hard"},
    ]
    p = tmp_path / "data.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records))
    docs = parse_jsonl(p)
    assert len(docs) == 2
    assert docs[0].title == "Doc A"
    assert docs[1].title == "Doc B"


def test_content_hash_is_normalisation_stable():
    record = {"question_id": "q1", "title": "T", "text": "Hello   World", "answer": "", "type": "", "level": ""}
    record2 = {"question_id": "q2", "title": "T", "text": "hello world", "answer": "", "type": "", "level": ""}
    doc1 = parse_hotpotqa_record(record)
    doc2 = parse_hotpotqa_record(record2)
    assert doc1.content_hash == doc2.content_hash
