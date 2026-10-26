"""
Parsers for four file formats: markdown/text, HTML, PDF, JSONL.
Each parser returns a ParsedDocument with title, text, metadata, source, format.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedDocument:
    title: str
    text: str
    source: str          # URL or dataset ID
    format: str          # md | txt | html | pdf | jsonl
    language: str = "en"
    metadata: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        normalised = " ".join(self.text.lower().split())
        return hashlib.sha256(normalised.encode()).hexdigest()


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_markdown(path: Path, source: str = "") -> ParsedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    title_match = re.search(r"^#\s+(.+)", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else path.stem
    clean = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    clean = re.sub(r"`[^`]+`", " ", clean)
    clean = re.sub(r"!\[.*?\]\(.*?\)", " ", clean)
    clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)
    clean = re.sub(r"#{1,6}\s+", " ", clean)
    clean = _normalise_whitespace(clean)
    return ParsedDocument(
        title=title,
        text=clean,
        source=source or str(path),
        format="md",
        metadata={"filename": path.name},
    )


def parse_text(path: Path, source: str = "") -> ParsedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = lines[0][:120] if lines else path.stem
    return ParsedDocument(
        title=title,
        text=_normalise_whitespace(text),
        source=source or str(path),
        format="txt",
        metadata={"filename": path.name},
    )


def parse_html(content: str | bytes, source: str = "", title_hint: str = "") -> ParsedDocument:
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError as exc:
        raise ImportError("beautifulsoup4 is required for HTML parsing: pip install beautifulsoup4") from exc

    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")

    soup = BeautifulSoup(content, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    title = title_hint
    if not title:
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            title = og["content"]
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)
        else:
            title = source or "untitled"

    text = _normalise_whitespace(soup.get_text(separator=" "))
    return ParsedDocument(
        title=title[:200],
        text=text,
        source=source,
        format="html",
        metadata={},
    )


def parse_pdf(path: Path, source: str = "") -> ParsedDocument:
    try:
        import pdfminer.high_level as pdfminer  # type: ignore
        import pdfminer.layout as layout  # type: ignore
    except ImportError as exc:
        raise ImportError("pdfminer.six is required for PDF parsing: pip install pdfminer.six") from exc

    pages = []
    with open(path, "rb") as f:
        for page_num, page_layout in enumerate(
            pdfminer.extract_pages(f, laparams=layout.LAParams()), start=1
        ):
            page_text = ""
            for element in page_layout:
                if hasattr(element, "get_text"):
                    page_text += element.get_text()
            pages.append((page_num, _normalise_whitespace(page_text)))

    full_text = " ".join(text for _, text in pages if text)
    first_line = full_text[:120].split(".")[0] if full_text else path.stem
    return ParsedDocument(
        title=first_line.strip(),
        text=full_text,
        source=source or str(path),
        format="pdf",
        metadata={"filename": path.name, "page_count": len(pages)},
    )


def parse_hotpotqa_record(record: dict) -> ParsedDocument:
    """Parse one HotpotQA context paragraph dict into a ParsedDocument."""
    title = record["title"]
    text = _normalise_whitespace(record["text"])
    source = f"hotpotqa/{record.get('question_id', 'unknown')}/{title}"
    return ParsedDocument(
        title=title,
        text=text,
        source=source,
        format="jsonl",
        metadata={
            "question_id": record.get("question_id", ""),
            "answer": record.get("answer", ""),
            "type": record.get("type", ""),
            "level": record.get("level", ""),
        },
    )


def parse_jsonl(path: Path, source: str = "") -> list[ParsedDocument]:
    """Parse a JSONL file where each line is a HotpotQA-style paragraph record."""
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            docs.append(parse_hotpotqa_record(record))
    return docs
