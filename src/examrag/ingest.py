"""PDF ingestion: extract text per page, tag units, and split into chunks.

Every chunk keeps its source file, page number and (when detected) the
syllabus unit it belongs to, so answers can cite "Unit 3, page 47".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# "UNIT III", "Unit 3: Memory", "Chapter 2 - Sorting", "Module 4"
UNIT_RE = re.compile(
    r"^\s*(?P<kind>unit|chapter|module)\s+(?P<num>[ivxlc]+|\d+)\b[\s:.\-–—]*(?P<title>.*)$",
    re.IGNORECASE | re.MULTILINE,
)

# Sentence/line boundaries, including the Devanagari danda used in Hindi.
_BOUNDARIES = (". ", "? ", "! ", "।", "\n\n", "\n")


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    unit: str = ""
    chunk_id: str = ""
    extra: dict = field(default_factory=dict)

    def metadata(self) -> dict:
        return {"source": self.source, "page": self.page, "unit": self.unit}


def detect_unit(page_text: str, current: str = "") -> str:
    """Return the unit label found on this page, or keep the current one."""
    match = UNIT_RE.search(page_text)
    if not match:
        return current
    kind = match.group("kind").capitalize()
    num = match.group("num").upper()
    title = match.group("title").strip()
    label = f"{kind} {num}"
    if title:
        label += f": {title[:60]}"
    return label


def clean_text(text: str) -> str:
    """Normalise whitespace without destroying paragraph breaks."""
    text = text.replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _last_boundary(window: str) -> int:
    """Index just after the last sentence/paragraph boundary in window, or -1."""
    best = -1
    for b in _BOUNDARIES:
        pos = window.rfind(b)
        if pos != -1:
            best = max(best, pos + len(b))
    return best


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Split text into ~size-character chunks, preferring sentence boundaries."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    text = clean_text(text)
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = _last_boundary(window)
            # only snap to a boundary if it keeps most of the chunk
            if cut > size * 0.5:
                end = start + cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_pages(pdf_path: str | Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] with 1-based page numbers."""
    from pypdf import PdfReader  # lazy import keeps unit tests light

    reader = PdfReader(str(pdf_path))
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # a single bad page should not kill the upload
            text = ""
        pages.append((idx, text))
    return pages


def chunks_from_pages(
    pages: list[tuple[int, str]],
    source: str,
    size: int = 900,
    overlap: int = 150,
) -> list[Chunk]:
    """Turn (page, text) pairs into tagged chunks."""
    chunks: list[Chunk] = []
    unit = ""
    for page_no, raw in pages:
        unit = detect_unit(raw, unit)
        for n, piece in enumerate(chunk_text(raw, size, overlap)):
            chunks.append(
                Chunk(
                    text=piece,
                    source=source,
                    page=page_no,
                    unit=unit,
                    chunk_id=f"{source}::p{page_no}::c{n}",
                )
            )
    return chunks


def ingest_pdf(
    pdf_path: str | Path, size: int = 900, overlap: int = 150
) -> list[Chunk]:
    """Full pipeline for one PDF."""
    path = Path(pdf_path)
    pages = extract_pages(path)
    chunks = chunks_from_pages(pages, source=path.name, size=size, overlap=overlap)
    if not chunks:
        raise ValueError(
            f"No extractable text in {path.name}. It may be a scanned PDF; "
            "run OCR first (see README, 'Known limitations')."
        )
    return chunks
