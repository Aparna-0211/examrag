"""Glue between uploads and the vector store."""
from __future__ import annotations

from pathlib import Path

from .config import Settings
from .embeddings import Embedder
from .ingest import ingest_pdf
from .vectorstore import VectorStore


def index_pdf(
    pdf_path: str | Path,
    store: VectorStore,
    embedder: Embedder,
    settings: Settings,
) -> int:
    """Ingest one PDF into the store (replacing any earlier copy). Returns chunk count."""
    chunks = ingest_pdf(pdf_path, settings.chunk_size, settings.chunk_overlap)
    store.delete_source(Path(pdf_path).name)
    vectors = embedder.embed([c.text for c in chunks])
    return store.add_chunks(chunks, vectors)
