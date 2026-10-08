"""Chroma-backed vector store with source/unit metadata filters."""
from __future__ import annotations

from dataclasses import dataclass

from .ingest import Chunk

COLLECTION = "examrag"


@dataclass
class Hit:
    text: str
    source: str
    page: int
    unit: str
    score: float  # cosine similarity, 0..1 (higher is better)


class VectorStore:
    def __init__(self, persist_dir: str, collection: str = COLLECTION):
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    # ---- writing -------------------------------------------------------
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        if not chunks:
            return 0
        self._col.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[c.metadata() for c in chunks],
        )
        return len(chunks)

    def delete_source(self, source: str) -> None:
        self._col.delete(where={"source": source})

    def clear(self) -> None:
        self._client.delete_collection(self._col.name)
        self._col = self._client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    # ---- reading -------------------------------------------------------
    def count(self) -> int:
        return self._col.count()

    def list_sources(self) -> list[str]:
        data = self._col.get(include=["metadatas"])
        return sorted({m["source"] for m in data["metadatas"]})

    def list_units(self, source: str | None = None) -> list[str]:
        where = {"source": source} if source else None
        data = self._col.get(where=where, include=["metadatas"])
        return sorted({m["unit"] for m in data["metadatas"] if m.get("unit")})

    def query(
        self,
        embedding: list[float],
        k: int = 5,
        source: str | None = None,
        unit: str | None = None,
    ) -> list[Hit]:
        if self.count() == 0:
            return []
        clauses = []
        if source:
            clauses.append({"source": source})
        if unit:
            clauses.append({"unit": unit})
        where = None
        if len(clauses) == 1:
            where = clauses[0]
        elif len(clauses) > 1:
            where = {"$and": clauses}

        res = self._col.query(
            query_embeddings=[embedding],
            n_results=min(k, self.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append(
                Hit(
                    text=doc,
                    source=meta["source"],
                    page=int(meta["page"]),
                    unit=meta.get("unit", ""),
                    score=max(0.0, min(1.0, 1.0 - float(dist))),
                )
            )
        return hits
