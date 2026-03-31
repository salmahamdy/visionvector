"""Hybrid retrieval: vector similarity (Faiss) + metadata filtering + re-ranking."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    id: int
    score: float
    path: str
    filename: str
    category: str
    tags: list[str]
    rerank_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "score": float(self.score),
            "rerank_score": float(self.rerank_score) if self.rerank_score is not None else None,
            "path": self.path,
            "filename": self.filename,
            "category": self.category,
            "tags": self.tags,
        }


@dataclass
class MetadataFilter:
    categories: list[str] | None = None
    tags: list[str] | None = None
    filenames: list[str] | None = None

    def matches(self, meta: dict) -> bool:
        if self.categories and meta.get("category") not in self.categories:
            return False
        if self.tags and not any(t in meta.get("tags", []) for t in self.tags):
            return False
        if self.filenames and meta.get("filename") not in self.filenames:
            return False
        return True


class HybridRetriever:
    """Combines Faiss ANN search with metadata filtering and cosine re-ranking."""

    def __init__(
        self,
        index_dir: str | Path = "index",
        metadata_path: str | Path = "embeddings/metadata.json",
        embeddings_path: str | Path = "embeddings/image_embeddings.npy",
        use_ivf: bool = True,
    ) -> None:
        self.index_dir = Path(index_dir)
        index_name = "ivf" if use_ivf else "flat"
        index_path = self.index_dir / f"{index_name}.faiss"

        # Graceful fallback: if IVF index not found try flat
        if not index_path.exists():
            logger.warning("%s not found. Falling back to flat index.", index_path)
            index_path = self.index_dir / "flat.faiss"

        self.index: faiss.Index = faiss.read_index(str(index_path))
        logger.info("Loaded index: %s  ntotal=%d", index_path, self.index.ntotal)

        with open(metadata_path) as f:
            self._metadata: list[dict] = json.load(f)

        self._embeddings: np.ndarray = np.load(embeddings_path).astype(np.float32)

    # ------------------------------------------------------------------
    # Core search methods
    # ------------------------------------------------------------------

    def _vector_search(
        self, query_embedding: np.ndarray, top_k: int
    ) -> list[tuple[int, float]]:
        """Return (index_id, score) pairs from Faiss."""
        q = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(q, top_k)
        return [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx != -1
        ]

    def _apply_metadata_filter(
        self, hits: list[tuple[int, float]], meta_filter: MetadataFilter | None
    ) -> list[tuple[int, float]]:
        if meta_filter is None:
            return hits
        return [
            (idx, score)
            for idx, score in hits
            if meta_filter.matches(self._metadata[idx])
        ]

    def _rerank(
        self, hits: list[tuple[int, float]], query_embedding: np.ndarray
    ) -> list[SearchResult]:
        """Re-rank using exact cosine similarity against stored embeddings."""
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        results: list[SearchResult] = []
        for idx, score in hits:
            stored = self._embeddings[idx]
            rerank_score = float(np.dot(q, stored))
            meta = self._metadata[idx]
            results.append(
                SearchResult(
                    id=idx,
                    score=score,
                    rerank_score=rerank_score,
                    path=meta["path"],
                    filename=meta["filename"],
                    category=meta["category"],
                    tags=meta["tags"],
                )
            )
        results.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        meta_filter: MetadataFilter | None = None,
        rerank: bool = True,
        fetch_k_multiplier: int = 4,
    ) -> list[SearchResult]:
        """Run hybrid search.

        Args:
            query_embedding: L2-normalised float32 vector (1-D or 2-D row).
            top_k: Number of final results to return.
            meta_filter: Optional metadata filter applied post-retrieval.
            rerank: Whether to re-rank with exact cosine similarity.
            fetch_k_multiplier: Fetch this multiple of top_k from Faiss to
                allow metadata filter to reduce the candidate pool.
        """
        query_embedding = np.array(query_embedding, dtype=np.float32).flatten()
        fetch_k = top_k * fetch_k_multiplier

        hits = self._vector_search(query_embedding, fetch_k)
        hits = self._apply_metadata_filter(hits, meta_filter)

        if rerank:
            results = self._rerank(hits, query_embedding)
        else:
            results = [
                SearchResult(
                    id=idx,
                    score=score,
                    path=self._metadata[idx]["path"],
                    filename=self._metadata[idx]["filename"],
                    category=self._metadata[idx]["category"],
                    tags=self._metadata[idx]["tags"],
                )
                for idx, score in hits
            ]

        return results[:top_k]
