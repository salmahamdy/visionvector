"""Faiss index builder: supports IndexFlatIP (baseline) and IndexIVFFlat (optimized)."""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class IndexBuilder:
    """Builds, saves, and loads Faiss indexes for embedding search."""

    def __init__(self, index_dir: str | Path = "index") -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def build_flat(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """Build an exact IndexFlatIP index (inner product on normalized vectors = cosine)."""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        logger.info("IndexFlatIP built: %d vectors, dim=%d", index.ntotal, dim)
        return index

    def build_ivf(
        self,
        embeddings: np.ndarray,
        n_list: int | None = None,
        n_probe: int = 10,
    ) -> faiss.IndexIVFFlat:
        """Build an IndexIVFFlat index for approximate nearest-neighbour search.

        Args:
            embeddings: L2-normalised float32 array of shape (N, D).
            n_list: Number of Voronoi cells. Defaults to sqrt(N), clamped to [4, 4096].
            n_probe: Cells visited at query time (speed/accuracy tradeoff).
        """
        n, dim = embeddings.shape
        if n_list is None:
            n_list = max(4, min(4096, int(np.sqrt(n))))
        if n < n_list:
            logger.warning(
                "n_list=%d > n_vectors=%d; falling back to IndexFlatIP.", n_list, n
            )
            return self.build_flat(embeddings)  # type: ignore[return-value]

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, n_list, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = n_probe
        logger.info(
            "IndexIVFFlat built: %d vectors, dim=%d, n_list=%d, n_probe=%d",
            index.ntotal,
            dim,
            n_list,
            n_probe,
        )
        return index

    def save(self, index: faiss.Index, name: str) -> Path:
        path = self.index_dir / f"{name}.faiss"
        faiss.write_index(index, str(path))
        logger.info("Index saved: %s", path)
        return path

    def load(self, name: str) -> faiss.Index:
        path = self.index_dir / f"{name}.faiss"
        index = faiss.read_index(str(path))
        logger.info("Index loaded: %s  ntotal=%d", path, index.ntotal)
        return index

    def build_and_save_all(
        self,
        embeddings_path: str | Path,
        n_probe: int = 10,
    ) -> None:
        """Load embeddings and persist both flat and IVF indexes.

        Args:
            embeddings_path: Path to .npy file produced by EmbeddingPipeline.
            n_probe: nprobe for IVF index.
        """
        embeddings = np.load(embeddings_path).astype(np.float32)
        # Re-normalise to guard against any drift
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-9)

        flat = self.build_flat(embeddings)
        self.save(flat, "flat")

        ivf = self.build_ivf(embeddings, n_probe=n_probe)
        self.save(ivf, "ivf")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build Faiss indexes from embeddings")
    parser.add_argument("--embeddings_path", default="embeddings/image_embeddings.npy")
    parser.add_argument("--index_dir", default="index")
    parser.add_argument("--n_probe", type=int, default=10)
    args = parser.parse_args()

    builder = IndexBuilder(args.index_dir)
    builder.build_and_save_all(args.embeddings_path, n_probe=args.n_probe)


if __name__ == "__main__":
    main()
