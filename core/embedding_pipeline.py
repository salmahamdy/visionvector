"""Batch image embedding pipeline using OpenCLIP via HuggingFace transformers."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Generator

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class EmbeddingPipeline:
    """Encodes images and text into normalized embeddings using CLIP."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        logger.info("Loading model %s on %s", model_name, self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Encode a list of text strings into L2-normalised embeddings."""
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs).float()
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    def encode_image(self, images: list[Image.Image]) -> np.ndarray:
        """Encode a list of PIL images into L2-normalised embeddings."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs).float()
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    def _iter_image_paths(self, data_dir: Path) -> Generator[Path, None, None]:
        for path in sorted(data_dir.rglob("*")):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path

    def run(
        self,
        data_dir: str | Path,
        embeddings_dir: str | Path,
        metadata_path: str | Path,
    ) -> None:
        """Batch-encode all images under data_dir and persist results.

        Args:
            data_dir: Root directory containing images.
            embeddings_dir: Directory to save .npy embedding shards.
            metadata_path: Path to save metadata JSON.
        """
        data_dir = Path(data_dir)
        embeddings_dir = Path(embeddings_dir)
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list(self._iter_image_paths(data_dir))
        if not image_paths:
            raise FileNotFoundError(f"No images found under {data_dir}")

        logger.info("Found %d images. Starting encoding…", len(image_paths))

        all_embeddings: list[np.ndarray] = []
        metadata: list[dict] = []

        for batch_start in tqdm(
            range(0, len(image_paths), self.batch_size), desc="Encoding batches"
        ):
            batch_paths = image_paths[batch_start : batch_start + self.batch_size]
            images: list[Image.Image] = []
            valid_paths: list[Path] = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                    valid_paths.append(p)
                except Exception as exc:
                    logger.warning("Skipping %s: %s", p, exc)

            if not images:
                continue

            embeddings = self.encode_image(images)
            all_embeddings.append(embeddings)

            for idx, p in enumerate(valid_paths):
                relative = p.relative_to(data_dir)
                parts = relative.parts
                metadata.append(
                    {
                        "id": batch_start + idx,
                        "path": str(p),
                        "filename": p.name,
                        "category": parts[0] if len(parts) > 1 else "uncategorized",
                        "tags": list(parts[:-1]),
                    }
                )

        combined = np.vstack(all_embeddings).astype(np.float32)
        shard_path = embeddings_dir / "image_embeddings.npy"
        np.save(shard_path, combined)
        logger.info("Saved embeddings: %s  shape=%s", shard_path, combined.shape)

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved metadata: %s", metadata_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Encode images into CLIP embeddings")
    parser.add_argument("--data_dir", default="data/images")
    parser.add_argument("--embeddings_dir", default="embeddings")
    parser.add_argument("--metadata_path", default="embeddings/metadata.json")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    pipeline = EmbeddingPipeline(model_name=args.model, batch_size=args.batch_size)
    pipeline.run(args.data_dir, args.embeddings_dir, args.metadata_path)


if __name__ == "__main__":
    main()
