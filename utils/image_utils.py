"""Image loading helpers."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image


def load_image(source: str | bytes | Path) -> Image.Image:
    """Load a PIL image from a file path, bytes, or base64 string."""
    if isinstance(source, (str, Path)) and Path(source).exists():
        return Image.open(source).convert("RGB")
    if isinstance(source, bytes):
        return Image.open(BytesIO(source)).convert("RGB")
    if isinstance(source, str):
        try:
            data = base64.b64decode(source)
            return Image.open(BytesIO(data)).convert("RGB")
        except Exception:
            pass
    raise ValueError(f"Cannot load image from: {type(source)}")


def image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    buf = BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()
