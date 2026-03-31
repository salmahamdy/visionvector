"""Download sample images for testing using reliable public sources."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

# Browser User-Agent required by most CDNs / Wikipedia
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# (category_subdir, filename, url)
# Sources: Unsplash source API (free, no auth key needed)
SAMPLE_IMAGES: list[tuple[str, str, str]] = [
    # Animals
    ("animals/dog",      "dog_1.jpg",      "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400&q=80"),
    ("animals/dog",      "dog_2.jpg",      "https://images.unsplash.com/photo-1552053831-71594a27632d?w=400&q=80"),
    ("animals/cat",      "cat_1.jpg",      "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400&q=80"),
    ("animals/cat",      "cat_2.jpg",      "https://images.unsplash.com/photo-1495360010541-f48722b34f7d?w=400&q=80"),
    ("animals/bird",     "bird_1.jpg",     "https://images.unsplash.com/photo-1444464666168-49d633b86797?w=400&q=80"),
    ("animals/bird",     "bird_2.jpg",     "https://images.unsplash.com/photo-1452570053594-1b985d6ea890?w=400&q=80"),
    # Vehicles
    ("vehicles/car",     "car_1.jpg",      "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=400&q=80"),
    ("vehicles/car",     "car_2.jpg",      "https://images.unsplash.com/photo-1502877338535-766e1452684a?w=400&q=80"),
    ("vehicles/bicycle", "bicycle_1.jpg",  "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400&q=80"),
    ("vehicles/bicycle", "bicycle_2.jpg",  "https://images.unsplash.com/photo-1541625602330-2277a4c46182?w=400&q=80"),
    # Nature
    ("nature/forest",    "forest_1.jpg",   "https://images.unsplash.com/photo-1448375240586-882707db888b?w=400&q=80"),
    ("nature/forest",    "forest_2.jpg",   "https://images.unsplash.com/photo-1542273917363-3b1817f69a2d?w=400&q=80"),
    ("nature/beach",     "beach_1.jpg",    "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=400&q=80"),
    ("nature/beach",     "beach_2.jpg",    "https://images.unsplash.com/photo-1519046904884-53103b34b206?w=400&q=80"),
    ("nature/mountain",  "mountain_1.jpg", "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=400&q=80"),
    # Food
    ("food/pizza",       "pizza_1.jpg",    "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=400&q=80"),
    ("food/pizza",       "pizza_2.jpg",    "https://images.unsplash.com/photo-1574071318508-1cdbab80d002?w=400&q=80"),
    ("food/burger",      "burger_1.jpg",   "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400&q=80"),
    ("food/burger",      "burger_2.jpg",   "https://images.unsplash.com/photo-1550547660-d9450f859349?w=400&q=80"),
    # People
    ("people/portrait",  "portrait_1.jpg", "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&q=80"),
    ("people/portrait",  "portrait_2.jpg", "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=400&q=80"),
]


def download_samples(output_dir: str = "data/images") -> None:
    base = Path(output_dir)
    ok = 0
    failed = 0

    for subdir, filename, url in SAMPLE_IMAGES:
        folder = base / subdir
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / filename

        if dest.exists():
            print(f"  ✓ exists  {dest}")
            ok += 1
            continue

        print(f"  ↓ {dest}", end=" ", flush=True)
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                dest.write_bytes(resp.read())
            print("✓")
            ok += 1
        except Exception as exc:
            print(f"✗  ({exc})")
            failed += 1

    print(f"\n{'─' * 50}")
    print(f"Downloaded: {ok}  Failed: {failed}  Total: {ok + failed}")

    if ok > 0:
        print("\nNext steps:")
        print("  python -m core.embedding_pipeline --data_dir data/images")
        print("  python -m core.indexing --embeddings_path embeddings/image_embeddings.npy")
    else:
        print("\n⚠  All downloads failed. Check your internet connection.")
        print("   Alternatively, place your own .jpg/.png images under data/images/<category>/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download sample images for multimodal search")
    parser.add_argument("--output_dir", default="data/images")
    args = parser.parse_args()
    download_samples(args.output_dir)
