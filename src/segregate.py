from __future__ import annotations

import os
import shutil
from pathlib import Path


IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".heic",
    ".heif",
    ".avif",
}

VIDEO_EXTS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
    ".wmv",
    ".flv",
    ".mpeg",
    ".mpg",
    ".3gp",
    ".3g2",
    ".mts",
    ".m2ts",
}


def _copy_file(src_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / src_path.name
    if target.exists():
        stem = src_path.stem
        suffix = src_path.suffix
        i = 1
        while True:
            candidate = dest_dir / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                target = candidate
                break
            i += 1
    shutil.copy2(str(src_path), str(target))


def segregate_resources(resources_dir: Path, segregated_dir: Path) -> None:
    images_dir = segregated_dir / "images"
    videos_dir = segregated_dir / "videos"
    others = segregated_dir / "others"

    for root, _, files in os.walk(resources_dir):
        for name in files:
            ext = Path(name).suffix.lower()
            if ext in IMAGE_EXTS:
                _copy_file(Path(root) / name, images_dir)
            elif ext in VIDEO_EXTS:
                _copy_file(Path(root) / name, videos_dir)
            else: _copy_file(Path(root) / name, others)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    resources_dir = root / "resources"
    segregated_dir = root / "segregated"
    if not resources_dir.exists():
        raise SystemExit(f"resources folder not found: {resources_dir}")
    segregate_resources(resources_dir, segregated_dir)


if __name__ == "__main__":
    main()
