from __future__ import annotations

import os
import shutil
from pathlib import Path
from hashlib import blake2b


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


def _file_digest(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = blake2b()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def segregate_resources(resources_dir: Path, segregated_dir: Path) -> None:
    images_dir = segregated_dir / "images"
    videos_dir = segregated_dir / "videos"
    others = segregated_dir / "others"
    seen_sizes: dict[int, Path | set[str]] = {}

    for root, _, files in os.walk(resources_dir):
        for name in files:
            src_path = Path(root) / name
            size = src_path.stat().st_size
            entry = seen_sizes.get(size)
            if entry is None:
                seen_sizes[size] = src_path
            elif isinstance(entry, Path):
                existing_digest = _file_digest(entry)
                digest = _file_digest(src_path)
                if digest == existing_digest:
                    continue
                digests = {existing_digest, digest}
                seen_sizes[size] = digests
            else:
                digest = _file_digest(src_path)
                if digest in entry:
                    continue
                entry.add(digest)
            ext = Path(name).suffix.lower()
            if ext in IMAGE_EXTS:
                _copy_file(src_path, images_dir)
            elif ext in VIDEO_EXTS:
                _copy_file(src_path, videos_dir)
            else:
                _copy_file(src_path, others)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    resources_dir = root / "resources"
    segregated_dir = root / "segregated"
    if not resources_dir.exists():
        raise SystemExit(f"resources folder not found: {resources_dir}")
    segregate_resources(resources_dir, segregated_dir)


if __name__ == "__main__":
    main()
