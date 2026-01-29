from __future__ import annotations

import os
import sys
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
    try:
        shutil.copy2(str(src_path), str(target))
    except PermissionError:
        return


def _file_digest(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    hasher = blake2b()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    files.append(Path(entry.path))
    return files


def segregate_resources(resources_dir: Path, segregated_dir: Path) -> None:
    images_dir = segregated_dir / "images"
    videos_dir = segregated_dir / "videos"
    others = segregated_dir / "others"
    seen_sizes: dict[int, Path | set[str]] = {}

    files = _iter_files(resources_dir)
    total = len(files)
    processed = 0
    progress_every = max(1, total // 200)

    for src_path in files:
        if src_path.name == ".gitkeep":
            continue
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
        ext = src_path.suffix.lower()
        if ext in IMAGE_EXTS:
            _copy_file(src_path, images_dir)
        elif ext in VIDEO_EXTS:
            _copy_file(src_path, videos_dir)
        else:
            _copy_file(src_path, others)
        processed += 1
        if processed % progress_every == 0 or processed == total:
            percent = (processed / total) * 100 if total else 100
            sys.stdout.write(f"\rProcessed {processed}/{total} files ({percent:5.1f}%)")
            sys.stdout.flush()
    if total:
        sys.stdout.write("\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    resources_dir = root / "resources"
    segregated_dir = root / "segregated"
    if not resources_dir.exists():
        raise SystemExit(f"resources folder not found: {resources_dir}")
    segregate_resources(resources_dir, segregated_dir)


if __name__ == "__main__":
    main()
