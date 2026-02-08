from pathlib import Path
import argparse
import os
import itertools
from concurrent.futures import ProcessPoolExecutor

import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(images_dir: Path):
    for path in images_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run face detection on all images in segregated/images."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=112,
        help="Output size for aligned faces (square).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1 if os.name == "nt" else max(1, min(4, (os.cpu_count() or 1) - 1)),
        help="Number of worker processes. Use 1 to disable multiprocessing.",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        default=640,
        help="Detection input size (square). Larger can improve accuracy but slower.",
    )
    parser.add_argument(
        "--max-input",
        type=int,
        default=2000,
        help="Max input dimension for detection. Larger images are downscaled.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="Chunk size for multiprocessing map.",
    )
    return parser.parse_args()


_APP = None
_ALIGN_SIZE = None


def _init_worker(det_size: int, align_size: int) -> None:
    global _APP, _ALIGN_SIZE
    _ALIGN_SIZE = align_size
    _APP = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    _APP.prepare(ctx_id=-1, det_size=(det_size, det_size))


def _process_image(
    image_path: Path,
    images_dir: Path,
    detected_dir: Path,
    max_input: int,
) -> tuple[int, int]:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Skipping unreadable image: {image_path}")
        return (1, 0)

    height, width = img.shape[:2]
    max_dim = max(height, width)
    if max_dim > max_input:
        scale = max_input / max_dim
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    faces = _APP.get(img)
    if not faces:
        return (1, 0)

    rel_stem = image_path.relative_to(images_dir).with_suffix("")
    safe_stem = "__".join(rel_stem.parts)
    saved = 0
    for idx, face in enumerate(faces, start=1):
        if face.kps is None:
            continue
        aligned = face_align.norm_crop(img, face.kps, image_size=_ALIGN_SIZE)
        output_name = f"{safe_stem}_face{idx}.jpg"
        output_path = detected_dir / output_name
        cv2.imwrite(str(output_path), aligned)
        saved += 1
    return (1, saved)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    images_dir = root / "segregated" / "images"
    detected_dir = root / "detected"
    if not images_dir.exists():
        raise SystemExit(f"segregated/images folder not found: {images_dir}")

    images = iter_images(images_dir)

    detected_dir.mkdir(parents=True, exist_ok=True)

    total_faces = 0
    processed = 0
    if args.workers <= 1:
        _init_worker(args.det_size, args.size)
        for image_path in images:
            p, f = _process_image(image_path, images_dir, detected_dir, args.max_input)
            processed += p
            total_faces += f
    else:
        try:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_worker,
                initargs=(args.det_size, args.size),
            ) as executor:
                results = executor.map(
                    _process_image,
                    images,
                    itertools.repeat(images_dir),
                    itertools.repeat(detected_dir),
                    itertools.repeat(args.max_input),
                    chunksize=args.chunksize,
                )
                for p, f in results:
                    processed += p
                    total_faces += f
        except Exception as exc:
            print(f"Multiprocessing failed ({exc}); retrying with --workers 1")
            _init_worker(args.det_size, args.size)
            images = iter_images(images_dir)
            for image_path in images:
                p, f = _process_image(image_path, images_dir, detected_dir, args.max_input)
                processed += p
                total_faces += f

    if processed == 0:
        raise SystemExit(f"No images found in {images_dir}")
    print(f"Processed {processed} images")
    print(f"Saved {total_faces} aligned faces to {detected_dir}")


if __name__ == "__main__":
    main()
