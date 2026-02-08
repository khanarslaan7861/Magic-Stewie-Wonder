from pathlib import Path
import argparse

import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(images_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(images_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    images_dir = root / "segregated" / "images"
    detected_dir = root / "detected"
    if not images_dir.exists():
        raise SystemExit(f"segregated/images folder not found: {images_dir}")

    images = iter_images(images_dir)
    if not images:
        raise SystemExit(f"No images found in {images_dir}")

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    detected_dir.mkdir(parents=True, exist_ok=True)

    total_faces = 0
    for image_path in images:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        faces = app.get(img)
        if not faces:
            continue

        rel_stem = image_path.relative_to(images_dir).with_suffix("")
        safe_stem = "__".join(rel_stem.parts)
        for idx, face in enumerate(faces, start=1):
            if face.kps is None:
                continue
            aligned = face_align.norm_crop(img, face.kps, image_size=args.size)
            output_name = f"{safe_stem}_face{idx}.jpg"
            output_path = detected_dir / output_name
            cv2.imwrite(str(output_path), aligned)
            total_faces += 1

    print(f"Processed {len(images)} images")
    print(f"Saved {total_faces} aligned faces to {detected_dir}")


if __name__ == "__main__":
    main()
