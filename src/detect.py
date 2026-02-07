from pathlib import Path
import argparse

import cv2
from insightface.app import FaceAnalysis

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def pick_sample_image(images_dir: Path) -> Path:
    candidates = [
        path
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    if not candidates:
        raise SystemExit(f"No images found in {images_dir}")
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run face detection on a sample image.")
    parser.add_argument(
        "--image",
        help="Image filename in segregated/images (defaults to first sorted image).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    images_dir = root / "segregated" / "images"
    detected_dir = root / "detected"
    if not images_dir.exists():
        raise SystemExit(f"segregated/images folder not found: {images_dir}")

    if args.image:
        image_path = images_dir / args.image
        if not image_path.exists():
            raise SystemExit(f"Image not found: {image_path}")
        if not image_path.is_file():
            raise SystemExit(f"Not a file: {image_path}")
    else:
        image_path = pick_sample_image(images_dir)
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Failed to read image: {image_path}")

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    faces = app.get(img)
    print(f"Using sample image: {image_path}")
    print(f"Detected {len(faces)} faces")

    res = img.copy()
    for face in faces:
        box = face.bbox.astype(int)
        cv2.rectangle(res, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    detected_dir.mkdir(parents=True, exist_ok=True)
    output_path = detected_dir / f"{image_path.stem}_detected.jpg"
    cv2.imwrite(str(output_path), res)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
