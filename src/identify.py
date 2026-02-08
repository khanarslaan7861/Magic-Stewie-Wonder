from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk
from deepface import DeepFace


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(images_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(images_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label faces from detected folder and save into identified." 
    )
    parser.add_argument(
        "--model",
        default="Facenet512",
        help="DeepFace model name used for representation.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=320,
        help="Max display size for the preview image.",
    )
    return parser.parse_args()


def _safe_label(label: str) -> str:
    cleaned = label.strip()
    cleaned = re.sub(r"[^a-zA-Z0-9 _-]+", "", cleaned)
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def _unique_destination(dest_dir: Path, name: str) -> Path:
    target = dest_dir / name
    if not target.exists():
        return target
    stem = Path(name).stem
    suffix = Path(name).suffix
    i = 1
    while True:
        candidate = dest_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


class LabelerApp:
    def __init__(
        self,
        root: tk.Tk,
        images: list[Path],
        identified_dir: Path,
        model_name: str,
        max_size: int,
    ) -> None:
        self.root = root
        self.images = images
        self.identified_dir = identified_dir
        self.model_name = model_name
        self.max_size = max_size
        self.index = 0
        self.current_image: Image.Image | None = None
        self.current_photo: ImageTk.PhotoImage | None = None

        self.root.title("Face Labeler")
        self.root.geometry("900x520")

        main = ttk.Frame(root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        self.image_label = ttk.Label(main)
        self.image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        side = ttk.Frame(main, padding=(16, 0))
        side.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(side, text="Label").pack(anchor=tk.W)
        self.label_var = tk.StringVar()
        self.label_entry = ttk.Entry(side, textvariable=self.label_var, width=32)
        self.label_entry.pack(anchor=tk.W, pady=(4, 12))
        self.label_entry.focus_set()

        self.status_var = tk.StringVar(value="")
        self.status_label = ttk.Label(side, textvariable=self.status_var, wraplength=280)
        self.status_label.pack(anchor=tk.W, pady=(4, 12))

        btn_frame = ttk.Frame(side)
        btn_frame.pack(anchor=tk.W, pady=(8, 0))

        self.save_btn = ttk.Button(btn_frame, text="Save Label", command=self.save_current)
        self.save_btn.pack(anchor=tk.W, fill=tk.X)

        self.skip_btn = ttk.Button(btn_frame, text="Skip", command=self.next_image)
        self.skip_btn.pack(anchor=tk.W, fill=tk.X, pady=(6, 0))

        self.quit_btn = ttk.Button(btn_frame, text="Quit", command=self.root.destroy)
        self.quit_btn.pack(anchor=tk.W, fill=tk.X, pady=(6, 0))

        self.root.bind("<Return>", lambda _event: self.save_current())
        self.root.bind("<Right>", lambda _event: self.next_image())
        self.root.bind("<Escape>", lambda _event: self.root.destroy())

        self.load_current()

    def set_status(self, message: str) -> None:
        self.status_var.set(message)

    def load_current(self) -> None:
        if self.index >= len(self.images):
            self.set_status("All images processed.")
            self.save_btn.configure(state=tk.DISABLED)
            self.skip_btn.configure(state=tk.DISABLED)
            self.label_entry.configure(state=tk.DISABLED)
            return

        image_path = self.images[self.index]
        try:
            self.current_image = Image.open(image_path).convert("RGB")
        except OSError:
            self.set_status(f"Failed to open {image_path.name}")
            self.index += 1
            self.load_current()
            return

        self._display_image(self.current_image)
        self.label_var.set("")
        self.set_status(f"Scanning {image_path.name} with {self.model_name}...")
        self.root.update_idletasks()

        try:
            DeepFace.represent(
                img_path=str(image_path),
                model_name=self.model_name,
                enforce_detection=False,
            )
            self.set_status(f"Ready to label: {image_path.name}")
        except Exception as exc:  # pragma: no cover - depends on DeepFace runtime
            self.set_status(f"DeepFace failed: {exc}")

    def _display_image(self, image: Image.Image) -> None:
        width, height = image.size
        scale = min(self.max_size / width, self.max_size / height, 1.0)
        if scale < 1.0:
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.BICUBIC)
        self.current_photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.current_photo)

    def save_current(self) -> None:
        if self.index >= len(self.images):
            return
        raw_label = self.label_var.get()
        label = _safe_label(raw_label)
        if not label:
            self.set_status("Enter a label before saving.")
            return

        dest_dir = self.identified_dir / label
        dest_dir.mkdir(parents=True, exist_ok=True)

        src_path = self.images[self.index]
        dest_path = _unique_destination(dest_dir, src_path.name)
        shutil.copy2(src_path, dest_path)
        self.set_status(f"Saved to {dest_path.parent.name}/")
        self.next_image()

    def next_image(self) -> None:
        self.index += 1
        self.load_current()


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]
    detected_dir = root_dir / "detected"
    identified_dir = root_dir / "identified"

    if not detected_dir.exists():
        raise SystemExit(f"detected folder not found: {detected_dir}")

    images = iter_images(detected_dir)
    if not images:
        raise SystemExit(f"No images found in {detected_dir}")

    identified_dir.mkdir(parents=True, exist_ok=True)

    root = tk.Tk()
    LabelerApp(root, images, identified_dir, args.model, args.size)
    root.mainloop()


if __name__ == "__main__":
    main()
