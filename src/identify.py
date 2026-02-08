from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import tkinter as tk
from tkinter import ttk

import numpy as np
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Cosine similarity threshold to auto-suggest a label.",
    )
    parser.add_argument(
        "--max-ref-per-label",
        type=int,
        default=4,
        help="Max reference images per existing label to build suggestions.",
    )
    parser.add_argument(
        "--auto-accept-suggestions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically save when a suggestion meets the threshold.",
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
        threshold: float,
        max_ref_per_label: int,
        auto_accept_suggestions: bool,
    ) -> None:
        self.root = root
        self.images = images
        self.identified_dir = identified_dir
        self.model_name = model_name
        self.max_size = max_size
        self.threshold = threshold
        self.max_ref_per_label = max_ref_per_label
        self.auto_accept_suggestions = auto_accept_suggestions
        self.index = 0
        self.current_image: Image.Image | None = None
        self.current_photo: ImageTk.PhotoImage | None = None
        self.current_embedding: np.ndarray | None = None
        self.label_index: dict[str, list[np.ndarray]] = {}
        self.auto_accept_var = tk.BooleanVar(value=auto_accept_suggestions)

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

        ttk.Label(side, text="Suggestion").pack(anchor=tk.W, pady=(6, 0))
        self.suggest_var = tk.StringVar(value="(none)")
        self.suggest_label = ttk.Label(side, textvariable=self.suggest_var, wraplength=280)
        self.suggest_label.pack(anchor=tk.W)
        self.auto_accept_check = ttk.Checkbutton(
            side,
            text="Auto-accept suggestions",
            variable=self.auto_accept_var,
        )
        self.auto_accept_check.pack(anchor=tk.W, pady=(6, 0))

        btn_frame = ttk.Frame(side)
        btn_frame.pack(anchor=tk.W, pady=(8, 0))

        self.use_suggest_btn = ttk.Button(
            btn_frame, text="Use Suggestion", command=self.use_suggestion
        )
        self.use_suggest_btn.pack(anchor=tk.W, fill=tk.X)

        self.save_btn = ttk.Button(btn_frame, text="Save Label", command=self.save_current)
        self.save_btn.pack(anchor=tk.W, fill=tk.X, pady=(6, 0))

        self.skip_btn = ttk.Button(btn_frame, text="Skip", command=self.next_image)
        self.skip_btn.pack(anchor=tk.W, fill=tk.X, pady=(6, 0))

        self.quit_btn = ttk.Button(btn_frame, text="Quit", command=self.root.destroy)
        self.quit_btn.pack(anchor=tk.W, fill=tk.X, pady=(6, 0))

        self.root.bind("<Return>", lambda _event: self.save_current())
        self.root.bind("<Right>", lambda _event: self.next_image())
        self.root.bind("<Escape>", lambda _event: self.root.destroy())

        self._build_label_index()
        self.load_current()

    def set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _set_suggestion(self, label: str | None, score: float | None) -> None:
        if not label or score is None:
            self.suggest_var.set("(none)")
            self.use_suggest_btn.configure(state=tk.DISABLED)
            return
        self.suggest_var.set(f"{label} ({score:.2f})")
        self.use_suggest_btn.configure(state=tk.NORMAL)

    def _build_label_index(self) -> None:
        self.set_status("Indexing existing labels...")
        self.root.update_idletasks()
        if not self.identified_dir.exists():
            return
        for label_dir in sorted(self.identified_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            images = iter_images(label_dir)[: self.max_ref_per_label]
            if not images:
                continue
            vectors: list[np.ndarray] = []
            for img_path in images:
                embedding = self._get_embedding(img_path)
                if embedding is None:
                    continue
                vectors.append(embedding)
            if vectors:
                self.label_index[label] = vectors
        self.set_status("Ready.")

    def _get_embedding(self, image_path: Path) -> np.ndarray | None:
        try:
            reps = DeepFace.represent(
                img_path=str(image_path),
                model_name=self.model_name,
                enforce_detection=False,
            )
        except Exception:
            return None
        if not reps:
            return None
        embedding = reps[0].get("embedding")
        if embedding is None:
            return None
        return np.asarray(embedding, dtype=np.float32)

    def _suggest_label(self, embedding: np.ndarray) -> tuple[str | None, float | None]:
        best_label: str | None = None
        best_score: float | None = None
        for label, vectors in self.label_index.items():
            for vec in vectors:
                score = self._cosine_similarity(embedding, vec)
                if best_score is None or score > best_score:
                    best_score = score
                    best_label = label
        if best_score is None or best_score < self.threshold:
            return None, None
        return best_label, best_score

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return -1.0
        return float(np.dot(a, b) / denom)

    def load_current(self) -> None:
        if self.index >= len(self.images):
            self.set_status("All images processed.")
            self.save_btn.configure(state=tk.DISABLED)
            self.skip_btn.configure(state=tk.DISABLED)
            self.label_entry.configure(state=tk.DISABLED)
            self.use_suggest_btn.configure(state=tk.DISABLED)
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
        self._set_suggestion(None, None)
        self.set_status(f"Scanning {image_path.name} with {self.model_name}...")
        self.root.update_idletasks()

        try:
            self.current_embedding = self._get_embedding(image_path)
            if self.current_embedding is not None and self.label_index:
                label, score = self._suggest_label(self.current_embedding)
                self._set_suggestion(label, score)
                if label and self.auto_accept_var.get():
                    self.label_var.set(label)
                    self.set_status(f"Auto-saved suggestion: {label}")
                    self.root.update_idletasks()
                    self.root.after(1, self.save_current)
                    return
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

    def use_suggestion(self) -> None:
        suggestion = self.suggest_var.get()
        if suggestion == "(none)":
            return
        label = suggestion.split("(", 1)[0].strip()
        if label:
            self.label_var.set(label)
            self.save_current()

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
        if self.current_embedding is None:
            self.current_embedding = self._get_embedding(src_path)
        if self.current_embedding is not None:
            self.label_index.setdefault(label, []).append(self.current_embedding)
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
    LabelerApp(
        root,
        images,
        identified_dir,
        args.model,
        args.size,
        args.threshold,
        args.max_ref_per_label,
        args.auto_accept_suggestions,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
