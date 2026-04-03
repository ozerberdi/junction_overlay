from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageTk
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
import tkinter as tk
from tkinter import filedialog, messagebox


OUTPUT_DIR = Path(__file__).resolve().parent
DISPLAY_MAX = (1600, 950)


def load_line_mask(base_img: Image.Image) -> np.ndarray:
    gray = np.asarray(base_img.convert("L"))
    thresh = threshold_otsu(gray)
    # The source is expected to be a light background with dark strokes.
    return gray < thresh


def detect_junctions(skeleton: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float]]]:
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = ndi.convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)
    neighbor_count = neighbor_count - skeleton.astype(np.uint8)

    junction_mask = skeleton & (neighbor_count >= 3)
    labels, count = ndi.label(junction_mask, structure=np.ones((3, 3), dtype=np.uint8))

    centers: list[tuple[float, float]] = []
    for label_id in range(1, count + 1):
        ys, xs = np.nonzero(labels == label_id)
        if ys.size == 0:
            continue
        centers.append((float(xs.mean()), float(ys.mean())))

    return junction_mask, centers


def overlay_result(base_img: Image.Image, skeleton: np.ndarray, centers: list[tuple[float, float]]) -> Image.Image:
    overlay = base_img.convert("RGBA")
    pixels = np.array(overlay, copy=True)

    # Paint skeleton pixels green on top of the original drawing.
    pixels[skeleton] = np.array([0, 180, 0, 255], dtype=np.uint8)
    overlay = Image.fromarray(pixels, mode="RGBA")

    draw = ImageDraw.Draw(overlay)
    radius = max(8, math.ceil(min(base_img.size) * 0.004))
    width = max(3, radius // 3)
    for x, y in centers:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(220, 30, 30, 255), width=width)
    return overlay


def fit_for_display(img: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    fitted = img.copy()
    fitted.thumbnail(max_size, Image.Resampling.LANCZOS)
    return fitted


def validate_input_path(input_path: Path) -> None:
    if input_path.stem.endswith("_junction_overlay"):
        raise ValueError("Please select the original source image, not a previously generated output.")


def build_output_path(input_path: Path) -> Path:
    return OUTPUT_DIR / f"{input_path.stem}_junction_overlay.png"


def process_image(input_path: Path) -> tuple[Image.Image, Path, int]:
    validate_input_path(input_path)
    base_img = Image.open(input_path).convert("RGB")
    line_mask = load_line_mask(base_img)
    skeleton = skeletonize(line_mask)
    _, centers = detect_junctions(skeleton)
    overlay = overlay_result(base_img, skeleton, centers)
    output_path = build_output_path(input_path)
    overlay.save(output_path)
    return overlay, output_path, len(centers)


class JunctionOverlayApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Junction Overlay")
        self.root.geometry("1660x1040")

        controls = tk.Frame(root, padx=12, pady=12)
        controls.pack(fill="x")

        open_button = tk.Button(controls, text="Open Image", command=self.open_image, padx=14, pady=8)
        open_button.pack(side="left")

        self.status = tk.Label(
            controls,
            text="Select an image to skeletonize and overlay junction detections.",
            anchor="w",
            padx=12,
        )
        self.status.pack(side="left", fill="x", expand=True)

        self.image_label = tk.Label(root, borderwidth=0)
        self.image_label.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.image_label.image = None

    def open_image(self) -> None:
        input_path = filedialog.askopenfilename(
            title="Open image",
            initialdir=str(OUTPUT_DIR),
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not input_path:
            return

        try:
            overlay, output_path, junction_count = process_image(Path(input_path))
        except Exception as exc:
            messagebox.showerror("Processing failed", str(exc))
            self.status.config(text=f"Failed to process: {input_path}")
            return

        display_img = fit_for_display(overlay, DISPLAY_MAX)
        tk_img = ImageTk.PhotoImage(display_img)
        self.image_label.config(image=tk_img)
        self.image_label.image = tk_img

        status_text = (
            f"Saved to {output_path.name} in {OUTPUT_DIR} | "
            f"Detected junction clusters: {junction_count}"
        )
        self.status.config(text=status_text)
        print(f"Input image: {input_path}")
        print(f"Saved overlay to: {output_path}")
        print(f"Detected junction clusters: {junction_count}")


def main() -> None:
    root = tk.Tk()
    app = JunctionOverlayApp(root)
    app.open_image()
    root.mainloop()


if __name__ == "__main__":
    main()
