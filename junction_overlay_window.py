from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageTk
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
import tkinter as tk


INPUT_PATH = Path(
    "/Users/ozerozkan/Documents/gemini-image-gen/1980s_boxy_suv/level5.png"
)
OUTPUT_PATH = Path("/Users/ozerozkan/Documents/gemini-image-gen/2010s_compact_hatchback/level52.png")
DISPLAY_MAX = (1600, 950)


def load_line_mask(path: Path) -> np.ndarray:
    gray = np.asarray(Image.open(path).convert("L"))
    thresh = threshold_otsu(gray)
    # The source is a light background with dark strokes.
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


def show_window(display_img: Image.Image, title: str) -> None:
    root = tk.Tk()
    root.title(title)

    tk_img = ImageTk.PhotoImage(display_img)
    label = tk.Label(root, image=tk_img, borderwidth=0)
    label.image = tk_img
    label.pack()

    info = tk.Label(
        root,
        text="Junction overlay preview. Close the window when done.",
        padx=10,
        pady=8,
    )
    info.pack()

    root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-window", action="store_true", help="Only save the overlay result.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_img = Image.open(INPUT_PATH).convert("RGB")
    line_mask = load_line_mask(INPUT_PATH)
    skeleton = skeletonize(line_mask)
    _, centers = detect_junctions(skeleton)
    overlay = overlay_result(base_img, skeleton, centers)
    overlay.save(OUTPUT_PATH)

    print(f"Saved overlay to: {OUTPUT_PATH}")
    print(f"Detected junction clusters: {len(centers)}")

    if not args.no_window:
        display_img = fit_for_display(overlay, DISPLAY_MAX)
        show_window(display_img, f"Junction Overlay - {INPUT_PATH.name}")


if __name__ == "__main__":
    main()
