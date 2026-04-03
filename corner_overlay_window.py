from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageTk
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
import tkinter as tk
from tkinter import filedialog, messagebox


OUTPUT_DIR = Path(__file__).resolve().parent
DISPLAY_MAX = (1600, 950)
LOOKAHEAD = 16
JUNCTION_RADIUS = 18
ENDPOINT_RADIUS = 10
MIN_TURN_DEG = 32
MIN_CORNER_DISTANCE = 18


def load_line_mask(base_img: Image.Image) -> np.ndarray:
    gray = np.asarray(base_img.convert("L"))
    threshold = threshold_otsu(gray)
    return gray < threshold


def classify_skeleton(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = ndi.convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)
    neighbor_count = neighbor_count - skeleton.astype(np.uint8)
    regular_mask = skeleton & (neighbor_count == 2)
    junction_mask = skeleton & (neighbor_count >= 3)
    endpoint_mask = skeleton & (neighbor_count == 1)
    return regular_mask, junction_mask, endpoint_mask


def connected_centers(mask: np.ndarray) -> list[tuple[int, int]]:
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    centers: list[tuple[int, int]] = []
    for label_id in range(1, count + 1):
        ys, xs = np.nonzero(labels == label_id)
        if ys.size == 0:
            continue
        cy = int(round(float(ys.mean())))
        cx = int(round(float(xs.mean())))
        centers.append((cx, cy))
    return centers


def build_adjacency(skeleton: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    ys, xs = np.nonzero(skeleton)
    coords = list(zip(ys.tolist(), xs.tolist()))
    coord_set = set(coords)
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for y, x in coords:
        pt = (y, x)
        neighbors: list[tuple[int, int]] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                other = (y + dy, x + dx)
                if other in coord_set:
                    neighbors.append(other)
        adjacency[pt] = neighbors
    return adjacency


def build_exclusion_mask(
    shape: tuple[int, int],
    junction_centers: list[tuple[int, int]],
    endpoint_centers: list[tuple[int, int]],
) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    exclude = np.zeros((h, w), dtype=bool)

    for cx, cy in junction_centers:
        exclude |= (yy - cy) ** 2 + (xx - cx) ** 2 <= JUNCTION_RADIUS**2

    for cx, cy in endpoint_centers:
        exclude |= (yy - cy) ** 2 + (xx - cx) ** 2 <= ENDPOINT_RADIUS**2

    return exclude


def walk_branch(
    start: tuple[int, int],
    prev: tuple[int, int],
    max_steps: int,
    adjacency: dict[tuple[int, int], list[tuple[int, int]]],
    regular_mask: np.ndarray,
    junction_mask: np.ndarray,
    endpoint_mask: np.ndarray,
) -> tuple[tuple[int, int], int]:
    current = start
    previous = prev
    steps = 1

    while steps < max_steps:
        next_pts = [pt for pt in adjacency[current] if pt != previous]
        next_pts = [
            pt for pt in next_pts if regular_mask[pt] or endpoint_mask[pt] or junction_mask[pt]
        ]
        if not next_pts or len(next_pts) > 1:
            break

        nxt = next_pts[0]
        previous, current = current, nxt
        steps += 1
        if endpoint_mask[current] or junction_mask[current]:
            break

    return current, steps


def detect_corners(
    skeleton: np.ndarray,
    regular_mask: np.ndarray,
    junction_mask: np.ndarray,
    endpoint_mask: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]], np.ndarray]:
    junction_centers = connected_centers(junction_mask)
    endpoint_centers = connected_centers(endpoint_mask)
    exclude = build_exclusion_mask(skeleton.shape, junction_centers, endpoint_centers)
    candidate_mask = regular_mask & (~exclude)
    adjacency = build_adjacency(skeleton)

    corner_strength = np.zeros(skeleton.shape, dtype=np.float32)
    for y, x in zip(*np.nonzero(candidate_mask)):
        pt = (int(y), int(x))
        neighbors = [n for n in adjacency[pt] if regular_mask[n] or endpoint_mask[n] or junction_mask[n]]
        if len(neighbors) != 2:
            continue

        p1, s1 = walk_branch(
            neighbors[0], pt, LOOKAHEAD, adjacency, regular_mask, junction_mask, endpoint_mask
        )
        p2, s2 = walk_branch(
            neighbors[1], pt, LOOKAHEAD, adjacency, regular_mask, junction_mask, endpoint_mask
        )
        if min(s1, s2) < max(5, LOOKAHEAD // 3):
            continue

        v1 = np.array([p1[1] - x, p1[0] - y], dtype=np.float32)
        v2 = np.array([p2[1] - x, p2[0] - y], dtype=np.float32)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            continue

        cosine = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle = math.degrees(math.acos(cosine))
        turn_angle = 180.0 - angle
        if turn_angle >= MIN_TURN_DEG:
            corner_strength[y, x] = turn_angle

    corner_coords = peak_local_max(
        corner_strength,
        min_distance=MIN_CORNER_DISTANCE,
        threshold_abs=MIN_TURN_DEG,
        exclude_border=False,
    )
    return corner_coords, junction_centers, endpoint_centers, exclude


def overlay_result(
    base_img: Image.Image,
    corner_coords: np.ndarray,
    junction_centers: list[tuple[int, int]],
) -> Image.Image:
    overlay = base_img.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")

    for cx, cy in junction_centers:
        draw.ellipse(
            (cx - JUNCTION_RADIUS, cy - JUNCTION_RADIUS, cx + JUNCTION_RADIUS, cy + JUNCTION_RADIUS),
            outline=(0, 180, 255, 170),
            width=4,
        )
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(0, 180, 255, 220))

    for y, x in corner_coords:
        radius = 7
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(220, 30, 30, 210),
            outline=(255, 255, 255, 230),
            width=2,
        )

    return overlay


def fit_for_display(img: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    fitted = img.copy()
    fitted.thumbnail(max_size, Image.Resampling.LANCZOS)
    return fitted


def build_output_path(input_path: Path) -> Path:
    return OUTPUT_DIR / f"{input_path.stem}_corner_overlay.png"


def validate_input_path(input_path: Path) -> None:
    if input_path.stem.endswith("_junction_overlay") or input_path.stem.endswith("_corner_overlay"):
        raise ValueError("Please select the original source image, not a previously generated overlay.")


def process_image(input_path: Path) -> tuple[Image.Image, Path, int, int, int]:
    validate_input_path(input_path)
    base_img = Image.open(input_path).convert("RGB")
    line_mask = load_line_mask(base_img)
    skeleton = skeletonize(line_mask)
    regular_mask, junction_mask, endpoint_mask = classify_skeleton(skeleton)
    corner_coords, junction_centers, endpoint_centers, _ = detect_corners(
        skeleton, regular_mask, junction_mask, endpoint_mask
    )
    overlay = overlay_result(base_img, corner_coords, junction_centers)
    output_path = build_output_path(input_path)
    overlay.save(output_path)
    return overlay, output_path, len(corner_coords), len(junction_centers), len(endpoint_centers)


class CornerOverlayApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Corner Overlay")
        self.root.geometry("1660x1040")

        controls = tk.Frame(root, padx=12, pady=12)
        controls.pack(fill="x")

        open_button = tk.Button(controls, text="Open Image", command=self.open_image, padx=14, pady=8)
        open_button.pack(side="left")

        self.status = tk.Label(
            controls,
            text="Select an image to run junction-masked corner detection.",
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
            overlay, output_path, corner_count, junction_count, endpoint_count = process_image(Path(input_path))
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
            f"Corners: {corner_count} | Junctions: {junction_count} | Endpoints excluded: {endpoint_count}"
        )
        self.status.config(text=status_text)
        print(f"Input image: {input_path}")
        print(f"Saved overlay to: {output_path}")
        print(f"Corners: {corner_count}")
        print(f"Junction clusters: {junction_count}")
        print(f"Endpoints excluded separately: {endpoint_count}")


def main() -> None:
    root = tk.Tk()
    app = CornerOverlayApp(root)
    app.open_image()
    root.mainloop()


if __name__ == "__main__":
    main()
