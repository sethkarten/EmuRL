#!/usr/bin/env python3
"""
Interactive video crop tool.

Scans the data/ folder for videos, shows a still frame with a draggable
crop rectangle, and saves crop coordinates to configs/video_crop_config.json.

Usage:
    uv run python crop_tool.py
    uv run python crop_tool.py --data ../data/my_videos --config ../configs/video_crop_config.json
"""

import argparse
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_CONFIG = ROOT / "configs" / "video_crop_config.json"

GBA_WIDTH = 240
GBA_HEIGHT = 160

CANVAS_MAX_W = 900
CANVAS_MAX_H = 600

# Zoom lens
LENS_SIZE = 16    # original pixels captured around each corner (NxN)
LENS_MAG  = 6     # magnification factor
LENS_PX   = LENS_SIZE * LENS_MAG   # = 96 display pixels per side
HANDLE_RADIUS = 10  # canvas-pixel hit radius around each corner handle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_videos(data_dir: Path) -> list[Path]:
    exts = ("*.mp4", "*.mkv", "*.webm", "*.avi")
    videos = []
    for ext in exts:
        videos.extend(data_dir.rglob(ext))
    # Exclude anything inside a "frames" directory (those are .npy, but just in case)
    videos = [v for v in videos if "frames" not in v.parts]
    return sorted(videos)


def load_config(path: Path) -> dict:
    if path.exists() and path.stat().st_size > 0:
        with open(path) as f:
            return json.load(f)
    return {"usable_videos": {}, "skip_videos": {}}


def save_config(config: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def extract_still(video_path: Path) -> tuple[np.ndarray | None, float, float, float]:
    """
    Extract a single representative frame.
    Returns (frame_bgr, fps, orig_width, orig_height).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Seek to ~25% to skip intros
    seek = max(0, int(total * 0.25))
    cap.set(cv2.CAP_PROP_POS_FRAMES, seek)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, fps, w, h
    return frame, fps, w, h


def scale_frame(frame_bgr: np.ndarray) -> tuple[ImageTk.PhotoImage, float]:
    """Scale frame to fit within CANVAS_MAX_W x CANVAS_MAX_H, return (PhotoImage, scale)."""
    h, w = frame_bgr.shape[:2]
    scale = min(CANVAS_MAX_W / w, CANVAS_MAX_H / h, 1.0)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray(rgb))
    return img, scale


def make_preview(frame_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> ImageTk.PhotoImage:
    """Crop and resize to exact GBA resolution 240x160."""
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_bgr.shape[1], x2)
    y2 = min(frame_bgr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        blank = np.zeros((GBA_HEIGHT, GBA_WIDTH, 3), dtype=np.uint8)
        return ImageTk.PhotoImage(Image.fromarray(blank))
    cropped = frame_bgr[y1:y2, x1:x2]
    gba = cv2.resize(cropped, (GBA_WIDTH, GBA_HEIGHT), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(gba, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class CropTool:
    def __init__(self, root: tk.Tk, videos: list[Path], config: dict, config_path: Path):
        self.root = root
        self.videos = videos
        self.config = config
        self.config_path = config_path
        self.idx = 0

        # Current video state
        self.frame_bgr: np.ndarray | None = None
        self.scale: float = 1.0
        self.orig_w: int = 0
        self.orig_h: int = 0
        self.fps: float = 0.0
        self.canvas_w: int = 0
        self.canvas_h: int = 0

        # Zoom lens photo references (prevent GC)
        self._lens_photos: list = []

        # Drag state (canvas coords)
        self.drag_mode: str | None = None   # "new" or "resize"
        self.drag_start: tuple[int, int] | None = None
        self.rect_canvas: tuple[int, int, int, int] | None = None  # canvas coords
        self.rect_orig: tuple[int, int, int, int] | None = None    # original pixel coords
        self.last_crop: tuple[int, int, int, int] | None = None    # crop from previous video

        self._build_ui()
        self._load_video(0)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.root.title("Video Crop Tool")
        self.root.configure(bg="#1e1e1e")
        self.root.resizable(True, True)
        self.root.geometry("1200x700")

        # ── Top info bar ───────────────────────────────────────────────
        info_frame = tk.Frame(self.root, bg="#2d2d2d", pady=6)
        info_frame.pack(fill=tk.X, padx=8, pady=(8, 0))

        self.lbl_file = tk.Label(info_frame, text="", bg="#2d2d2d", fg="#e0e0e0",
                                 font=("Helvetica", 11, "bold"), anchor="w")
        self.lbl_file.pack(side=tk.LEFT, padx=10)

        self.lbl_info = tk.Label(info_frame, text="", bg="#2d2d2d", fg="#aaaaaa",
                                 font=("Helvetica", 10), anchor="e")
        self.lbl_info.pack(side=tk.RIGHT, padx=10)

        # ── Main area ─────────────────────────────────────────────────
        main = tk.Frame(self.root, bg="#1e1e1e")
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Canvas (left)
        canvas_frame = tk.Frame(main, bg="#1e1e1e")
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#111111", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack(expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # Right panel
        right = tk.Frame(main, bg="#1e1e1e", width=260)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right.pack_propagate(False)

        # Preview
        tk.Label(right, text="Preview (GBA 240×160)", bg="#1e1e1e", fg="#888888",
                 font=("Helvetica", 9)).pack(anchor="w")
        self.preview_canvas = tk.Canvas(right, width=GBA_WIDTH, height=GBA_HEIGHT,
                                        bg="#111111", highlightthickness=1,
                                        highlightbackground="#444444")
        self.preview_canvas.pack(pady=(2, 10))

        # Coordinates
        tk.Label(right, text="Crop coordinates", bg="#1e1e1e", fg="#888888",
                 font=("Helvetica", 9)).pack(anchor="w")
        self.lbl_coords = tk.Label(right, text="x1=0  y1=0\nx2=0  y2=0\nw=0  h=0\naspect=—",
                                   bg="#2d2d2d", fg="#4caf50",
                                   font=("Courier", 10), justify=tk.LEFT,
                                   relief=tk.SOLID, padx=8, pady=6)
        self.lbl_coords.pack(fill=tk.X, pady=(2, 10))

        # Existing crop indicator
        self.lbl_status = tk.Label(right, text="", bg="#1e1e1e", fg="#ff9800",
                                   font=("Helvetica", 9), wraplength=200, justify=tk.LEFT)
        self.lbl_status.pack(anchor="w", pady=(0, 8))

        # Buttons
        btn_cfg = dict(bg="#333333", fg="black", activebackground="#555555",
                       activeforeground="black", relief=tk.FLAT,
                       font=("Helvetica", 10), padx=8, pady=5, cursor="hand2")

        tk.Button(right, text="💾  Save crop", command=self._save_crop,
                  bg="#2e7d32", fg="black", activebackground="#388e3c",
                  activeforeground="black", relief=tk.FLAT,
                  font=("Helvetica", 10, "bold"), padx=8, pady=6,
                  cursor="hand2").pack(fill=tk.X, pady=(0, 4))

        tk.Button(right, text="✖  Mark as skip", command=self._mark_skip,
                  bg="#b71c1c", fg="black", activebackground="#c62828",
                  activeforeground="black", relief=tk.FLAT,
                  font=("Helvetica", 10), padx=8, pady=5,
                  cursor="hand2").pack(fill=tk.X, pady=(0, 4))

        tk.Button(right, text="✕  Clear crop", command=self._clear_crop,
                  **btn_cfg).pack(fill=tk.X, pady=(0, 4))

        tk.Button(right, text="⛶  No crop (full frame)", command=self._no_crop,
                  **btn_cfg).pack(fill=tk.X, pady=(0, 4))

        self.btn_copy_last = tk.Button(right, text="⧉  Copy crop from last video",
                                       command=self._copy_last_crop, state=tk.DISABLED,
                                       disabledforeground="black",
                                       **btn_cfg)
        self.btn_copy_last.pack(fill=tk.X, pady=(0, 12))

        # Progress
        self.lbl_progress = tk.Label(right, text="", bg="#1e1e1e", fg="#888888",
                                     font=("Helvetica", 9))
        self.lbl_progress.pack(anchor="w", pady=(0, 4))

        prog_frame = tk.Frame(right, bg="#1e1e1e")
        prog_frame.pack(fill=tk.X, pady=(0, 8))
        tk.Button(prog_frame, text="◀ Prev", command=self._prev,
                  **btn_cfg).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        tk.Button(prog_frame, text="Next ▶", command=self._next,
                  **btn_cfg).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

    # ------------------------------------------------------------------
    # Video loading
    # ------------------------------------------------------------------

    def _load_video(self, idx: int):
        self.idx = idx
        path = self.videos[idx]
        video_id = path.stem

        self.lbl_progress.config(text=f"Video {idx + 1} / {len(self.videos)}")
        self.lbl_file.config(text=path.name)

        frame, fps, w, h = extract_still(path)
        self.fps = fps
        self.orig_w = int(w)
        self.orig_h = int(h)

        if frame is None:
            self.frame_bgr = None
            self.lbl_info.config(text="Could not read video")
            self.canvas.delete("all")
            self.canvas.create_text(200, 150, text="Could not read video",
                                    fill="#ff5555", font=("Helvetica", 14))
            return

        self.frame_bgr = frame
        aspect = w / h if h else 0
        self.lbl_info.config(text=f"{int(w)}×{int(h)}  |  {fps:.2f} fps  |  aspect {aspect:.3f}")

        self._render_frame()

        # Restore existing crop if any
        self.rect_canvas = None
        self.rect_orig = None
        self._update_coords_display()
        self._check_existing_config(video_id)

    def _render_frame(self):
        if self.frame_bgr is None:
            return
        photo, scale = scale_frame(self.frame_bgr)
        self.scale = scale
        self._photo = photo  # keep reference
        self.canvas_w = photo.width()
        self.canvas_h = photo.height()

        self.canvas.config(width=photo.width(), height=photo.height())
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo, tags="bg")

    def _check_existing_config(self, video_id: str):
        """Restore and display any previously saved crop for this video."""
        usable = self.config.get("usable_videos", {})
        skip = self.config.get("skip_videos", {})

        for key in usable:
            if key.startswith("_"):
                continue
            if video_id.startswith(key) or key in video_id or key == video_id:
                crop = usable[key].get("crop")
                if crop and isinstance(crop, list) and len(crop) == 4:
                    x1, y1, x2, y2 = crop
                    self.rect_orig = (x1, y1, x2, y2)
                    self._orig_to_canvas_rect(x1, y1, x2, y2)
                    self._update_coords_display()
                    self.lbl_status.config(text="Saved crop loaded", fg="#4caf50")
                return

        for key in skip:
            if video_id.startswith(key) or key in video_id or key == video_id:
                self.lbl_status.config(text=f"Marked as SKIP:\n{skip[key]}", fg="#ff5555")
                return

        self.lbl_status.config(text="Not configured yet", fg="#ff9800")

    def _orig_to_canvas_rect(self, x1, y1, x2, y2):
        s = self.scale
        cx1, cy1, cx2, cy2 = int(x1 * s), int(y1 * s), int(x2 * s), int(y2 * s)
        self.rect_canvas = (cx1, cy1, cx2, cy2)
        self._draw_rect(cx1, cy1, cx2, cy2)

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def _clamp_canvas(self, x: int, y: int) -> tuple[int, int]:
        """Clamp canvas coordinates to the video frame bounds."""
        return max(0, min(x, self.canvas_w)), max(0, min(y, self.canvas_h))

    def _hit_corner(self, x: int, y: int) -> str | None:
        """Return corner name ('tl','tr','bl','br') if (x,y) is within HANDLE_RADIUS of it."""
        if self.rect_canvas is None:
            return None
        cx1, cy1, cx2, cy2 = self.rect_canvas
        for name, (hx, hy) in [("tl", (cx1, cy1)), ("tr", (cx2, cy1)),
                                ("bl", (cx1, cy2)), ("br", (cx2, cy2))]:
            if abs(x - hx) <= HANDLE_RADIUS and abs(y - hy) <= HANDLE_RADIUS:
                return name
        return None

    def _on_press(self, event):
        x, y = self._clamp_canvas(event.x, event.y)
        hit = self._hit_corner(x, y)
        if hit and self.rect_canvas is not None:
            # Resize: anchor = opposite corner, stays fixed
            cx1, cy1, cx2, cy2 = self.rect_canvas
            anchors = {"tl": (cx2, cy2), "tr": (cx1, cy2),
                       "bl": (cx2, cy1), "br": (cx1, cy1)}
            self.drag_start = anchors[hit]
            self.drag_mode = "resize"
            self.canvas.config(cursor="none")
        else:
            self.drag_start = (x, y)
            self.drag_mode = "new"
            self.canvas.delete("rect")
            self.canvas.delete("lens")
            self._lens_photos = []

    def _on_drag(self, event):
        if not self.drag_start:
            return
        x0, y0 = self.drag_start
        x, y = self._clamp_canvas(event.x, event.y)
        self._draw_rect(x0, y0, x, y)
        self._canvas_to_orig_update(x0, y0, x, y)

    def _on_release(self, event):
        if not self.drag_start:
            return
        x0, y0 = self.drag_start
        ex, ey = self._clamp_canvas(event.x, event.y)
        x1, x2 = sorted([x0, ex])
        y1, y2 = sorted([y0, ey])
        self.rect_canvas = (x1, y1, x2, y2)
        self._draw_rect(x1, y1, x2, y2)
        self._canvas_to_orig_update(x0, y0, ex, ey)
        self.drag_start = None
        self.drag_mode = None
        self.canvas.config(cursor="crosshair")

        # Update preview
        if self.rect_orig and self.frame_bgr is not None:
            ox1, oy1, ox2, oy2 = self.rect_orig
            preview = make_preview(self.frame_bgr, ox1, oy1, ox2, oy2)
            self._preview_photo = preview
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=preview)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_rect(self, x1, y1, x2, y2):
        self.canvas.delete("rect")
        # Rectangle outline
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="#ff9800",
                                     width=2, tags="rect", dash=(4, 2))
        # Corner handles
        sz = 5
        for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            self.canvas.create_rectangle(cx - sz, cy - sz, cx + sz, cy + sz,
                                         fill="#ff9800", outline="#ffffff",
                                         width=1, tags="rect")
        self._draw_zoom_lenses(x1, y1, x2, y2)

    def _draw_zoom_lenses(self, cx1: int, cy1: int, cx2: int, cy2: int):
        """Draw a magnified zoom lens at each corner of the crop rectangle."""
        self.canvas.delete("lens")
        if self.frame_bgr is None or self.scale == 0:
            return

        self._lens_photos = []
        s = self.scale

        corners = [(cx1, cy1), (cx2, cy1), (cx1, cy2), (cx2, cy2)]

        for (ccx, ccy) in corners:
            # Center the lens box on the corner pixel — may extend beyond canvas edge
            lx = ccx - LENS_PX // 2
            ly = ccy - LENS_PX // 2

            # Original-frame corner coords
            corner_col = int(ccx / s)
            corner_row = int(ccy / s)

            # Extract LENS_SIZE × LENS_SIZE patch centred on the corner
            half = LENS_SIZE // 2
            x1p = max(0, min(corner_col - half, self.orig_w - LENS_SIZE))
            y1p = max(0, min(corner_row - half, self.orig_h - LENS_SIZE))
            x2p = x1p + LENS_SIZE
            y2p = y1p + LENS_SIZE

            patch = self.frame_bgr[y1p:y2p, x1p:x2p]
            ph, pw = patch.shape[:2]
            if ph < LENS_SIZE or pw < LENS_SIZE:
                padded = np.zeros((LENS_SIZE, LENS_SIZE, 3), dtype=np.uint8)
                padded[:ph, :pw] = patch
                patch = padded

            # Nearest-neighbour magnification to preserve hard pixel edges
            display = cv2.resize(patch, (LENS_PX, LENS_PX),
                                 interpolation=cv2.INTER_NEAREST)
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._lens_photos.append(photo)

            self.canvas.create_image(lx, ly, anchor=tk.NW, image=photo, tags="lens")

            # Outer border
            self.canvas.create_rectangle(lx, ly, lx + LENS_PX, ly + LENS_PX,
                                         outline="#ff9800", width=1, tags="lens")

            # Pixel grid
            for i in range(1, LENS_SIZE):
                gx = lx + i * LENS_MAG
                gy = ly + i * LENS_MAG
                self.canvas.create_line(gx, ly, gx, ly + LENS_PX,
                                        fill="#444444", tags="lens")
                self.canvas.create_line(lx, gy, lx + LENS_PX, gy,
                                        fill="#444444", tags="lens")

            # Crosshair through the exact corner pixel
            px_col = max(0, min(corner_col - x1p, LENS_SIZE - 1))
            px_row = max(0, min(corner_row - y1p, LENS_SIZE - 1))
            hx = lx + px_col * LENS_MAG
            hy = ly + px_row * LENS_MAG
            mid = LENS_MAG // 2
            self.canvas.create_line(lx, hy + mid, lx + LENS_PX, hy + mid,
                                    fill="#ff4444", width=1, tags="lens")
            self.canvas.create_line(hx + mid, ly, hx + mid, ly + LENS_PX,
                                    fill="#ff4444", width=1, tags="lens")
            # White highlight box around the corner pixel
            self.canvas.create_rectangle(hx, hy, hx + LENS_MAG, hy + LENS_MAG,
                                         outline="#ffffff", width=2, tags="lens")

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _canvas_to_orig_update(self, cx0, cy0, cx1, cy1):
        s = self.scale
        if s == 0:
            return
        x1 = int(min(cx0, cx1) / s)
        y1 = int(min(cy0, cy1) / s)
        x2 = int(max(cx0, cx1) / s)
        y2 = int(max(cy0, cy1) / s)
        # Clamp to frame
        x1 = max(0, min(x1, self.orig_w))
        y1 = max(0, min(y1, self.orig_h))
        x2 = max(0, min(x2, self.orig_w))
        y2 = max(0, min(y2, self.orig_h))
        self.rect_orig = (x1, y1, x2, y2)
        self._update_coords_display()

    def _update_coords_display(self):
        if self.rect_orig:
            x1, y1, x2, y2 = self.rect_orig
            w = x2 - x1
            h = y2 - y1
            aspect = w / h if h else 0
            self.lbl_coords.config(
                text=f"x1={x1}  y1={y1}\nx2={x2}  y2={y2}\nw={w}  h={h}\naspect={aspect:.3f}"
            )
        else:
            self.lbl_coords.config(text="x1=—  y1=—\nx2=—  y2=—\nw=—  h=—\naspect=—")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _save_crop(self):
        if not self.rect_orig:
            messagebox.showwarning("No crop", "Draw a crop rectangle first.")
            return
        video_id = self.videos[self.idx].stem
        x1, y1, x2, y2 = self.rect_orig
        if "usable_videos" not in self.config:
            self.config["usable_videos"] = {}
        # Remove from skip if present
        self.config.get("skip_videos", {}).pop(video_id, None)
        self.config["usable_videos"][video_id] = {"crop": [x1, y1, x2, y2]}
        save_config(self.config, self.config_path)
        self.last_crop = (x1, y1, x2, y2)
        self.btn_copy_last.config(state=tk.NORMAL)
        self.lbl_status.config(text=f"Saved ✓\n[{x1}, {y1}, {x2}, {y2}]", fg="#4caf50")

    def _mark_skip(self):
        video_id = self.videos[self.idx].stem
        if "skip_videos" not in self.config:
            self.config["skip_videos"] = {}
        self.config.get("usable_videos", {}).pop(video_id, None)
        self.config["skip_videos"][video_id] = "manually skipped"
        save_config(self.config, self.config_path)
        self.lbl_status.config(text="Marked as SKIP ✓", fg="#ff5555")

    def _clear_crop(self):
        self.rect_canvas = None
        self.rect_orig = None
        self.canvas.delete("rect")
        self.canvas.delete("lens")
        self._lens_photos = []
        self.preview_canvas.delete("all")
        self._update_coords_display()

    def _no_crop(self):
        """Use the full video frame as the crop region."""
        if self.orig_w == 0 or self.orig_h == 0:
            return
        self.rect_orig = (0, 0, self.orig_w, self.orig_h)
        self._orig_to_canvas_rect(0, 0, self.orig_w, self.orig_h)
        self._update_coords_display()
        if self.frame_bgr is not None:
            preview = make_preview(self.frame_bgr, 0, 0, self.orig_w, self.orig_h)
            self._preview_photo = preview
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=preview)

    def _copy_last_crop(self):
        """Apply the crop from the previously saved video."""
        if not self.last_crop:
            return
        x1, y1, x2, y2 = self.last_crop
        self.rect_orig = (x1, y1, x2, y2)
        self._orig_to_canvas_rect(x1, y1, x2, y2)
        self._update_coords_display()
        if self.frame_bgr is not None:
            preview = make_preview(self.frame_bgr, x1, y1, x2, y2)
            self._preview_photo = preview
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=preview)
        self.lbl_status.config(text=f"Copied from last\n[{x1}, {y1}, {x2}, {y2}]", fg="#ff9800")

    def _prev(self):
        if self.idx > 0:
            self._load_video(self.idx - 1)

    def _next(self):
        if self.idx < len(self.videos) - 1:
            self._load_video(self.idx + 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive video crop tool")
    parser.add_argument("--data", default=str(DEFAULT_DATA_DIR),
                        help="Root data directory to scan for videos")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="Path to video_crop_config.json")
    args = parser.parse_args()

    data_dir = Path(args.data)
    config_path = Path(args.config)

    videos = find_videos(data_dir)
    if not videos:
        print(f"No videos found under {data_dir}")
        sys.exit(1)

    print(f"Found {len(videos)} video(s)")
    print(f"Config: {config_path}")

    config = load_config(config_path)

    root = tk.Tk()
    CropTool(root, videos, config, config_path)
    root.mainloop()


if __name__ == "__main__":
    main()
