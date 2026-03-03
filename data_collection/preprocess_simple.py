#!/usr/bin/env python3
"""
Simple preprocessing for Pokemon Red videos.

Applies manually configured crop regions from video_crop_config.json
(same [x1, y1, x2, y2] format used by preprocess_videos.py), then resizes
to GBA resolution 240x160 and saves as RGB uint8 .npy arrays.

Videos with no crop entry are processed full-frame.
Videos in skip_videos are skipped entirely.

Usage:
    uv run python preprocess_simple.py
    uv run python preprocess_simple.py --input video.mp4 --output data/frames/
    uv run python preprocess_simple.py --input videos/ --output data/frames/ --fps 2.0
    uv run python preprocess_simple.py --config configs/video_crop_config.json
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

GBA_WIDTH = 240
GBA_HEIGHT = 160

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT  = ROOT / "data" / "dummy"
DEFAULT_OUTPUT = ROOT / "data" / "frames"
DEFAULT_CONFIG = ROOT / "configs" / "video_crop_config.json"


# ---------------------------------------------------------------------------
# Crop config helpers  (same contract as preprocess_videos.py)
# ---------------------------------------------------------------------------

def load_crop_config(path: Path) -> dict:
    """Load video_crop_config.json.  Returns empty config if file missing."""
    if not path.exists() or path.stat().st_size == 0:
        return {"usable_videos": {}, "skip_videos": {}}
    with open(path) as f:
        return json.load(f)


def _match_key(video_id: str, key: str) -> bool:
    """Return True if key refers to this video_id (prefix / substring / exact)."""
    return video_id.startswith(key) or key in video_id or key == video_id


def get_crop_box(video_id: str, config: dict) -> Optional[tuple[int, int, int, int]]:
    """
    Return (x1, y1, x2, y2) crop box for the video, or None if not found.
    Keys with a leading '_' are treated as comments and ignored.
    """
    for key, value in config.get("usable_videos", {}).items():
        if key.startswith("_"):
            continue
        if _match_key(video_id, key):
            crop = value.get("crop")
            if isinstance(crop, list) and len(crop) == 4:
                x1, y1, x2, y2 = (int(v) for v in crop)
                return x1, y1, x2, y2
    return None


def should_skip(video_id: str, config: dict) -> Optional[str]:
    """Return skip reason string, or None if the video should be processed."""
    for key, reason in config.get("skip_videos", {}).items():
        if _match_key(video_id, key):
            return reason
    return None


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_video(
    video_path: Path,
    output_dir: Path,
    fps: float = 2.0,
    crop_config: Optional[dict] = None,
    total_pbar: Optional["tqdm"] = None,
) -> dict:
    video_id = video_path.stem

    # --- skip check ---
    if crop_config:
        reason = should_skip(video_id, crop_config)
        if reason:
            print(f"Skipping {video_id}: {reason}")
            return {"video_id": video_id, "status": "skipped", "reason": reason}

    # --- crop box ---
    crop_box: Optional[tuple[int, int, int, int]] = None
    if crop_config:
        crop_box = get_crop_box(video_id, crop_config)

    if crop_box:
        x1, y1, x2, y2 = crop_box
        print(f"{video_id}: crop=[{x1},{y1},{x2},{y2}]")
    else:
        print(f"{video_id}: no crop config — using full frame")

    # --- open video ---
    frames_dir = output_dir / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        return {"video_id": video_id, "status": "error"}

    video_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps))

    print(f"  {total_frames} frames @ {video_fps:.1f} fps → every {frame_interval} frames")

    frame_idx   = 0
    saved_count = 0

    with tqdm(total=total_frames, unit="frame", desc=video_id,
              position=1, leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Apply crop if configured
                if crop_box is not None:
                    x1, y1, x2, y2 = crop_box
                    # Clamp to actual frame dimensions
                    fh, fw = frame.shape[:2]
                    x1c, y1c = max(0, x1), max(0, y1)
                    x2c, y2c = min(fw, x2), min(fh, y2)
                    if x2c > x1c and y2c > y1c:
                        frame = frame[y1c:y2c, x1c:x2c]

                resized = cv2.resize(frame, (GBA_WIDTH, GBA_HEIGHT),
                                     interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                np.save(frames_dir / f"frame_{saved_count:06d}.npy", rgb.astype(np.uint8))
                saved_count += 1
                pbar.set_postfix(saved=saved_count)

            frame_idx += 1
            pbar.update(1)
            if total_pbar is not None:
                total_pbar.update(1)

    cap.release()

    metadata = {
        "video_id":    video_id,
        "source_path": str(video_path),
        "frames":      saved_count,
        "fps":         fps,
        "resolution":  f"{GBA_WIDTH}x{GBA_HEIGHT}",
        "crop":        list(crop_box) if crop_box else None,
    }
    with open(frames_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved {saved_count} frames → {frames_dir}")
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Simple video preprocessor with crop-config support")
    parser.add_argument("--input",  default=str(DEFAULT_INPUT),
                        help="Input video file or directory")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output frames directory")
    parser.add_argument("--fps",    type=float, default=2.0,
                        help="Output frames per second (default: 2.0)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="Path to video_crop_config.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_config = load_crop_config(Path(args.config))
    usable = len([k for k in crop_config.get("usable_videos", {}) if not k.startswith("_")])
    skip   = len(crop_config.get("skip_videos", {}))
    print(f"Crop config: {usable} usable, {skip} skip  ({args.config})")

    if input_path.is_file():
        video_files = [input_path]
    else:
        exts = ("*.mp4", "*.mkv", "*.webm", "*.avi")
        video_files = sorted(
            f for ext in exts for f in input_path.glob(ext)
        )

    print(f"Found {len(video_files)} video(s)")

    # Pre-probe frame counts to size the total progress bar
    grand_total = 0
    for v in video_files:
        cap = cv2.VideoCapture(str(v))
        if cap.isOpened():
            grand_total += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

    results = []
    with tqdm(total=grand_total, unit="frame", desc="total", position=0, leave=True) as total_pbar:
        for v in video_files:
            result = process_video(v, output_dir, args.fps, crop_config, total_pbar)
            results.append(result)

    total_frames = sum(r.get("frames", 0) for r in results)
    print(f"\nDone: {len(video_files)} video(s), {total_frames:,} frames → {output_dir}")


if __name__ == "__main__":
    main()
