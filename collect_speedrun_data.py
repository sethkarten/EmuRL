#!/usr/bin/env python3
"""
Collect and process Pokemon Red speedrun videos for CNN reward model training.

Steps:
1. Download speedrun videos from YouTube (linked on speedrun.com)
2. Extract frames at regular intervals
3. Crop to game area (detect Game Boy screen)
4. Downsample to 144x160x3
5. Save with temporal progress labels

Usage:
    python collect_speedrun_data.py --video URL --output data/speedruns/
    python collect_speedrun_data.py --video-list speedrun_urls.txt --output data/speedruns/
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import subprocess
import json
from dataclasses import dataclass
from typing import Optional, Tuple, List
import hashlib


@dataclass
class VideoConfig:
    # Frame extraction
    fps: float = 2.0  # Extract 2 frames per second
    start_time: float = 0.0  # Skip intro (seconds)
    end_time: Optional[float] = None  # Stop before credits

    # Game Boy screen dimensions
    target_width: int = 160
    target_height: int = 144

    # Crop detection
    min_game_area_ratio: float = 0.1  # Minimum fraction of frame that's game
    border_color_threshold: int = 30  # For detecting black borders


def download_video(url: str, output_dir: Path) -> Path:
    """Download video using yt-dlp."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename from URL hash
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    output_path = output_dir / f"video_{url_hash}.mp4"

    if output_path.exists():
        print(f"Video already downloaded: {output_path}")
        return output_path

    print(f"Downloading: {url}")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "-o", str(output_path),
        "--no-playlist",
        url
    ]

    try:
        subprocess.run(cmd, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Failed to download: {e}")
        return None


def detect_game_area(frame: np.ndarray, config: VideoConfig) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the Game Boy screen area in a frame.
    Returns (x, y, width, height) or None if not found.

    Looks for:
    - Rectangular region with Game Boy aspect ratio (~160:144 = 1.11)
    - Surrounded by borders (usually black or solid color)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Try multiple detection strategies

    # Strategy 1: Find largest bright rectangle (game area is usually brighter than borders)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    best_area = 0
    target_ratio = 160 / 144  # ~1.11

    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch

        if area < h * w * config.min_game_area_ratio:
            continue

        ratio = cw / ch if ch > 0 else 0

        # Check if aspect ratio is close to Game Boy
        if 0.9 < ratio < 1.3 and area > best_area:
            best_area = area
            best_rect = (x, y, cw, ch)

    if best_rect:
        return best_rect

    # Strategy 2: Assume game is centered with black borders
    # Find the non-black region
    mask = gray > config.border_color_threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return (x_min, y_min, x_max - x_min, y_max - y_min)


def crop_and_resize(frame: np.ndarray, crop_rect: Tuple[int, int, int, int],
                    config: VideoConfig) -> np.ndarray:
    """Crop to game area and resize to Game Boy resolution."""
    x, y, w, h = crop_rect
    cropped = frame[y:y+h, x:x+w]

    # Resize to exact Game Boy dimensions
    resized = cv2.resize(cropped, (config.target_width, config.target_height),
                         interpolation=cv2.INTER_AREA)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    return rgb


def extract_frames(video_path: Path, output_dir: Path, config: VideoConfig) -> List[Path]:
    """Extract and process frames from video."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    print(f"Video: {duration:.1f}s, {video_fps:.1f} fps, {total_frames} frames")

    # Calculate frame indices to extract
    frame_interval = int(video_fps / config.fps)
    start_frame = int(config.start_time * video_fps)
    end_frame = int(config.end_time * video_fps) if config.end_time else total_frames

    # First pass: detect game area from sample frames
    print("Detecting game area...")
    sample_indices = np.linspace(start_frame, min(end_frame, total_frames-1), 10, dtype=int)
    crop_rect = None

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            crop_rect = detect_game_area(frame, config)
            if crop_rect:
                print(f"Found game area: {crop_rect}")
                break

    if not crop_rect:
        print("Could not detect game area, using full frame")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            crop_rect = (0, 0, w, h)
        else:
            return []

    # Second pass: extract frames
    print(f"Extracting frames from {start_frame} to {end_frame} (every {frame_interval} frames)...")

    frame_paths = []
    frame_idx = start_frame
    extracted = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while frame_idx < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            break

        # Process frame
        processed = crop_and_resize(frame, crop_rect, config)

        # Save frame
        timestamp = frame_idx / video_fps
        frame_path = output_dir / f"frame_{extracted:06d}_t{timestamp:.2f}.npy"
        np.save(frame_path, processed)
        frame_paths.append(frame_path)

        extracted += 1
        frame_idx += frame_interval

        if extracted % 100 == 0:
            print(f"  Extracted {extracted} frames...")

    cap.release()
    print(f"Extracted {extracted} frames to {output_dir}")

    return frame_paths


def create_dataset_index(frame_paths: List[Path], output_path: Path, video_url: str):
    """Create a JSON index of the dataset with progress labels."""

    # Sort by frame number to ensure temporal ordering
    sorted_paths = sorted(frame_paths, key=lambda p: int(p.stem.split('_')[1]))

    total_frames = len(sorted_paths)

    entries = []
    for i, path in enumerate(sorted_paths):
        # Progress label: 0.0 (start) to 1.0 (end)
        progress = i / (total_frames - 1) if total_frames > 1 else 0.0

        # Extract timestamp from filename
        parts = path.stem.split('_t')
        timestamp = float(parts[1]) if len(parts) > 1 else 0.0

        entries.append({
            "path": str(path.relative_to(output_path.parent)),
            "frame_idx": i,
            "progress": progress,
            "timestamp": timestamp,
        })

    index = {
        "video_url": video_url,
        "total_frames": total_frames,
        "resolution": [144, 160, 3],
        "frames": entries,
    }

    with open(output_path, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"Created index: {output_path}")


def process_video(url: str, output_dir: Path, config: VideoConfig):
    """Full pipeline: download, extract, process, index."""

    # Download
    video_path = download_video(url, output_dir / "videos")
    if not video_path:
        return

    # Extract frames
    video_name = video_path.stem
    frames_dir = output_dir / "frames" / video_name
    frame_paths = extract_frames(video_path, frames_dir, config)

    if not frame_paths:
        return

    # Create index
    index_path = output_dir / "frames" / f"{video_name}_index.json"
    create_dataset_index(frame_paths, index_path, url)


def main():
    parser = argparse.ArgumentParser(description="Collect speedrun data for CNN training")
    parser.add_argument("--video", type=str, help="Single video URL")
    parser.add_argument("--video-list", type=str, help="File with video URLs (one per line)")
    parser.add_argument("--output", type=str, default="data/speedruns", help="Output directory")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames to extract per second")
    parser.add_argument("--start", type=float, default=0.0, help="Start time (seconds)")
    parser.add_argument("--end", type=float, default=None, help="End time (seconds)")
    args = parser.parse_args()

    config = VideoConfig(
        fps=args.fps,
        start_time=args.start,
        end_time=args.end,
    )

    output_dir = Path(args.output)

    urls = []
    if args.video:
        urls.append(args.video)
    if args.video_list:
        with open(args.video_list) as f:
            urls.extend(line.strip() for line in f if line.strip() and not line.startswith('#'))

    if not urls:
        print("No videos specified. Use --video URL or --video-list FILE")
        print("\nExample Pokemon Red speedrun URLs to add to a file:")
        print("  https://www.youtube.com/watch?v=...  # WR run")
        return

    for url in urls:
        print(f"\n{'='*60}")
        print(f"Processing: {url}")
        print('='*60)
        process_video(url, output_dir, config)

    print("\n" + "="*60)
    print("Data collection complete!")
    print(f"Frames saved to: {output_dir}/frames/")
    print("\nNext step: Train CNN with train_reward_cnn.py")


if __name__ == "__main__":
    main()
