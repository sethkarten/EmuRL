#!/usr/bin/env python3
"""
Analyze videos and create grid overlay images for crop configuration.

Creates 5 sample frames with grid overlays from each video to help
identify the correct crop coordinates for the game screen.

Usage:
    uv run python analyze_video_crops.py
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict
import subprocess


# Output directory for analysis images
OUTPUT_DIR = Path("data/crop_analysis")

# Video sources
VIDEO_SOURCES = {
    "youtube_speedruns": Path("/mnt/storage/datasets/pokemon_red_youtube"),
    "letsplay": Path("/mnt/storage/datasets/pokemon_red_letsplay"),
    "existing_speedruns": Path("data/speedrun_videos"),
    "existing_speedruns2": Path("data/speedruns/videos"),
}

# Game Boy aspect ratio
GB_ASPECT = 160 / 144  # ~1.111


def get_video_id(path: Path) -> str:
    """Extract a short ID from video filename."""
    name = path.stem
    # For YouTube videos, extract the ID in brackets
    if '[' in name and ']' in name:
        return name[name.rfind('[')+1:name.rfind(']')][:8]
    # For yt_ prefixed files
    if name.startswith('yt_'):
        return name[3:11]
    # For video_ prefixed files
    if name.startswith('video_'):
        return name[6:14]
    # Default: first 8 chars
    return name[:8]


def extract_frames(video_path: Path, num_frames: int = 5) -> List[np.ndarray]:
    """Extract evenly spaced frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    # Skip first and last 10% to avoid intros/outros
    start_frame = int(total_frames * 0.1)
    end_frame = int(total_frames * 0.9)

    # Calculate frame positions
    if end_frame <= start_frame:
        positions = [total_frames // 2]
    else:
        step = (end_frame - start_frame) // (num_frames + 1)
        positions = [start_frame + step * (i + 1) for i in range(num_frames)]

    frames = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def add_grid_overlay(frame: np.ndarray, step: int = 50) -> np.ndarray:
    """Add a coordinate grid overlay to a frame."""
    result = frame.copy()
    h, w = result.shape[:2]

    # Draw vertical lines
    for x in range(0, w, step):
        color = (0, 255, 0) if x % 100 == 0 else (0, 128, 0)
        thickness = 2 if x % 100 == 0 else 1
        cv2.line(result, (x, 0), (x, h), color, thickness)
        if x % 100 == 0:
            cv2.putText(result, str(x), (x + 3, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result, str(x), (x + 3, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Draw horizontal lines
    for y in range(0, h, step):
        color = (0, 255, 0) if y % 100 == 0 else (0, 128, 0)
        thickness = 2 if y % 100 == 0 else 1
        cv2.line(result, (0, y), (w, y), color, thickness)
        if y % 100 == 0:
            cv2.putText(result, str(y), (5, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result, str(y), (5, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Add dimension info
    info = f"{w}x{h}"
    cv2.putText(result, info, (w - 120, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result, info, (w - 120, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    return result


def suggest_crop(frame: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Attempt to auto-detect game screen boundaries.
    Returns (x1, y1, x2, y2) or None if detection fails.
    """
    h, w = frame.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    best_score = 0

    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch

        # Skip too small or too large
        if area < w * h * 0.1 or area > w * h * 0.95:
            continue

        # Check aspect ratio (should be close to GB)
        aspect = cw / ch if ch > 0 else 0
        aspect_diff = abs(aspect - GB_ASPECT)

        if aspect_diff < 0.3:
            score = area * (1 - aspect_diff)
            if score > best_score:
                best_score = score
                best_rect = (x, y, x + cw, y + ch)

    return best_rect


def analyze_video(video_path: Path, output_dir: Path, source_name: str) -> Dict:
    """Analyze a single video and create grid overlay images."""
    vid_id = get_video_id(video_path)
    vid_output_dir = output_dir / source_name

    vid_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Analyzing {vid_id}...")

    # Extract frames
    frames = extract_frames(video_path, num_frames=5)
    if not frames:
        print(f"    Failed to extract frames")
        return {"id": vid_id, "status": "failed"}

    h, w = frames[0].shape[:2]

    # Create combined image with all frames + grids
    # 5 frames in a row
    grid_frames = [add_grid_overlay(f) for f in frames]

    # Create a combined image (2 rows: original on top, grid on bottom)
    # Just use first frame for the main analysis image
    main_frame = grid_frames[0]

    # Save individual grid frames
    for i, gf in enumerate(grid_frames):
        cv2.imwrite(str(vid_output_dir / f"{vid_id}_frame{i+1}.jpg"), gf)

    # Try to auto-detect crop region
    suggested = suggest_crop(frames[0])

    # Save metadata
    metadata = {
        "id": vid_id,
        "source": source_name,
        "path": str(video_path),
        "dimensions": f"{w}x{h}",
        "suggested_crop": list(suggested) if suggested else None,
    }

    return metadata


def create_summary_html(all_results: List[Dict], output_dir: Path):
    """Create an HTML summary page for easy viewing."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Video Crop Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
        h1 { color: #4CAF50; }
        h2 { color: #2196F3; margin-top: 40px; }
        .video-group { margin: 20px 0; padding: 15px; background: #2a2a2a; border-radius: 8px; }
        .video-id { font-weight: bold; color: #FF9800; font-size: 18px; }
        .dimensions { color: #888; }
        .suggested { color: #4CAF50; font-family: monospace; }
        .frames { display: flex; gap: 10px; overflow-x: auto; margin-top: 10px; }
        .frames img { height: 200px; border: 2px solid #444; border-radius: 4px; }
        .frames img:hover { border-color: #4CAF50; cursor: pointer; }
        .source-section { margin-bottom: 50px; }
    </style>
</head>
<body>
    <h1>Pokemon Red Video Crop Analysis</h1>
    <p>Click on images to view full size. Use coordinates to configure crops in video_crop_config.json</p>
"""

    # Group by source
    by_source = {}
    for r in all_results:
        src = r.get("source", "unknown")
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(r)

    for source, videos in by_source.items():
        html += f'<div class="source-section">\n'
        html += f'<h2>{source} ({len(videos)} videos)</h2>\n'

        for v in videos:
            vid_id = v["id"]
            dims = v.get("dimensions", "?")
            suggested = v.get("suggested_crop")

            html += f'<div class="video-group">\n'
            html += f'  <span class="video-id">{vid_id}</span>\n'
            html += f'  <span class="dimensions">({dims})</span>\n'

            if suggested:
                html += f'  <div class="suggested">Suggested crop: [{suggested[0]}, {suggested[1]}, {suggested[2]}, {suggested[3]}]</div>\n'

            html += f'  <div class="frames">\n'
            for i in range(5):
                img_path = f"{source}/{vid_id}_frame{i+1}.jpg"
                html += f'    <a href="{img_path}" target="_blank"><img src="{img_path}" alt="Frame {i+1}"></a>\n'
            html += f'  </div>\n'
            html += f'</div>\n'

        html += '</div>\n'

    html += """
</body>
</html>
"""

    with open(output_dir / "index.html", "w") as f:
        f.write(html)

    print(f"\nCreated summary: {output_dir}/index.html")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for source_name, source_dir in VIDEO_SOURCES.items():
        if not source_dir.exists():
            print(f"Skipping {source_name}: directory not found")
            continue

        videos = list(source_dir.glob("*.mp4")) + list(source_dir.glob("*.webm"))
        if not videos:
            print(f"Skipping {source_name}: no videos found")
            continue

        print(f"\n=== {source_name} ({len(videos)} videos) ===")

        for video_path in sorted(videos):
            result = analyze_video(video_path, OUTPUT_DIR, source_name)
            all_results.append(result)

    # Create summary
    create_summary_html(all_results, OUTPUT_DIR)

    # Save JSON metadata
    with open(OUTPUT_DIR / "analysis.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAnalysis complete!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Total videos analyzed: {len(all_results)}")
    print(f"\nOpen {OUTPUT_DIR}/index.html in a browser to review frames")


if __name__ == "__main__":
    main()
