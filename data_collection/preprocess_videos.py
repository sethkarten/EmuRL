#!/usr/bin/env python3
"""
Preprocess Pokemon Red gameplay videos for training.

This script handles:
1. Game Boy screen cropping using manual config (video_crop_config.json)
2. Non-gameplay frame filtering (intros, outros, pauses)
3. Frame extraction at consistent resolution (160x144)

The Game Boy screen is 160x144 pixels. Videos may have:
- Borders/bezels around the screen
- Facecam overlays
- Commentary text
- Different aspect ratios

Crop coordinates are manually configured in video_crop_config.json for accuracy.
Auto-detection is only used as a fallback.

Usage:
    uv run python preprocess_videos.py --input /path/to/videos --output /path/to/frames
    uv run python preprocess_videos.py --config video_crop_config.json --output data/frames
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import Tuple, Optional, List, Dict, Any
import json
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import subprocess


# Game Boy screen dimensions
GB_WIDTH = 160
GB_HEIGHT = 144
GB_ASPECT = GB_WIDTH / GB_HEIGHT  # ~1.11


@dataclass
class ScreenRegion:
    """Detected game screen region."""
    x: int
    y: int
    width: int
    height: int
    confidence: float

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Crop frame to this region and resize to GB resolution."""
        cropped = frame[self.y:self.y+self.height, self.x:self.x+self.width]
        resized = cv2.resize(cropped, (GB_WIDTH, GB_HEIGHT), interpolation=cv2.INTER_AREA)
        return resized


def load_crop_config(config_path: Path) -> Dict[str, Any]:
    """Load manually configured crop regions from JSON."""
    if not config_path.exists():
        return {"usable_videos": {}, "skip_videos": {}}
    with open(config_path) as f:
        return json.load(f)


def get_video_crop_region(video_id: str, config: Dict) -> Optional[ScreenRegion]:
    """Get manually configured crop region for a video."""
    usable = config.get("usable_videos", {})

    for key, value in usable.items():
        if key.startswith("_"):  # Skip comments
            continue
        if video_id.startswith(key) or key in video_id:
            crop = value.get("crop")
            # Skip "auto" or invalid crop values
            if crop == "auto" or not crop:
                return None
            if isinstance(crop, list) and len(crop) == 4:
                x1, y1, x2, y2 = crop
                return ScreenRegion(x1, y1, x2 - x1, y2 - y1, 1.0)
    return None


def should_skip_video(video_id: str, config: Dict) -> Optional[str]:
    """Check if video should be skipped and return reason."""
    skip = config.get("skip_videos", {})
    for key, reason in skip.items():
        if video_id.startswith(key) or key in video_id:
            return reason
    return None


def detect_game_screen_by_edges(frame: np.ndarray) -> Optional[ScreenRegion]:
    """
    Detect game screen by finding rectangular regions with sharp edges.

    Game Boy screens typically have clear borders separating them from
    the surrounding video content.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_region = None
    best_score = 0

    frame_h, frame_w = frame.shape[:2]
    min_area = frame_w * frame_h * 0.05  # At least 5% of frame
    max_area = frame_w * frame_h * 0.95  # At most 95% of frame

    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if area < min_area or area > max_area:
            continue

        # Check aspect ratio (should be close to GB aspect)
        aspect = w / h if h > 0 else 0
        aspect_diff = abs(aspect - GB_ASPECT)

        if aspect_diff > 0.3:  # Allow some tolerance
            continue

        # Score based on size and aspect ratio match
        size_score = area / (frame_w * frame_h)
        aspect_score = 1.0 - aspect_diff
        score = size_score * aspect_score

        if score > best_score:
            best_score = score
            best_region = ScreenRegion(x, y, w, h, score)

    return best_region


def detect_game_screen_by_color(frame: np.ndarray) -> Optional[ScreenRegion]:
    """
    Detect game screen by looking for Game Boy color characteristics.

    Game Boy games have limited color palettes. We look for regions
    with colors typical of Pokemon Red (greens, blues, browns).
    """
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Game Boy Color palette tends to have specific saturation ranges
    # Look for regions with consistent, limited color variation

    # Calculate color variance in sliding windows
    h, w = frame.shape[:2]

    # Try different window sizes
    best_region = None
    best_score = 0

    for scale in [0.3, 0.4, 0.5, 0.6, 0.7]:
        win_w = int(w * scale)
        win_h = int(win_w / GB_ASPECT)

        if win_h > h:
            continue

        # Slide window across frame
        for y in range(0, h - win_h, win_h // 4):
            for x in range(0, w - win_w, win_w // 4):
                window = frame[y:y+win_h, x:x+win_w]

                # Check for limited color palette (Game Boy characteristic)
                unique_colors = len(np.unique(window.reshape(-1, 3), axis=0))
                color_density = unique_colors / (win_w * win_h)

                # Game Boy screens have low color density
                if color_density > 0.1:  # Too many unique colors
                    continue

                # Check for non-black content
                mean_brightness = np.mean(window)
                if mean_brightness < 20 or mean_brightness > 245:
                    continue

                # Score based on size and color characteristics
                size_score = (win_w * win_h) / (w * h)
                color_score = 1.0 - min(color_density * 10, 1.0)
                score = size_score * color_score

                if score > best_score:
                    best_score = score
                    best_region = ScreenRegion(x, y, win_w, win_h, score)

    return best_region


def detect_game_screen_by_template(
    frame: np.ndarray,
    template_frames: List[np.ndarray]
) -> Optional[ScreenRegion]:
    """
    Detect game screen by matching against known game frames.

    Uses template matching with frames from our existing dataset.
    """
    if not template_frames:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    best_region = None
    best_score = 0

    for template in template_frames[:5]:  # Use first few templates
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Try multiple scales
        for scale in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            scaled_w = int(frame.shape[1] * scale)
            scaled_h = int(scaled_w / GB_ASPECT)

            if scaled_h > frame.shape[0]:
                continue

            # Resize template to match search scale
            template_scaled = cv2.resize(template_gray, (scaled_w, scaled_h))

            # Template matching
            result = cv2.matchTemplate(gray, template_scaled, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_score and max_val > 0.5:
                best_score = max_val
                best_region = ScreenRegion(
                    max_loc[0], max_loc[1],
                    scaled_w, scaled_h,
                    max_val
                )

    return best_region


def detect_game_screen(
    frame: np.ndarray,
    template_frames: Optional[List[np.ndarray]] = None
) -> Optional[ScreenRegion]:
    """
    Detect game screen using multiple methods.
    """
    # Method 1: Template matching (most reliable if we have templates)
    if template_frames:
        region = detect_game_screen_by_template(frame, template_frames)
        if region and region.confidence > 0.6:
            return region

    # Method 2: Edge detection
    region = detect_game_screen_by_edges(frame)
    if region and region.confidence > 0.3:
        return region

    # Method 3: Color analysis
    region = detect_game_screen_by_color(frame)
    if region:
        return region

    # Fallback: assume game fills most of the frame
    h, w = frame.shape[:2]
    # Find largest region with GB aspect ratio
    if w / h > GB_ASPECT:
        # Frame is wider than GB, crop sides
        new_w = int(h * GB_ASPECT)
        x = (w - new_w) // 2
        return ScreenRegion(x, 0, new_w, h, 0.5)
    else:
        # Frame is taller than GB, crop top/bottom
        new_h = int(w / GB_ASPECT)
        y = (h - new_h) // 2
        return ScreenRegion(0, y, w, new_h, 0.5)


def is_gameplay_frame(frame: np.ndarray) -> bool:
    """
    Check if a frame contains actual gameplay (not intro/outro/pause).
    """
    # Check for mostly black frame (loading/transition)
    mean_brightness = np.mean(frame)
    if mean_brightness < 15:
        return False

    # Check for mostly white frame
    if mean_brightness > 245:
        return False

    # Check for enough color variation (not solid color)
    std_dev = np.std(frame)
    if std_dev < 10:
        return False

    return True


def load_template_frames(template_dir: Path, max_templates: int = 10) -> List[np.ndarray]:
    """Load template frames from existing dataset."""
    templates = []

    if not template_dir.exists():
        return templates

    # Load some frames from each video directory
    for video_dir in sorted(template_dir.iterdir())[:3]:
        if not video_dir.is_dir():
            continue

        npy_files = sorted(video_dir.glob("*.npy"))
        for npy_file in npy_files[100:100 + max_templates // 3]:
            try:
                frame = np.load(npy_file)
                if frame.shape == (144, 160, 3):
                    # Convert to BGR for OpenCV
                    templates.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            except:
                continue

    return templates


def process_video(
    video_path: Path,
    output_dir: Path,
    fps: float = 2.0,
    template_frames: Optional[List[np.ndarray]] = None,
    detect_region_interval: int = 300,  # Re-detect region every N frames
    crop_config: Optional[Dict] = None,
) -> dict:
    """
    Process a video file: detect game screen, crop, and extract frames.

    Returns dict with processing statistics.
    """
    video_id = video_path.stem

    # Check skip list first
    if crop_config:
        skip_reason = should_skip_video(video_id, crop_config)
        if skip_reason:
            print(f"Skipping {video_id}: {skip_reason}")
            return {'video_id': video_id, 'status': 'skipped', 'reason': skip_reason}

    frames_dir = output_dir / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    existing_frames = list(frames_dir.glob("*.npy"))
    if len(existing_frames) > 100:
        return {
            'video_id': video_id,
            'status': 'already_processed',
            'frames': len(existing_frames),
        }

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {'video_id': video_id, 'status': 'error', 'error': 'Cannot open video'}

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps))

    print(f"Processing {video_id}: {total_frames} frames @ {video_fps:.1f} fps")
    print(f"  Extracting every {frame_interval} frames ({fps} fps output)")

    # Try to get manually configured crop region first
    region = None
    if crop_config:
        region = get_video_crop_region(video_id, crop_config)
        if region:
            print(f"  Using configured crop: ({region.x}, {region.y}) {region.width}x{region.height}")

    # Fallback to auto-detection if no config
    if region is None:
        print("  No configured crop, using auto-detection...")
        for _ in range(10):
            ret, frame = cap.read()
            if not ret:
                break
            region = detect_game_screen(frame, template_frames)
            if region and region.confidence > 0.5:
                break

    if region is None:
        # Fallback to center crop
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            new_h = int(w / GB_ASPECT)
            if new_h > h:
                new_w = int(h * GB_ASPECT)
                region = ScreenRegion((w - new_w) // 2, 0, new_w, h, 0.3)
            else:
                region = ScreenRegion(0, (h - new_h) // 2, w, new_h, 0.3)

    print(f"  Screen region: ({region.x}, {region.y}) {region.width}x{region.height} (conf: {region.confidence:.2f})")

    # Reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Extract frames
    frame_idx = 0
    saved_count = 0
    skipped_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Periodically re-detect region (in case of scene changes)
            if saved_count > 0 and saved_count % detect_region_interval == 0:
                new_region = detect_game_screen(frame, template_frames)
                if new_region and new_region.confidence > region.confidence:
                    region = new_region

            # Crop to game screen
            try:
                cropped = region.crop(frame)
            except:
                frame_idx += 1
                continue

            # Check if it's gameplay
            if not is_gameplay_frame(cropped):
                skipped_count += 1
                frame_idx += 1
                continue

            # Convert BGR to RGB
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

            # Save as numpy array
            npy_path = frames_dir / f"frame_{saved_count:06d}.npy"
            np.save(npy_path, rgb.astype(np.uint8))
            saved_count += 1

            if saved_count % 1000 == 0:
                print(f"  Saved {saved_count} frames (skipped {skipped_count} non-gameplay)...")

        frame_idx += 1

    cap.release()

    # Save metadata
    metadata = {
        'video_id': video_id,
        'source_path': str(video_path),
        'frames': saved_count,
        'skipped': skipped_count,
        'region': {
            'x': region.x, 'y': region.y,
            'width': region.width, 'height': region.height,
            'confidence': region.confidence,
        },
        'fps': fps,
    }

    with open(frames_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Done: {saved_count} frames saved, {skipped_count} skipped")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Preprocess Pokemon videos")
    parser.add_argument("--input", type=str, help="Input video directory")
    parser.add_argument("--output", type=str, required=True, help="Output frames directory")
    parser.add_argument("--fps", type=float, default=2.0, help="Output frames per second")
    parser.add_argument("--templates", type=str, default="data/speedruns/frames",
                       help="Directory with template frames")
    parser.add_argument("--config", type=str, default="video_crop_config.json",
                       help="Crop configuration JSON file")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load crop configuration
    config_path = Path(args.config)
    crop_config = load_crop_config(config_path)
    usable_count = len([k for k in crop_config.get("usable_videos", {}) if not k.startswith("_")])
    skip_count = len(crop_config.get("skip_videos", {}))
    print(f"Loaded crop config: {usable_count} usable, {skip_count} skip")

    # Load template frames for fallback detection
    print("Loading template frames...")
    templates = load_template_frames(Path(args.templates))
    print(f"  Loaded {len(templates)} template frames")

    # Find video files - either from input dir or from videos referenced in config
    video_files = []
    if args.input:
        input_dir = Path(args.input)
        video_files = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.mkv")) + \
                      list(input_dir.glob("*.webm")) + list(input_dir.glob("*.avi"))
    else:
        # Search in common video locations for videos matching config
        search_dirs = [
            Path("data/speedruns/videos"),
            Path("data/archive_videos"),
            Path("data/letsplays"),
        ]
        for search_dir in search_dirs:
            if search_dir.exists():
                video_files.extend(search_dir.glob("*.mp4"))
                video_files.extend(search_dir.glob("*.webm"))

    print(f"\nFound {len(video_files)} videos to process")
    print("=" * 60)

    # Process videos
    results = []
    for video_path in sorted(video_files):
        result = process_video(video_path, output_dir, args.fps, templates, crop_config=crop_config)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_frames = sum(r.get('frames', 0) for r in results)
    processed = sum(1 for r in results if r.get('status') != 'error')

    print(f"Videos processed: {processed}/{len(video_files)}")
    print(f"Total frames: {total_frames:,}")
    print(f"Output directory: {output_dir}")

    # Save overall metadata
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump({
            'videos': len(video_files),
            'total_frames': total_frames,
            'fps': args.fps,
            'results': results,
        }, f, indent=2)


if __name__ == "__main__":
    main()
