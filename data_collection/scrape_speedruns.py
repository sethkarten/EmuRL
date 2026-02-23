#!/usr/bin/env python3
"""
Scrape Pokemon Red speedruns from speedrun.com and download videos.

Usage:
    # Fetch run metadata
    uv run python scrape_speedruns.py --fetch-runs --output data/speedrun_metadata.json

    # Download videos
    uv run python scrape_speedruns.py --download --metadata data/speedrun_metadata.json --output data/speedrun_videos

    # Extract frames
    uv run python scrape_speedruns.py --extract-frames --videos data/speedrun_videos --output data/speedruns/frames
"""

import requests
import json
import time
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import hashlib


# Pokemon Red/Blue game ID on speedrun.com
GAME_ID = "46w22l6r"

# Categories to scrape (prioritize glitchless for clean gameplay)
CATEGORIES = {
    "any_glitchless": "wk6oork1",
    "any_glitchless_classic": "02qo3wpk",
    "any_no_save_corruption": "jdr77526",
    "any": "02qvyl9d",
    "catch_em_all": "xd1r0ozk",
}

API_BASE = "https://www.speedrun.com/api/v1"


def fetch_runs(category_id: str, max_runs: int = 200, offset: int = 0) -> List[Dict]:
    """Fetch verified runs for a category."""
    runs = []

    while len(runs) < max_runs:
        url = f"{API_BASE}/runs"
        params = {
            "game": GAME_ID,
            "category": category_id,
            "status": "verified",
            "max": min(200, max_runs - len(runs)),
            "offset": offset + len(runs),
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            batch = data.get("data", [])
            if not batch:
                break

            runs.extend(batch)
            print(f"  Fetched {len(runs)} runs so far...")

            # Rate limiting
            time.sleep(0.5)

            # Check pagination
            pagination = data.get("pagination", {})
            if not pagination.get("links", []):
                break

        except Exception as e:
            print(f"  Error fetching runs: {e}")
            break

    return runs


def extract_video_url(run: Dict) -> Optional[str]:
    """Extract video URL from run data."""
    videos = run.get("videos")
    if videos is None:
        return None

    # Check for direct video links
    links = videos.get("links", []) or []
    for link in links:
        uri = link.get("uri", "")
        if "youtube.com" in uri or "youtu.be" in uri:
            return uri
        if "twitch.tv" in uri:
            return uri

    # Check for embedded video
    text = videos.get("text", "")
    if text and ("youtube" in text.lower() or "twitch" in text.lower()):
        return text

    return None


def fetch_all_runs(categories: Dict[str, str], max_per_category: int = 500) -> List[Dict]:
    """Fetch runs from all categories."""
    all_runs = []
    seen_videos = set()

    for cat_name, cat_id in categories.items():
        print(f"\nFetching {cat_name} runs...")
        runs = fetch_runs(cat_id, max_runs=max_per_category)

        for run in runs:
            video_url = extract_video_url(run)
            if video_url and video_url not in seen_videos:
                seen_videos.add(video_url)
                all_runs.append({
                    "id": run.get("id"),
                    "category": cat_name,
                    "time": run.get("times", {}).get("primary_t"),
                    "video_url": video_url,
                    "date": run.get("date"),
                    "player": run.get("players", [{}])[0].get("id", "unknown"),
                })

        print(f"  Found {len(runs)} runs, {len(seen_videos)} unique videos total")

    return all_runs


def download_video(video_url: str, output_dir: Path, video_id: str) -> Optional[Path]:
    """Download video using yt-dlp."""
    output_path = output_dir / f"{video_id}.mp4"

    if output_path.exists():
        print(f"  Video already exists: {output_path}")
        return output_path

    try:
        # Use yt-dlp to download with better options
        cmd = [
            "yt-dlp",
            "-f", "best[height<=720]/best",  # Fallback to best if 720p not available
            "-o", str(output_path),
            "--no-playlist",
            "--socket-timeout", "30",
            "--retries", "3",
            "--fragment-retries", "3",
            "--concurrent-fragments", "4",
            "--no-check-certificates",
            video_url
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)  # 15 min timeout

        if result.returncode == 0 and output_path.exists():
            print(f"  Downloaded: {output_path}")
            return output_path
        else:
            print(f"  Failed to download: {result.stderr[:300] if result.stderr else 'Unknown error'}")
            # Cleanup partial files
            for ext in ['.part', '.ytdl']:
                partial = output_dir / f"{video_id}.mp4{ext}"
                if partial.exists():
                    partial.unlink()
            return None

    except subprocess.TimeoutExpired:
        print(f"  Download timed out (15min): {video_url}")
        # Cleanup partial files
        for ext in ['.part', '.ytdl']:
            partial = output_dir / f"{video_id}.mp4{ext}"
            if partial.exists():
                partial.unlink()
        return None
    except Exception as e:
        print(f"  Error downloading: {e}")
        return None


def extract_frames(video_path: Path, output_dir: Path, fps: float = 2.0) -> int:
    """Extract frames from video at specified FPS."""
    video_id = video_path.stem
    frames_dir = output_dir / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    existing_frames = list(frames_dir.glob("*.png"))
    if len(existing_frames) > 100:
        print(f"  Frames already extracted: {len(existing_frames)} frames")
        return len(existing_frames)

    try:
        # Use ffmpeg to extract frames
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", "2",
            str(frames_dir / "frame_%06d.png"),
            "-y"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        if result.returncode == 0:
            num_frames = len(list(frames_dir.glob("*.png")))
            print(f"  Extracted {num_frames} frames to {frames_dir}")
            return num_frames
        else:
            print(f"  FFmpeg error: {result.stderr[:200]}")
            return 0

    except Exception as e:
        print(f"  Error extracting frames: {e}")
        return 0


def crop_gameboy_frame(frame_path: Path, output_path: Path) -> bool:
    """Crop frame to Game Boy screen area (144x160)."""
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(frame_path)
        arr = np.array(img)

        # Game Boy resolution is 160x144 (width x height)
        # Videos may have borders, so we need to detect the game area

        # For now, try to find the game area by looking for consistent aspect ratio
        h, w = arr.shape[:2]

        # Target aspect ratio: 160/144 = 1.11
        target_ratio = 160 / 144

        # Try center crop with correct aspect ratio
        if w / h > target_ratio:
            # Width is too wide, crop horizontally
            new_w = int(h * target_ratio)
            start_x = (w - new_w) // 2
            arr = arr[:, start_x:start_x+new_w]
        else:
            # Height is too tall, crop vertically
            new_h = int(w / target_ratio)
            start_y = (h - new_h) // 2
            arr = arr[start_y:start_y+new_h, :]

        # Resize to 160x144
        img_cropped = Image.fromarray(arr)
        img_resized = img_cropped.resize((160, 144), Image.Resampling.LANCZOS)

        img_resized.save(output_path)
        return True

    except Exception as e:
        print(f"  Error cropping frame: {e}")
        return False


def process_frames_to_npy(frames_dir: Path, output_dir: Path) -> int:
    """Convert PNG frames to NPY format with proper cropping."""
    import numpy as np
    from PIL import Image

    video_id = frames_dir.name
    npy_dir = output_dir / video_id
    npy_dir.mkdir(parents=True, exist_ok=True)

    # Create index file
    index = {"video_id": video_id, "frames": []}

    frame_files = sorted(frames_dir.glob("frame_*.png"))
    processed = 0

    for i, frame_path in enumerate(frame_files):
        try:
            img = Image.open(frame_path)
            arr = np.array(img)

            # Resize to 160x144 if needed
            if arr.shape[:2] != (144, 160):
                img_resized = img.resize((160, 144), Image.Resampling.LANCZOS)
                arr = np.array(img_resized)

            # Ensure RGB
            if len(arr.shape) == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.shape[2] == 4:
                arr = arr[:, :, :3]

            # Save as NPY
            npy_path = npy_dir / f"frame_{i:06d}.npy"
            np.save(npy_path, arr.astype(np.uint8))

            # Add to index
            index["frames"].append({
                "path": f"{video_id}/frame_{i:06d}.npy",
                "frame_idx": i,
                "timestamp": i / 2.0,  # Assuming 2 FPS
            })

            processed += 1

        except Exception as e:
            print(f"  Error processing {frame_path}: {e}")

    # Save index
    with open(output_dir / f"{video_id}_index.json", "w") as f:
        json.dump(index, f, indent=2)

    return processed


def main():
    parser = argparse.ArgumentParser(description="Scrape Pokemon Red speedruns")
    parser.add_argument("--fetch-runs", action="store_true", help="Fetch run metadata from speedrun.com")
    parser.add_argument("--download", action="store_true", help="Download videos")
    parser.add_argument("--extract-frames", action="store_true", help="Extract frames from videos")
    parser.add_argument("--process-npy", action="store_true", help="Convert frames to NPY format")
    parser.add_argument("--metadata", type=str, default="data/speedrun_metadata.json", help="Metadata file")
    parser.add_argument("--videos", type=str, default="data/speedrun_videos", help="Video directory")
    parser.add_argument("--frames", type=str, default="data/speedrun_frames", help="Frames directory")
    parser.add_argument("--output", type=str, default="data/speedruns/frames", help="Final output directory")
    parser.add_argument("--max-videos", type=int, default=50, help="Max videos to download")
    parser.add_argument("--max-per-category", type=int, default=200, help="Max runs per category")
    args = parser.parse_args()

    if args.fetch_runs:
        print("=" * 60)
        print("Fetching Pokemon Red speedrun metadata")
        print("=" * 60)

        runs = fetch_all_runs(CATEGORIES, max_per_category=args.max_per_category)

        # Sort by time (fastest first)
        runs.sort(key=lambda x: x.get("time", float("inf")))

        # Save metadata
        metadata_path = Path(args.metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, "w") as f:
            json.dump(runs, f, indent=2)

        print(f"\nSaved {len(runs)} runs to {metadata_path}")

        # Print summary
        print(f"\nSummary:")
        for cat in CATEGORIES:
            count = sum(1 for r in runs if r["category"] == cat)
            print(f"  {cat}: {count} runs")

    if args.download:
        print("=" * 60)
        print("Downloading speedrun videos")
        print("=" * 60)

        # Load metadata
        with open(args.metadata) as f:
            runs = json.load(f)

        videos_dir = Path(args.videos)
        videos_dir.mkdir(parents=True, exist_ok=True)

        # Download videos (prioritize glitchless categories)
        priority_order = ["any_glitchless", "any_glitchless_classic", "any_no_save_corruption", "catch_em_all", "any"]
        runs.sort(key=lambda x: (priority_order.index(x["category"]) if x["category"] in priority_order else 99, x.get("time", float("inf"))))

        downloaded = 0
        for run in runs[:args.max_videos]:
            video_url = run["video_url"]
            video_id = hashlib.md5(video_url.encode()).hexdigest()[:8]

            print(f"\n[{downloaded+1}/{args.max_videos}] {run['category']} - {run.get('time', '?')}s")
            print(f"  URL: {video_url}")

            result = download_video(video_url, videos_dir, video_id)
            if result:
                downloaded += 1
                run["local_video"] = str(result)

        # Save updated metadata
        with open(args.metadata, "w") as f:
            json.dump(runs, f, indent=2)

        print(f"\nDownloaded {downloaded} videos")

    if args.extract_frames:
        print("=" * 60)
        print("Extracting frames from videos")
        print("=" * 60)

        videos_dir = Path(args.videos)
        frames_dir = Path(args.frames)
        frames_dir.mkdir(parents=True, exist_ok=True)

        for video_path in sorted(videos_dir.glob("*.mp4")):
            print(f"\nProcessing: {video_path}")
            extract_frames(video_path, frames_dir, fps=2.0)

    if args.process_npy:
        print("=" * 60)
        print("Converting frames to NPY format")
        print("=" * 60)

        frames_dir = Path(args.frames)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        for video_dir in sorted(frames_dir.iterdir()):
            if video_dir.is_dir():
                print(f"\nProcessing: {video_dir}")
                process_frames_to_npy(video_dir, output_dir)


if __name__ == "__main__":
    main()
