#!/usr/bin/env python3
"""Download Pokemon Red speedrun videos from YouTube.

Run directly (not through Claude sandbox):
    python download_youtube_videos.py
"""
import json
import subprocess
import sys
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("/mnt/storage/datasets/pokemon_red_youtube")
METADATA_FILE = Path("data/speedrun_metadata.json")
MAX_VIDEOS = 10


def get_youtube_videos():
    """Get YouTube video URLs from metadata, sorted by date (newest first)."""
    with open(METADATA_FILE) as f:
        data = json.load(f)

    videos = []
    for run in data:
        url = run.get('video_url', '')
        if 'youtube.com' in url or 'youtu.be' in url:
            videos.append({
                'id': run['id'][:8],
                'date': run.get('date', ''),
                'time': run.get('time', 0),
                'url': url
            })

    # Sort by date descending (newest first)
    videos.sort(key=lambda x: x['date'], reverse=True)

    # Filter out very short runs (likely glitch runs < 2 min)
    videos = [v for v in videos if v['time'] > 120]

    return videos


def download_video(url: str, output_path: Path) -> bool:
    """Download a single video using yt-dlp."""
    cmd = [
        "yt-dlp",
        "--no-check-certificates",
        # Use android client which works better with SABR restrictions
        "--extractor-args", "youtube:player_client=android",
        # Fallback format selection - try multiple options
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "--merge-output-format", "mp4",
        "-o", str(output_path),
        "--no-playlist",
        url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            return True
        else:
            # Show more of the error
            stderr = result.stderr.strip()
            # Find the actual ERROR line
            for line in stderr.split('\n'):
                if 'ERROR' in line:
                    print(f"  Error: {line[:300]}")
                    break
            else:
                print(f"  Error: {stderr[-500:]}")
            return False
    except subprocess.TimeoutExpired:
        print("  Timeout (1 hour)")
        return False
    except Exception as e:
        print(f"  Exception: {e}")
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = get_youtube_videos()
    print(f"Found {len(videos)} YouTube videos (filtered for runs > 2 min)")

    # Check which we already have
    existing = set(f.stem.replace('yt_', '') for f in OUTPUT_DIR.glob("*.mp4"))

    downloaded = 0
    skipped = 0
    for v in videos:
        if downloaded >= MAX_VIDEOS:
            break

        vid_id = v['id']
        output_path = OUTPUT_DIR / f"yt_{vid_id}.mp4"

        if vid_id in existing or f"yt_{vid_id}" in existing:
            print(f"[skip] {vid_id} - already exists")
            skipped += 1
            continue

        print(f"[{downloaded+1}/{MAX_VIDEOS}] Downloading {vid_id}...")
        print(f"  URL: {v['url']}")
        print(f"  Date: {v['date']}, Duration: {v['time']:.0f}s ({v['time']/60:.1f} min)")

        if download_video(v['url'], output_path):
            print(f"  Success! Saved to {output_path}")
            downloaded += 1
        else:
            print(f"  Failed, trying next...")

    print(f"\nDownloaded {downloaded} videos to {OUTPUT_DIR}")
    print(f"Skipped {skipped} existing videos")


if __name__ == "__main__":
    main()
