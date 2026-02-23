#!/usr/bin/env python3
"""Download Twitch speedrun videos using yt-dlp library.

Run outside sandbox: python download_twitch_videos.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("data/speedruns/videos_new")
METADATA_FILE = Path("data/speedrun_metadata.json")
MAX_VIDEOS = 10


def get_recent_twitch_videos():
    """Get recent Twitch video URLs from metadata."""
    with open(METADATA_FILE) as f:
        data = json.load(f)

    videos = []
    for run in data:
        url = run.get('video_url', '')
        if 'twitch.tv/videos' in url:
            date = run.get('date', '')
            # Only get 2024-2025 videos (more likely to still exist)
            if date.startswith('2024') or date.startswith('2025'):
                videos.append({
                    'id': run['id'][:8],
                    'date': date,
                    'time': run.get('time', 0),
                    'url': url
                })

    # Sort by date descending (newest first)
    videos.sort(key=lambda x: x['date'], reverse=True)
    return videos


def download_video(url: str, output_path: Path) -> bool:
    """Download a single video using yt-dlp subprocess."""
    cmd = [
        "yt-dlp",
        "--no-check-certificates",
        "-f", "best[height<=720]",  # Limit to 720p to save space
        "-o", str(output_path),
        url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            return True
        else:
            print(f"  Error: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print("  Timeout (1 hour)")
        return False
    except Exception as e:
        print(f"  Exception: {e}")
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = get_recent_twitch_videos()
    print(f"Found {len(videos)} recent Twitch videos")

    # Check which we already have
    existing = set(f.stem.replace('video_', '') for f in OUTPUT_DIR.glob("*.mp4"))

    downloaded = 0
    for v in videos[:MAX_VIDEOS]:
        vid_id = v['id']
        output_path = OUTPUT_DIR / f"video_{vid_id}.mp4"

        if vid_id in existing:
            print(f"[{downloaded+1}/{MAX_VIDEOS}] {vid_id} - already exists, skipping")
            downloaded += 1
            continue

        print(f"[{downloaded+1}/{MAX_VIDEOS}] Downloading {vid_id}...")
        print(f"  URL: {v['url']}")
        print(f"  Date: {v['date']}, Duration: {v['time']:.0f}s")

        if download_video(v['url'], output_path):
            print(f"  Success! Saved to {output_path}")
            downloaded += 1
        else:
            print(f"  Failed")

        if downloaded >= MAX_VIDEOS:
            break

    print(f"\nDownloaded {downloaded} videos to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
