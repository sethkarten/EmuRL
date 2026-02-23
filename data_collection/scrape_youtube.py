#!/usr/bin/env python3
"""
Scrape Pokemon Red Let's Play videos from YouTube for training data diversity.

Let's Plays provide diverse gameplay that speedruns skip:
- Full exploration of areas
- All trainer battles
- Menu navigation
- Pokemon catching/evolution
- Story sequences

Usage:
    # Search and download videos
    uv run python scrape_youtube.py --search "pokemon red let's play" --max-videos 50

    # Download from a playlist
    uv run python scrape_youtube.py --playlist "PLxxxxxxx" --max-videos 20

    # Download specific channels known for Pokemon content
    uv run python scrape_youtube.py --channels --max-videos 100
"""

import subprocess
import argparse
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
import time


# Search queries for diverse Pokemon Red gameplay
SEARCH_QUERIES = [
    "pokemon red let's play full game",
    "pokemon red walkthrough",
    "pokemon red playthrough",
    "pokemon red longplay",
    "pokemon red complete playthrough",
    "pokemon red gameplay no commentary",
    "pokemon blue let's play",  # Same visuals
    "pokemon red nuzlocke",
    "pokemon red randomizer",
]

# Known good channels/playlists for Pokemon content
KNOWN_PLAYLISTS = [
    # Add playlist IDs here if you find good ones
]


def search_youtube(query: str, max_results: int = 20) -> List[Dict]:
    """Search YouTube for videos matching query."""
    print(f"Searching: {query}")

    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-json",
        f"ytsearch{max_results}:{query}",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        videos = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    data = json.loads(line)
                    videos.append({
                        'id': data.get('id'),
                        'title': data.get('title'),
                        'url': data.get('url') or f"https://www.youtube.com/watch?v={data.get('id')}",
                        'duration': data.get('duration'),
                        'channel': data.get('channel'),
                        'query': query,
                    })
                except json.JSONDecodeError:
                    continue

        return videos

    except Exception as e:
        print(f"  Error searching: {e}")
        return []


def filter_videos(videos: List[Dict], min_duration: int = 1800, max_duration: int = 36000) -> List[Dict]:
    """
    Filter videos by duration and relevance.

    Args:
        min_duration: Minimum video length in seconds (default 30 min)
        max_duration: Maximum video length in seconds (default 10 hours)
    """
    filtered = []
    seen_ids = set()

    for video in videos:
        vid_id = video.get('id')
        duration = video.get('duration', 0) or 0
        title = (video.get('title') or '').lower()

        # Skip duplicates
        if vid_id in seen_ids:
            continue
        seen_ids.add(vid_id)

        # Duration filter
        if duration < min_duration or duration > max_duration:
            continue

        # Skip likely non-gameplay content
        skip_keywords = ['review', 'trailer', 'music', 'ost', 'soundtrack', 'remix',
                        'tier list', 'ranking', 'theory', 'analysis', 'explained',
                        'speedrun', 'world record', 'wr ', ' wr']
        if any(kw in title for kw in skip_keywords):
            continue

        # Prefer content with these keywords
        prefer_keywords = ['let\'s play', 'playthrough', 'walkthrough', 'longplay',
                          'full game', 'part 1', 'episode 1', 'ep 1', 'ep. 1']
        video['score'] = sum(1 for kw in prefer_keywords if kw in title)

        filtered.append(video)

    # Sort by score (higher = more relevant)
    filtered.sort(key=lambda x: x.get('score', 0), reverse=True)

    return filtered


def download_video(video_url: str, output_dir: Path, video_id: str) -> Optional[Path]:
    """Download video using yt-dlp."""
    output_path = output_dir / f"{video_id}.mp4"

    if output_path.exists():
        print(f"  Video already exists: {output_path}")
        return output_path

    try:
        cmd = [
            "yt-dlp",
            "-f", "best[height<=720]/best",  # 720p max for reasonable file sizes
            "-o", str(output_path),
            "--no-playlist",
            "--socket-timeout", "30",
            "--retries", "3",
            "--fragment-retries", "3",
            "--concurrent-fragments", "4",
            "--no-check-certificates",
            video_url
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout

        if result.returncode == 0 and output_path.exists():
            print(f"  Downloaded: {output_path}")
            return output_path
        else:
            print(f"  Failed: {result.stderr[:200] if result.stderr else 'Unknown error'}")
            # Cleanup partial files
            for ext in ['.part', '.ytdl']:
                partial = output_dir / f"{video_id}.mp4{ext}"
                if partial.exists():
                    partial.unlink()
            return None

    except subprocess.TimeoutExpired:
        print(f"  Download timed out (30min)")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def search_all_queries(max_per_query: int = 10) -> List[Dict]:
    """Search all predefined queries and combine results."""
    all_videos = []

    for query in SEARCH_QUERIES:
        videos = search_youtube(query, max_results=max_per_query)
        all_videos.extend(videos)
        print(f"  Found {len(videos)} videos")
        time.sleep(1)  # Rate limiting

    return all_videos


def main():
    parser = argparse.ArgumentParser(description="Scrape Pokemon Red Let's Plays from YouTube")
    parser.add_argument("--search", type=str, help="Custom search query")
    parser.add_argument("--channels", action="store_true", help="Search predefined queries")
    parser.add_argument("--output", type=str, default="/mnt/storage/datasets/pokemon_red_letsplays",
                       help="Output directory")
    parser.add_argument("--metadata", type=str, default="data/letsplay_metadata.json",
                       help="Metadata output file")
    parser.add_argument("--max-videos", type=int, default=50, help="Max videos to download")
    parser.add_argument("--max-per-query", type=int, default=15, help="Max results per search query")
    parser.add_argument("--min-duration", type=int, default=1800, help="Min video duration (seconds)")
    parser.add_argument("--download", action="store_true", help="Download videos")
    parser.add_argument("--search-only", action="store_true", help="Only search, don't download")
    args = parser.parse_args()

    output_dir = Path(args.output)
    videos_dir = output_dir / "videos"
    metadata_path = Path(args.metadata)

    # Search for videos
    all_videos = []

    if args.search:
        print(f"\n{'='*60}")
        print(f"Searching YouTube: {args.search}")
        print(f"{'='*60}")
        videos = search_youtube(args.search, max_results=args.max_per_query * 3)
        all_videos.extend(videos)

    if args.channels or not args.search:
        print(f"\n{'='*60}")
        print("Searching predefined queries for Pokemon Red content")
        print(f"{'='*60}")
        videos = search_all_queries(max_per_query=args.max_per_query)
        all_videos.extend(videos)

    # Filter videos
    print(f"\nFiltering {len(all_videos)} videos...")
    filtered = filter_videos(all_videos, min_duration=args.min_duration)
    print(f"  {len(filtered)} videos passed filters")

    # Save metadata
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(filtered, f, indent=2)
    print(f"\nSaved metadata to {metadata_path}")

    # Print summary
    print(f"\nTop videos found:")
    for i, video in enumerate(filtered[:10]):
        duration_min = (video.get('duration') or 0) // 60
        print(f"  {i+1}. [{duration_min}min] {video.get('title', 'Unknown')[:60]}")

    if args.search_only:
        return

    # Download videos
    if args.download or not args.search_only:
        print(f"\n{'='*60}")
        print(f"Downloading videos to {videos_dir}")
        print(f"{'='*60}")

        videos_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        for i, video in enumerate(filtered[:args.max_videos]):
            video_url = video.get('url')
            if not video_url:
                continue

            video_id = hashlib.md5(video_url.encode()).hexdigest()[:8]

            print(f"\n[{downloaded+1}/{args.max_videos}] {video.get('title', 'Unknown')[:50]}")
            print(f"  URL: {video_url}")

            result = download_video(video_url, videos_dir, video_id)
            if result:
                downloaded += 1
                video['local_video'] = str(result)
                video['video_id'] = video_id

        # Update metadata with local paths
        with open(metadata_path, 'w') as f:
            json.dump(filtered, f, indent=2)

        print(f"\nDownloaded {downloaded} videos")


if __name__ == "__main__":
    main()
