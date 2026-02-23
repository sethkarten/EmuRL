#!/usr/bin/env python3
"""
Scrape Pokemon Red gameplay videos from archive.org

Archive.org has a large collection of gaming content that's easy to download.
This provides diverse gameplay coverage without YouTube's bot detection.

Usage:
    uv run python scrape_archive.py --search --max-videos 30
    uv run python scrape_archive.py --download --max-videos 30
"""

import requests
import subprocess
import argparse
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
import time


def search_archive(query: str = "pokemon red", max_results: int = 50) -> List[Dict]:
    """Search archive.org for Pokemon gameplay videos."""
    print(f"Searching archive.org for: {query}")

    # Archive.org search API
    url = "https://archive.org/advancedsearch.php"
    params = {
        'q': f'{query} AND mediatype:movies',
        'fl[]': ['identifier', 'title', 'description', 'downloads', 'item_size'],
        'sort[]': 'downloads desc',
        'rows': max_results,
        'page': 1,
        'output': 'json',
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        videos = []
        for doc in data.get('response', {}).get('docs', []):
            identifier = doc.get('identifier', '')
            title = doc.get('title', '')

            # Filter for relevant content
            title_lower = title.lower()
            if not any(kw in title_lower for kw in ['pokemon', 'pokémon', 'pocket monster']):
                continue

            videos.append({
                'id': identifier,
                'title': title,
                'description': doc.get('description', ''),
                'downloads': doc.get('downloads', 0),
                'size': doc.get('item_size', 0),
                'url': f"https://archive.org/details/{identifier}",
            })

        return videos

    except Exception as e:
        print(f"  Error searching: {e}")
        return []


def get_video_files(identifier: str) -> List[Dict]:
    """Get downloadable video files for an archive.org item."""
    url = f"https://archive.org/metadata/{identifier}"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        files = []
        for f in data.get('files', []):
            name = f.get('name', '')
            # Look for video files
            if name.endswith(('.mp4', '.ogv', '.avi', '.mkv', '.webm')):
                files.append({
                    'name': name,
                    'size': int(f.get('size', 0)),
                    'format': f.get('format', ''),
                    'url': f"https://archive.org/download/{identifier}/{name}",
                })

        # Sort by size (prefer smaller files for manageable downloads)
        files.sort(key=lambda x: x['size'])

        return files

    except Exception as e:
        print(f"  Error getting files: {e}")
        return []


def download_video(url: str, output_path: Path) -> bool:
    """Download a video file."""
    if output_path.exists():
        print(f"  Already exists: {output_path}")
        return True

    try:
        cmd = [
            "wget",
            "-q", "--show-progress",
            "-O", str(output_path),
            "--timeout=60",
            "--tries=3",
            url
        ]

        result = subprocess.run(cmd, timeout=1800)  # 30 min timeout

        if result.returncode == 0 and output_path.exists():
            print(f"  Downloaded: {output_path}")
            return True
        else:
            print(f"  Failed to download")
            if output_path.exists():
                output_path.unlink()
            return False

    except subprocess.TimeoutExpired:
        print(f"  Download timed out")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def search_multiple_queries() -> List[Dict]:
    """Search multiple queries for diverse content."""
    queries = [
        "pokemon red longplay",
        "pokemon blue longplay",
        "pokemon red gameplay",
        "pokemon red walkthrough",
        "pokemon gameboy",
        "pokemon red complete",
    ]

    all_videos = []
    seen_ids = set()

    for query in queries:
        videos = search_archive(query, max_results=20)
        for v in videos:
            if v['id'] not in seen_ids:
                seen_ids.add(v['id'])
                all_videos.append(v)
        time.sleep(0.5)  # Rate limiting

    return all_videos


def main():
    parser = argparse.ArgumentParser(description="Scrape Pokemon videos from archive.org")
    parser.add_argument("--search", action="store_true", help="Search for videos")
    parser.add_argument("--download", action="store_true", help="Download videos")
    parser.add_argument("--output", type=str, default="/mnt/storage/datasets/pokemon_red_archive",
                       help="Output directory")
    parser.add_argument("--metadata", type=str, default="data/archive_metadata.json",
                       help="Metadata file")
    parser.add_argument("--max-videos", type=int, default=30, help="Max videos to download")
    parser.add_argument("--max-size-gb", type=float, default=2.0, help="Max file size in GB")
    args = parser.parse_args()

    output_dir = Path(args.output)
    videos_dir = output_dir / "videos"
    metadata_path = Path(args.metadata)

    if args.search or not metadata_path.exists():
        print("=" * 60)
        print("Searching archive.org for Pokemon Red content")
        print("=" * 60)

        videos = search_multiple_queries()
        print(f"\nFound {len(videos)} unique items")

        # Save metadata
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(videos, f, indent=2)

        # Print summary
        print("\nTop items by downloads:")
        for v in sorted(videos, key=lambda x: x.get('downloads', 0), reverse=True)[:10]:
            print(f"  [{v.get('downloads', 0)} downloads] {v['title'][:60]}")

    if args.download:
        print("\n" + "=" * 60)
        print(f"Downloading videos to {videos_dir}")
        print("=" * 60)

        # Load metadata
        with open(metadata_path) as f:
            videos = json.load(f)

        videos_dir.mkdir(parents=True, exist_ok=True)
        max_size = int(args.max_size_gb * 1024 * 1024 * 1024)

        downloaded = 0
        for video in videos[:args.max_videos * 2]:  # Try more items in case some fail
            if downloaded >= args.max_videos:
                break

            print(f"\n[{downloaded+1}/{args.max_videos}] {video['title'][:50]}")

            # Get video files
            files = get_video_files(video['id'])
            if not files:
                print("  No video files found")
                continue

            # Find best file (prefer MP4, under size limit)
            best_file = None
            for f in files:
                if f['size'] > max_size:
                    continue
                if f['name'].endswith('.mp4'):
                    best_file = f
                    break
                if best_file is None:
                    best_file = f

            if not best_file:
                print(f"  No suitable files (all > {args.max_size_gb}GB)")
                continue

            # Download
            video_id = hashlib.md5(video['id'].encode()).hexdigest()[:8]
            ext = Path(best_file['name']).suffix
            output_path = videos_dir / f"{video_id}{ext}"

            print(f"  File: {best_file['name']} ({best_file['size'] / 1024 / 1024:.1f}MB)")

            if download_video(best_file['url'], output_path):
                downloaded += 1
                video['local_video'] = str(output_path)
                video['video_id'] = video_id

        # Update metadata
        with open(metadata_path, 'w') as f:
            json.dump(videos, f, indent=2)

        print(f"\nDownloaded {downloaded} videos")


if __name__ == "__main__":
    main()
