#!/usr/bin/env python3
"""Download Pokemon Red Let's Play playlist from YouTube.

Run directly:
    python download_letsplay_playlist.py
"""
import subprocess
import sys
from pathlib import Path

# Configuration
PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLR2FYtFWHTrWr4lpB9zgdfx0yno9FRz3S"
OUTPUT_DIR = Path("/mnt/storage/datasets/pokemon_red_letsplay")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Let's Play playlist to {OUTPUT_DIR}")
    print(f"Playlist: {PLAYLIST_URL}")
    print()

    cmd = [
        "yt-dlp",
        "--no-check-certificates",
        # Use android client for better compatibility
        "--extractor-args", "youtube:player_client=android",
        # Format selection
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "--merge-output-format", "mp4",
        # Output template with episode number
        "-o", str(OUTPUT_DIR / "ep_%(playlist_index)02d_%(id)s.%(ext)s"),
        # Download full playlist
        "--yes-playlist",
        # Show progress
        "--progress",
        # Continue from where we left off
        "--download-archive", str(OUTPUT_DIR / "downloaded.txt"),
        PLAYLIST_URL
    ]

    print("Running:", " ".join(cmd[:8]) + " ...")
    print()

    try:
        # Run with live output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='')
        process.wait()

        if process.returncode == 0:
            print("\nPlaylist download complete!")
        else:
            print(f"\nDownload finished with return code {process.returncode}")

    except KeyboardInterrupt:
        print("\nDownload interrupted. Run again to resume.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
