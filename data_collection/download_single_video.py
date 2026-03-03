#!/usr/bin/env python3
"""Download a single YouTube video using yt-dlp.

Run directly:
    python download_single_video.py <url> [--output-dir DIR] [--max-height 720]

Example:
    python download_single_video.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
"""
import argparse
import subprocess
import re
from pathlib import Path

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "videos"
DEFAULT_MAX_HEIGHT = 720
DEFAULT_URL = "https://www.youtube.com/watch?v=Rs6wk0oGUmU"


def extract_video_id(url: str) -> str | None:
    """Return the YouTube video ID from a watch or short URL, or None if not found.

    Supports:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
    """
    patterns = [
        r"[?&]v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"shorts/([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def resolve_output_path(url: str, output_dir: Path) -> Path:
    """Return the full output path for a video given its URL and output directory.

    File name format: ``<video_id>.mp4``.  Falls back to ``video.mp4`` when the
    ID cannot be parsed from the URL.
    """
    video_id = extract_video_id(url)
    filename = f"{video_id}.mp4" if video_id else "video.mp4"
    return output_dir / filename


def build_download_command(url: str, output_path: Path, max_height: int = DEFAULT_MAX_HEIGHT) -> list[str]:
    """Build the yt-dlp command list for downloading a single video.

    Parameters
    ----------
    url:
        Full YouTube video URL.
    output_path:
        Absolute (or relative) path where the file should be saved.
    max_height:
        Maximum video height in pixels (e.g. 720, 1080).  Used to build the
        format selection string.

    Returns
    -------
    list[str]
        The argv list ready to be passed to :func:`subprocess.run`.
    """
    format_str = f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best"
    return [
        "yt-dlp",
        "--no-check-certificates",
        # Use android client for better compatibility
        "--extractor-args", "youtube:player_client=android",
        "-f", format_str,
        "--merge-output-format", "mp4",
        "-o", str(output_path),
        "--no-playlist",
        url,
    ]


def download_video(url: str, output_path: Path, max_height: int = DEFAULT_MAX_HEIGHT, timeout: int = 3600) -> bool:
    """Download a single video using yt-dlp.

    Parameters
    ----------
    url:
        Full YouTube video URL.
    output_path:
        Destination file path (parent directory must exist).
    max_height:
        Maximum video height in pixels.
    timeout:
        Seconds before the subprocess is abandoned.

    Returns
    -------
    bool
        ``True`` on success, ``False`` on any failure.
    """
    cmd = build_download_command(url, output_path, max_height)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return True
        stderr = result.stderr.strip()
        for line in stderr.splitlines():
            if "ERROR" in line:
                print(f"  Error: {line[:300]}")
                break
        else:
            print(f"  Error: {stderr[-500:]}")
        return False
    except subprocess.TimeoutExpired:
        print(f"  Timeout ({timeout}s)")
        return False
    except Exception as exc:
        print(f"  Exception: {exc}")
        return False


def main(argv: list[str] | None = None) -> int:
    """Entry-point for CLI usage.

    Returns the exit code (0 = success, 1 = failure).
    """
    parser = argparse.ArgumentParser(description="Download a single YouTube video.")
    parser.add_argument("url", nargs="?", default=DEFAULT_URL, help=f"YouTube video URL (default: {DEFAULT_URL})")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the video (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=DEFAULT_MAX_HEIGHT,
        help=f"Maximum video height in pixels (default: {DEFAULT_MAX_HEIGHT})",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = resolve_output_path(args.url, args.output_dir)

    print(f"Downloading: {args.url}")
    print(f"Saving to:   {output_path}")
    print()

    success = download_video(args.url, output_path, max_height=args.max_height)

    if success:
        print(f"\nDownload complete: {output_path}")
        return 0
    else:
        print("\nDownload failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
