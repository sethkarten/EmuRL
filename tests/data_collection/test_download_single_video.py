"""
Unit tests for data_collection/download_single_video.py

Tests cover the pure, side-effect-free functions:
- extract_video_id: parse a YouTube video ID from various URL formats
- resolve_output_path: build the destination file path
- build_download_command: construct the yt-dlp argv list

And, using mocking:
- download_video: subprocess interaction
- main: CLI argument parsing and exit code
"""
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_collection.download_single_video import (
    DEFAULT_MAX_HEIGHT,
    build_download_command,
    download_video,
    extract_video_id,
    main,
    resolve_output_path,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WATCH_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
_SHORT_URL = "https://youtu.be/dQw4w9WgXcQ"
_SHORTS_URL = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
_VIDEO_ID = "dQw4w9WgXcQ"


# ---------------------------------------------------------------------------
# extract_video_id
# ---------------------------------------------------------------------------


class TestExtractVideoId:
    def test_watch_url(self):
        assert extract_video_id(_WATCH_URL) == _VIDEO_ID

    def test_short_url(self):
        assert extract_video_id(_SHORT_URL) == _VIDEO_ID

    def test_shorts_url(self):
        assert extract_video_id(_SHORTS_URL) == _VIDEO_ID

    def test_watch_url_with_extra_params(self):
        url = f"https://www.youtube.com/watch?v={_VIDEO_ID}&t=42s&list=PLfoo"
        assert extract_video_id(url) == _VIDEO_ID

    def test_returns_none_for_non_youtube_url(self):
        assert extract_video_id("https://example.com/video") is None

    def test_returns_none_for_empty_string(self):
        assert extract_video_id("") is None

    def test_returns_none_for_playlist_only_url(self):
        # Playlist URLs without a video ID should return None
        assert extract_video_id("https://www.youtube.com/playlist?list=PLabc123") is None


# ---------------------------------------------------------------------------
# resolve_output_path
# ---------------------------------------------------------------------------


class TestResolveOutputPath:
    def test_uses_video_id_as_filename(self, tmp_path):
        path = resolve_output_path(_WATCH_URL, tmp_path)
        assert path == tmp_path / f"{_VIDEO_ID}.mp4"

    def test_short_url_produces_same_filename(self, tmp_path):
        assert resolve_output_path(_SHORT_URL, tmp_path) == resolve_output_path(_WATCH_URL, tmp_path)

    def test_fallback_filename_for_unrecognised_url(self, tmp_path):
        path = resolve_output_path("https://example.com/video", tmp_path)
        assert path == tmp_path / "video.mp4"

    def test_output_dir_is_embedded_in_path(self, tmp_path):
        sub = tmp_path / "sub" / "dir"
        path = resolve_output_path(_WATCH_URL, sub)
        assert path.parent == sub


# ---------------------------------------------------------------------------
# build_download_command
# ---------------------------------------------------------------------------


class TestBuildDownloadCommand:
    def _cmd(self, url=_WATCH_URL, output_path=Path("/tmp/out.mp4"), max_height=DEFAULT_MAX_HEIGHT):
        return build_download_command(url, output_path, max_height)

    def test_starts_with_yt_dlp(self):
        assert self._cmd()[0] == "yt-dlp"

    def test_contains_url(self):
        assert _WATCH_URL in self._cmd()

    def test_contains_no_playlist_flag(self):
        assert "--no-playlist" in self._cmd()

    def test_contains_android_extractor_arg(self):
        cmd = self._cmd()
        idx = cmd.index("--extractor-args")
        assert "android" in cmd[idx + 1]

    def test_output_path_is_included(self):
        path = Path("/some/path/video.mp4")
        cmd = build_download_command(_WATCH_URL, path)
        assert str(path) in cmd

    def test_default_max_height_in_format(self):
        cmd = self._cmd()
        format_str = cmd[cmd.index("-f") + 1]
        assert f"height<={DEFAULT_MAX_HEIGHT}" in format_str

    def test_custom_max_height_in_format(self):
        cmd = self._cmd(max_height=1080)
        format_str = cmd[cmd.index("-f") + 1]
        assert "height<=1080" in format_str

    def test_merge_output_format_is_mp4(self):
        cmd = self._cmd()
        idx = cmd.index("--merge-output-format")
        assert cmd[idx + 1] == "mp4"

    def test_returns_list_of_strings(self):
        cmd = self._cmd()
        assert isinstance(cmd, list)
        assert all(isinstance(item, str) for item in cmd)


# ---------------------------------------------------------------------------
# download_video  (subprocess mocked)
# ---------------------------------------------------------------------------


class TestDownloadVideo:
    def _run(self, returncode=0, stderr="", **kwargs):
        mock_result = MagicMock()
        mock_result.returncode = returncode
        mock_result.stderr = stderr
        with patch("subprocess.run", return_value=mock_result) as mock_sub:
            result = download_video(_WATCH_URL, Path("/tmp/out.mp4"), **kwargs)
        return result, mock_sub

    def test_returns_true_on_success(self):
        success, _ = self._run(returncode=0)
        assert success is True

    def test_returns_false_on_nonzero_returncode(self):
        success, _ = self._run(returncode=1, stderr="ERROR: something went wrong")
        assert success is False

    def test_subprocess_called_once(self):
        _, mock_sub = self._run()
        mock_sub.assert_called_once()

    def test_subprocess_receives_url(self):
        _, mock_sub = self._run()
        cmd_used = mock_sub.call_args[0][0]
        assert _WATCH_URL in cmd_used

    def test_timeout_returns_false(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="yt-dlp", timeout=1)):
            result = download_video(_WATCH_URL, Path("/tmp/out.mp4"))
        assert result is False

    def test_exception_returns_false(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("yt-dlp not found")):
            result = download_video(_WATCH_URL, Path("/tmp/out.mp4"))
        assert result is False

    def test_custom_max_height_forwarded_to_command(self):
        mock_result = MagicMock(returncode=0, stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_sub:
            download_video(_WATCH_URL, Path("/tmp/out.mp4"), max_height=360)
        cmd_used = mock_sub.call_args[0][0]
        format_str = cmd_used[cmd_used.index("-f") + 1]
        assert "height<=360" in format_str


# ---------------------------------------------------------------------------
# main  (end-to-end CLI, subprocess mocked)
# ---------------------------------------------------------------------------


class TestMain:
    def _run_main(self, extra_args=None, returncode=0):
        mock_result = MagicMock(returncode=returncode, stderr="")
        args = [_WATCH_URL] + (extra_args or [])
        with patch("subprocess.run", return_value=mock_result):
            with patch("pathlib.Path.mkdir"):
                exit_code = main(args)
        return exit_code

    def test_exit_code_zero_on_success(self):
        assert self._run_main() == 0

    def test_exit_code_one_on_failure(self):
        assert self._run_main(returncode=1) == 1

    def test_custom_output_dir_accepted(self, tmp_path):
        mock_result = MagicMock(returncode=0, stderr="")
        with patch("subprocess.run", return_value=mock_result):
            exit_code = main([_WATCH_URL, "--output-dir", str(tmp_path)])
        assert exit_code == 0

    def test_custom_max_height_accepted(self):
        assert self._run_main(extra_args=["--max-height", "1080"]) == 0

    def test_no_url_uses_default(self):
        mock_result = MagicMock(returncode=0, stderr="")
        with patch("subprocess.run", return_value=mock_result):
            with patch("pathlib.Path.mkdir"):
                exit_code = main([])
        assert exit_code == 0
