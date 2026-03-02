"""
Unit tests for data_collection/scrape_speedruns.py

Tests cover the pure, side-effect-free function:
- extract_video_url: parses a speedrun.com API run dict and returns a video URL
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_collection.scrape_speedruns import extract_video_url


# ---------------------------------------------------------------------------
# extract_video_url
# ---------------------------------------------------------------------------


class TestExtractVideoUrl:
    def _run(self, links=None, text=None) -> dict:
        """Helper to build a minimal run dict."""
        videos: dict = {}
        if links is not None:
            videos["links"] = links
        if text is not None:
            videos["text"] = text
        return {"videos": videos}

    def test_youtube_link_returned(self):
        run = self._run(links=[{"uri": "https://www.youtube.com/watch?v=abc123"}])
        assert extract_video_url(run) == "https://www.youtube.com/watch?v=abc123"

    def test_youtu_be_shortlink_returned(self):
        run = self._run(links=[{"uri": "https://youtu.be/abc123"}])
        assert extract_video_url(run) == "https://youtu.be/abc123"

    def test_twitch_link_returned(self):
        run = self._run(links=[{"uri": "https://www.twitch.tv/videos/123456789"}])
        assert extract_video_url(run) == "https://www.twitch.tv/videos/123456789"

    def test_youtube_text_fallback(self):
        """If no link but text contains 'youtube', return the text."""
        run = self._run(text="https://www.youtube.com/watch?v=xyz")
        assert extract_video_url(run) == "https://www.youtube.com/watch?v=xyz"

    def test_twitch_text_fallback(self):
        run = self._run(text="https://twitch.tv/videos/999")
        assert extract_video_url(run) == "https://twitch.tv/videos/999"

    def test_no_videos_field_returns_none(self):
        run = {}
        assert extract_video_url(run) is None

    def test_videos_none_returns_none(self):
        run = {"videos": None}
        assert extract_video_url(run) is None

    def test_empty_links_and_no_text_returns_none(self):
        run = self._run(links=[])
        assert extract_video_url(run) is None

    def test_unrecognised_link_ignored(self):
        """A link that is neither YouTube nor Twitch should be skipped."""
        run = self._run(links=[{"uri": "https://vimeo.com/123456"}])
        assert extract_video_url(run) is None

    def test_first_youtube_wins_over_twitch(self):
        """YouTube link appears before Twitch in the list — should be returned."""
        run = self._run(
            links=[
                {"uri": "https://www.youtube.com/watch?v=yt"},
                {"uri": "https://www.twitch.tv/videos/tv"},
            ]
        )
        assert "youtube.com" in extract_video_url(run)

    def test_links_empty_dict_skipped(self):
        """Link dict with no 'uri' key should not crash."""
        run = self._run(links=[{}])
        assert extract_video_url(run) is None

    def test_text_with_neither_youtube_nor_twitch_returns_none(self):
        run = self._run(text="https://vimeo.com/12345")
        assert extract_video_url(run) is None

    def test_text_empty_string_returns_none(self):
        run = self._run(text="")
        assert extract_video_url(run) is None
