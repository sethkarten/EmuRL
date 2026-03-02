"""
Unit tests for data_collection/scrape_youtube.py

Tests cover the pure, side-effect-free function:
- filter_videos: filters and ranks a list of video metadata dicts
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_collection.scrape_youtube import filter_videos


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _video(title: str, duration: int, vid_id: str = None) -> dict:
    """Build a minimal video metadata dict."""
    return {
        "id": vid_id or title[:8].replace(" ", "_"),
        "title": title,
        "duration": duration,
        "channel": "TestChannel",
        "url": f"https://www.youtube.com/watch?v={vid_id or 'test'}",
    }


# ---------------------------------------------------------------------------
# filter_videos
# ---------------------------------------------------------------------------


class TestFilterVideos:
    # --- duration boundaries ---

    def test_video_meeting_min_duration_passes(self):
        videos = [_video("Pokemon Red Playthrough", 1800, "v1")]
        result = filter_videos(videos, min_duration=1800)
        assert len(result) == 1

    def test_video_below_min_duration_excluded(self):
        videos = [_video("Pokemon Red Short", 1799, "v1")]
        result = filter_videos(videos, min_duration=1800)
        assert len(result) == 0

    def test_video_at_max_duration_passes(self):
        videos = [_video("Pokemon Red Long", 36000, "v1")]
        result = filter_videos(videos, min_duration=1800, max_duration=36000)
        assert len(result) == 1

    def test_video_above_max_duration_excluded(self):
        videos = [_video("Pokemon Red Marathon", 36001, "v1")]
        result = filter_videos(videos, min_duration=1800, max_duration=36000)
        assert len(result) == 0

    def test_zero_duration_excluded(self):
        videos = [_video("Pokemon Red Clip", 0, "v1")]
        result = filter_videos(videos)
        assert len(result) == 0

    def test_none_duration_excluded(self):
        v = _video("Pokemon Red Live", 3600, "v1")
        v["duration"] = None
        result = filter_videos([v])
        assert len(result) == 0

    # --- skip keywords ---

    def test_review_video_excluded(self):
        videos = [_video("Pokemon Red Review 2024", 5400, "v1")]
        result = filter_videos(videos)
        assert len(result) == 0

    def test_soundtrack_excluded(self):
        videos = [_video("Pokemon Red OST Full Soundtrack", 5400, "v1")]
        result = filter_videos(videos)
        assert len(result) == 0

    def test_speedrun_excluded(self):
        videos = [_video("Pokemon Red Speedrun World Record", 3000, "v1")]
        result = filter_videos(videos)
        assert len(result) == 0

    def test_tier_list_excluded(self):
        videos = [_video("Pokemon Red Tier List", 3600, "v1")]
        result = filter_videos(videos)
        assert len(result) == 0

    def test_normal_lets_play_passes(self):
        videos = [_video("Pokemon Red - Let's Play - Part 1", 5400, "v1")]
        result = filter_videos(videos)
        assert len(result) == 1

    # --- deduplication ---

    def test_duplicate_ids_deduplicated(self):
        v = _video("Pokemon Red Walkthrough", 5400, "dup001")
        result = filter_videos([v, v])
        assert len(result) == 1

    def test_different_ids_both_kept(self):
        v1 = _video("Pokemon Red Walkthrough Part 1", 5400, "vid001")
        v2 = _video("Pokemon Red Walkthrough Part 2", 5400, "vid002")
        result = filter_videos([v1, v2])
        assert len(result) == 2

    # --- prefer keywords scoring ---

    def test_preferred_keywords_increase_score(self):
        """Videos with preferred keywords should appear first."""
        low_score = _video("Pokemon Red Gameplay", 5400, "low")
        high_score = _video("Pokemon Red Let's Play Part 1", 5400, "high")
        result = filter_videos([low_score, high_score])
        # high_score contains "let's play" and "part 1" — both preferred keywords
        assert result[0]["id"] == "high"

    def test_score_field_added(self):
        v = _video("Pokemon Red Playthrough", 5400, "v1")
        result = filter_videos([v])
        assert "score" in result[0]

    # --- empty input ---

    def test_empty_input_returns_empty(self):
        assert filter_videos([]) == []

    # --- title case insensitivity ---

    def test_skip_keyword_case_insensitive(self):
        """The filter lowercases titles before checking keywords."""
        videos = [_video("Pokemon Red REVIEW 2024", 5400, "v1")]
        result = filter_videos(videos)
        assert len(result) == 0
