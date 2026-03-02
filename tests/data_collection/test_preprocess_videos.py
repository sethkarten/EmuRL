"""
Unit tests for data_collection/preprocess_videos.py

Tests cover the pure, side-effect-free functions:
- load_crop_config: reading and defaulting crop JSON
- get_video_crop_region: mapping video IDs to ScreenRegion
- should_skip_video: checking the skip-list
- ScreenRegion.crop: cropping and resizing a numpy frame
- is_gameplay_frame: filtering out black/white/flat frames
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_collection.preprocess_videos import (
    GB_HEIGHT,
    GB_WIDTH,
    ScreenRegion,
    get_video_crop_region,
    is_gameplay_frame,
    load_crop_config,
    should_skip_video,
)

# ---------------------------------------------------------------------------
# load_crop_config
# ---------------------------------------------------------------------------


class TestLoadCropConfig:
    def test_missing_file_returns_empty_config(self, tmp_path):
        cfg = load_crop_config(tmp_path / "nonexistent.json")
        assert cfg == {"usable_videos": {}, "skip_videos": {}}

    def test_reads_existing_file(self, tmp_path):
        data = {
            "usable_videos": {"abc123": {"crop": [0, 0, 160, 144]}},
            "skip_videos": {},
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(data))

        cfg = load_crop_config(config_path)
        assert "abc123" in cfg["usable_videos"]

    def test_returns_full_config_structure(self, tmp_path):
        data = {
            "usable_videos": {"vid1": {"crop": [10, 20, 170, 164]}},
            "skip_videos": {"bad_vid": "shaky camera"},
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(data))

        cfg = load_crop_config(config_path)
        assert cfg["skip_videos"]["bad_vid"] == "shaky camera"


# ---------------------------------------------------------------------------
# get_video_crop_region
# ---------------------------------------------------------------------------


class TestGetVideoCropRegion:
    def _config(self, entries: dict) -> dict:
        return {"usable_videos": entries, "skip_videos": {}}

    def test_exact_match(self):
        cfg = self._config({"abc123xyz": {"crop": [10, 20, 170, 164]}})
        region = get_video_crop_region("abc123xyz", cfg)
        assert region is not None
        assert region.x == 10
        assert region.y == 20
        assert region.width == 160  # x2 - x1
        assert region.height == 144  # y2 - y1

    def test_prefix_match(self):
        """Video ID that starts with the config key should match."""
        cfg = self._config({"abc123": {"crop": [0, 0, 160, 144]}})
        region = get_video_crop_region("abc123_extra_suffix", cfg)
        assert region is not None

    def test_no_match_returns_none(self):
        cfg = self._config({"abc123": {"crop": [0, 0, 160, 144]}})
        region = get_video_crop_region("xyz999", cfg)
        assert region is None

    def test_auto_crop_returns_none(self):
        """'auto' crop value is not a usable ScreenRegion."""
        cfg = self._config({"vid1": {"crop": "auto"}})
        region = get_video_crop_region("vid1", cfg)
        assert region is None

    def test_missing_crop_key_returns_none(self):
        cfg = self._config({"vid1": {"note": "no crop field"}})
        region = get_video_crop_region("vid1", cfg)
        assert region is None

    def test_skips_comment_keys(self):
        """Keys starting with '_' are treated as comments."""
        cfg = self._config(
            {"_comment": {"crop": [0, 0, 160, 144]}, "real": {"crop": [0, 0, 160, 144]}}
        )
        region = get_video_crop_region("_comment_should_not_match", cfg)
        assert region is None

    def test_confidence_is_one_for_manual_crop(self):
        cfg = self._config({"vid1": {"crop": [0, 0, 160, 144]}})
        region = get_video_crop_region("vid1", cfg)
        assert region.confidence == 1.0


# ---------------------------------------------------------------------------
# should_skip_video
# ---------------------------------------------------------------------------


class TestShouldSkipVideo:
    def _config(self, skip_entries: dict) -> dict:
        return {"usable_videos": {}, "skip_videos": skip_entries}

    def test_known_skip_returns_reason(self):
        cfg = self._config({"bad_vid": "shaky camera"})
        reason = should_skip_video("bad_vid_abc", cfg)
        assert reason == "shaky camera"

    def test_unknown_video_returns_none(self):
        cfg = self._config({"bad_vid": "shaky camera"})
        assert should_skip_video("good_vid_001", cfg) is None

    def test_empty_skip_list_returns_none(self):
        cfg = self._config({})
        assert should_skip_video("any_video", cfg) is None

    def test_key_contained_in_video_id(self):
        """Key that appears anywhere in the video ID should trigger skip."""
        cfg = self._config({"no_sound": "no audio"})
        reason = should_skip_video("run_no_sound_2024", cfg)
        assert reason == "no audio"


# ---------------------------------------------------------------------------
# ScreenRegion.crop
# ---------------------------------------------------------------------------


class TestScreenRegionCrop:
    def _bgr_frame(self, h: int, w: int, color=(100, 150, 200)):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = color
        return frame

    def test_output_shape_is_gb_resolution(self):
        region = ScreenRegion(x=10, y=10, width=320, height=288, confidence=1.0)
        frame = self._bgr_frame(480, 640)
        result = region.crop(frame)
        assert result.shape == (GB_HEIGHT, GB_WIDTH, 3)

    def test_crop_full_frame(self):
        region = ScreenRegion(x=0, y=0, width=160, height=144, confidence=1.0)
        frame = self._bgr_frame(144, 160)
        result = region.crop(frame)
        assert result.shape == (GB_HEIGHT, GB_WIDTH, 3)

    def test_dtype_preserved(self):
        region = ScreenRegion(x=0, y=0, width=160, height=144, confidence=1.0)
        frame = self._bgr_frame(144, 160)
        result = region.crop(frame)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# is_gameplay_frame
# ---------------------------------------------------------------------------


class TestIsGameplayFrame:
    def _uniform_frame(self, value: int) -> np.ndarray:
        """Create a 144×160×3 frame filled with a single value."""
        return np.full((144, 160, 3), value, dtype=np.uint8)

    def test_normal_frame_passes(self):
        # Random-ish frame with good brightness and variation
        rng = np.random.default_rng(42)
        frame = rng.integers(50, 200, size=(144, 160, 3), dtype=np.uint8)
        assert is_gameplay_frame(frame) is True

    def test_black_frame_rejected(self):
        frame = self._uniform_frame(0)
        assert is_gameplay_frame(frame) is False

    def test_near_black_frame_rejected(self):
        frame = self._uniform_frame(10)
        assert is_gameplay_frame(frame) is False

    def test_white_frame_rejected(self):
        frame = self._uniform_frame(255)
        assert is_gameplay_frame(frame) is False

    def test_near_white_frame_rejected(self):
        frame = self._uniform_frame(250)
        assert is_gameplay_frame(frame) is False

    def test_solid_mid_gray_rejected(self):
        """Solid color with no variation should be filtered out."""
        frame = self._uniform_frame(128)
        assert is_gameplay_frame(frame) is False

    def test_frame_with_high_variation_passes(self):
        frame = np.zeros((144, 160, 3), dtype=np.uint8)
        frame[:72, :] = 200  # Top half bright
        frame[72:, :] = 50  # Bottom half dark
        assert is_gameplay_frame(frame) is True
