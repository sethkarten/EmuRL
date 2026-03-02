"""
Unit tests for data_collection/collect_speedrun_data.py

Tests cover the pure, side-effect-free functions:
- VideoConfig defaults and field validation
- crop_and_resize: numpy frame cropping and resizing
- create_dataset_index: JSON index generation from frame paths
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the module importable without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_collection.collect_speedrun_data import (
    VideoConfig,
    crop_and_resize,
    create_dataset_index,
)


# ---------------------------------------------------------------------------
# VideoConfig
# ---------------------------------------------------------------------------


class TestVideoConfig:
    def test_defaults(self):
        cfg = VideoConfig()
        assert cfg.fps == 2.0
        assert cfg.start_time == 0.0
        assert cfg.end_time is None
        assert cfg.target_width == 160
        assert cfg.target_height == 144

    def test_custom_values(self):
        cfg = VideoConfig(fps=5.0, start_time=30.0, end_time=120.0)
        assert cfg.fps == 5.0
        assert cfg.start_time == 30.0
        assert cfg.end_time == 120.0

    def test_min_game_area_ratio_default(self):
        cfg = VideoConfig()
        assert cfg.min_game_area_ratio == 0.1

    def test_border_color_threshold_default(self):
        cfg = VideoConfig()
        assert cfg.border_color_threshold == 30


# ---------------------------------------------------------------------------
# crop_and_resize
# ---------------------------------------------------------------------------


class TestCropAndResize:
    def _bgr_frame(self, h: int, w: int, color=(128, 64, 32)):
        """Create a solid-color BGR frame."""
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = color
        return frame

    def test_output_shape_matches_target(self):
        cfg = VideoConfig()
        frame = self._bgr_frame(480, 640)
        result = crop_and_resize(frame, (80, 40, 320, 288), cfg)
        assert result.shape == (cfg.target_height, cfg.target_width, 3)

    def test_output_is_rgb_not_bgr(self):
        """crop_and_resize should convert BGR → RGB."""
        cfg = VideoConfig()
        # Pure blue in BGR is (255, 0, 0); in RGB it becomes (0, 0, 255)
        frame = self._bgr_frame(288, 320, color=(255, 0, 0))
        result = crop_and_resize(frame, (0, 0, 320, 288), cfg)
        # Most pixels should have high red channel (0) and high blue channel (255)
        assert result.shape[2] == 3
        assert result[72, 80, 2] > 200  # Blue channel should dominate (blue in RGB)

    def test_full_frame_crop(self):
        cfg = VideoConfig()
        h, w = 144, 160
        frame = self._bgr_frame(h, w)
        result = crop_and_resize(frame, (0, 0, w, h), cfg)
        assert result.shape == (144, 160, 3)

    def test_crop_rect_subset(self):
        """Cropping a sub-region should still produce the correct output shape."""
        cfg = VideoConfig()
        frame = self._bgr_frame(720, 1280)
        # Crop to a small region inside the frame
        result = crop_and_resize(frame, (100, 50, 480, 432), cfg)
        assert result.shape == (144, 160, 3)

    def test_dtype_uint8(self):
        cfg = VideoConfig()
        frame = self._bgr_frame(288, 320)
        result = crop_and_resize(frame, (0, 0, 320, 288), cfg)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# create_dataset_index
# ---------------------------------------------------------------------------


class TestCreateDatasetIndex(object):
    def test_basic_structure(self, tmp_path):
        """Returned JSON should have the expected top-level keys."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        # Create fake .npy files
        paths = []
        for i in range(5):
            p = frames_dir / f"frame_{i:06d}_t{i * 0.5:.2f}.npy"
            p.touch()
            paths.append(p)

        index_path = tmp_path / "index.json"
        url = "https://www.youtube.com/watch?v=test"
        create_dataset_index(paths, index_path, url)

        with open(index_path) as f:
            data = json.load(f)

        assert data["video_url"] == url
        assert data["total_frames"] == 5
        assert data["resolution"] == [144, 160, 3]
        assert len(data["frames"]) == 5

    def test_progress_labels(self, tmp_path):
        """First frame should have progress 0.0, last should have 1.0."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        paths = []
        for i in range(10):
            p = frames_dir / f"frame_{i:06d}_t{i:.2f}.npy"
            p.touch()
            paths.append(p)

        index_path = tmp_path / "index.json"
        create_dataset_index(paths, index_path, "https://example.com")

        with open(index_path) as f:
            data = json.load(f)

        assert data["frames"][0]["progress"] == pytest.approx(0.0)
        assert data["frames"][-1]["progress"] == pytest.approx(1.0)

    def test_single_frame(self, tmp_path):
        """Edge case: only one frame should have progress 0.0."""
        p = tmp_path / "frame_000000_t0.00.npy"
        p.touch()
        index_path = tmp_path / "index.json"
        create_dataset_index([p], index_path, "https://example.com")

        with open(index_path) as f:
            data = json.load(f)

        assert data["total_frames"] == 1
        assert data["frames"][0]["progress"] == pytest.approx(0.0)

    def test_frames_sorted_by_index(self, tmp_path):
        """Frames should be ordered by their numeric index even if paths are shuffled."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        # Create out-of-order paths
        paths = []
        for i in [4, 1, 3, 0, 2]:
            p = frames_dir / f"frame_{i:06d}_t{i:.2f}.npy"
            p.touch()
            paths.append(p)

        index_path = tmp_path / "index.json"
        create_dataset_index(paths, index_path, "https://example.com")

        with open(index_path) as f:
            data = json.load(f)

        frame_indices = [e["frame_idx"] for e in data["frames"]]
        assert frame_indices == list(range(5))

    def test_timestamp_extracted_from_filename(self, tmp_path):
        """Timestamps should be parsed correctly from filenames."""
        p = tmp_path / "frame_000000_t12.50.npy"
        p.touch()
        index_path = tmp_path / "index.json"
        create_dataset_index([p], index_path, "https://example.com")

        with open(index_path) as f:
            data = json.load(f)

        assert data["frames"][0]["timestamp"] == pytest.approx(12.50)
