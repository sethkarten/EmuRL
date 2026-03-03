"""
Unit tests for data_collection/preprocess_simple.py

Tests cover:
- Constants (GBA_WIDTH, GBA_HEIGHT, defaults)
- load_crop_config: missing file, empty file, valid file
- get_crop_box: exact match, prefix match, substring match, missing key, skip-key
- should_skip: match, no match
- process_video: frame extraction, resizing, BGR->RGB, .npy output
- process_video: frame interval / fps downsampling
- process_video: metadata.json contents (includes crop field)
- process_video: crop applied before resize
- process_video: skip when in skip_videos
- process_video: error case (unreadable video)
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_collection.preprocess_simple import (
    DEFAULT_CONFIG,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    GBA_HEIGHT,
    GBA_WIDTH,
    get_crop_box,
    load_crop_config,
    process_video,
    should_skip,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_fake_cap(frames: list[np.ndarray], video_fps: float = 30.0):
    """
    Build a MagicMock that mimics cv2.VideoCapture.

    frames: list of BGR uint8 arrays, each (H, W, 3)
    """
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        # cv2.CAP_PROP_FPS = 5
        5: video_fps,
        # cv2.CAP_PROP_FRAME_COUNT = 7
        7: float(len(frames)),
    }.get(prop, 0.0)

    read_returns = [(True, f) for f in frames] + [(False, None)]
    cap.read.side_effect = read_returns
    return cap


def bgr_frame(h: int = 60, w: int = 80, color=(100, 150, 200)) -> np.ndarray:
    """Solid-color BGR frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color  # color is (B, G, R)
    return frame


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_gba_width(self):
        assert GBA_WIDTH == 240

    def test_gba_height(self):
        assert GBA_HEIGHT == 160

    def test_aspect_ratio(self):
        assert GBA_WIDTH / GBA_HEIGHT == pytest.approx(1.5)

    def test_default_input_is_path(self):
        assert isinstance(DEFAULT_INPUT, Path)

    def test_default_output_is_path(self):
        assert isinstance(DEFAULT_OUTPUT, Path)

    def test_default_config_is_path(self):
        assert isinstance(DEFAULT_CONFIG, Path)

    def test_default_config_filename(self):
        assert DEFAULT_CONFIG.name == "video_crop_config.json"


# ---------------------------------------------------------------------------
# load_crop_config
# ---------------------------------------------------------------------------


class TestLoadCropConfig:
    def test_missing_file_returns_empty(self, tmp_path):
        cfg = load_crop_config(tmp_path / "nonexistent.json")
        assert cfg == {"usable_videos": {}, "skip_videos": {}}

    def test_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "cfg.json"
        p.write_text("")
        cfg = load_crop_config(p)
        assert cfg == {"usable_videos": {}, "skip_videos": {}}

    def test_valid_file_loaded(self, tmp_path):
        data = {"usable_videos": {"v1": {"crop": [0, 0, 640, 360]}}, "skip_videos": {}}
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(data))
        assert load_crop_config(p) == data

    def test_returns_dict(self, tmp_path):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({"usable_videos": {}, "skip_videos": {}}))
        assert isinstance(load_crop_config(p), dict)


# ---------------------------------------------------------------------------
# get_crop_box
# ---------------------------------------------------------------------------


class TestGetCropBox:
    BASE_CFG = {
        "usable_videos": {
            "my_video": {"crop": [10, 20, 300, 200]},
            "prefixed_vid": {"crop": [0, 0, 640, 360]},
        },
        "skip_videos": {},
    }

    def test_exact_match(self):
        assert get_crop_box("my_video", self.BASE_CFG) == (10, 20, 300, 200)

    def test_prefix_match(self):
        # key "my_video" is a prefix of "my_video_extra"
        cfg = {"usable_videos": {"my_vid": {"crop": [5, 5, 100, 80]}}, "skip_videos": {}}
        assert get_crop_box("my_vid_extra", cfg) == (5, 5, 100, 80)

    def test_substring_match(self):
        # key "vid" is substring of "my_vid_extra"
        cfg = {"usable_videos": {"vid": {"crop": [1, 2, 3, 4]}}, "skip_videos": {}}
        assert get_crop_box("my_vid_extra", cfg) == (1, 2, 3, 4)

    def test_missing_returns_none(self):
        assert get_crop_box("unknown_video", self.BASE_CFG) is None

    def test_underscore_key_ignored(self):
        cfg = {"usable_videos": {"_comment": {"crop": [0, 0, 1, 1]}}, "skip_videos": {}}
        assert get_crop_box("_comment", cfg) is None

    def test_returns_integers(self):
        box = get_crop_box("my_video", self.BASE_CFG)
        assert all(isinstance(v, int) for v in box)


# ---------------------------------------------------------------------------
# should_skip
# ---------------------------------------------------------------------------


class TestShouldSkip:
    CFG = {
        "usable_videos": {},
        "skip_videos": {"bad_video": "low quality", "outro_": "credits only"},
    }

    def test_exact_match(self):
        assert should_skip("bad_video", self.CFG) == "low quality"

    def test_prefix_match(self):
        # key "outro_" is prefix of "outro_final"
        assert should_skip("outro_final", self.CFG) == "credits only"

    def test_no_match(self):
        assert should_skip("good_video", self.CFG) is None

    def test_empty_config(self):
        assert should_skip("any_video", {"skip_videos": {}}) is None


# ---------------------------------------------------------------------------
# process_video — output shape / format
# ---------------------------------------------------------------------------


class TestProcessVideoOutput:
    def test_creates_output_directory(self, tmp_path):
        frames = [bgr_frame() for _ in range(3)]
        cap = make_fake_cap(frames, video_fps=1.0)

        with patch("data_collection.preprocess_simple.cv2.VideoCapture", return_value=cap):
            process_video(Path("fake_video.mp4"), tmp_path, fps=1.0)

        assert (tmp_path / "fake_video").is_dir()

    def test_saves_npy_files(self, tmp_path):
        frames = [bgr_frame() for _ in range(3)]
        cap = make_fake_cap(frames, video_fps=1.0)

        with patch("data_collection.preprocess_simple.cv2.VideoCapture", return_value=cap):
            process_video(Path("myvid.mp4"), tmp_path, fps=1.0)

        npy_files = sorted((tmp_path / "myvid").glob("frame_*.npy"))
        assert len(npy_files) == 3

    def test_frame_shape_is_gba(self, tmp_path):
        frames = [bgr_frame() for _ in range(2)]
        cap = make_fake_cap(frames, video_fps=1.0)

        with patch("data_collection.preprocess_simple.cv2.VideoCapture", return_value=cap):
            process_video(Path("vid.mp4"), tmp_path, fps=1.0)

        f = np.load(tmp_path / "vid" / "frame_000000.npy")
        assert f.shape == (GBA_HEIGHT, GBA_WIDTH, 3)

    def test_frame_dtype_is_uint8(self, tmp_path):
        frames = [bgr_frame() for _ in range(1)]
        cap = make_fake_cap(frames, video_fps=1.0)

        with patch("data_collection.preprocess_simple.cv2.VideoCapture", return_value=cap):
            process_video(Path("vid.mp4"), tmp_path, fps=1.0)

        f = np.load(tmp_path / "vid" / "frame_000000.npy")
        assert f.dtype == np.uint8

    def test_bgr_converted_to_rgb(self, tmp_path):
        """
        Input frame is pure blue in BGR (B=255, G=0, R=0).
        After BGR->RGB conversion the saved array should be (R=0, G=0, B=255),
        i.e. the first channel (R) should be 0, the third (B) should be 255.
        """
        blue_bgr = bgr_frame(color=(255, 0, 0))  # pure blue in BGR
        cap = make_fake_cap([blue_bgr], video_fps=1.0)

        with patch("data_collection.preprocess_simple.cv2.VideoCapture", return_value=cap):
            process_video(Path("vid.mp4"), tmp_path, fps=1.0)

        f = np.load(tmp_path / "vid" / "frame_000000.npy")
        # In RGB: red channel should be 0, blue channel should be 255
        assert f[0, 0, 0] == 0    # R
        assert f[0, 0, 2] == 255  # B


# ---------------------------------------------------------------------------
# process_video — fps / frame interval
# ---------------------------------------------------------------------------


class TestProcessVideoFps:
    def test_frame_interval_downsamples(self, tmp_path):
        """
        30 fps video, requesting 1 fps output -> every 30th frame saved.
        Provide 90 frames -> expect 3 saved frames.
        """
        frames = [bgr_frame() for _ in range(90)]
        cap = make_fake_cap(frames, video_fps=30.0)

        with patch("data_collection.preprocess_simple.cv2.VideoCapture", return_value=cap):
            result = process_video(Path("vid.mp4"), tmp_path, fps=1.0)

        assert result["frames"] == 3

    def test_fps_higher_than_source_saves_all(self, tmp_path):
        """
        Requesting fps >= source fps -> frame_interval=1 -> every frame saved.
        """
        frames = [bgr_frame() for _ in range(5)]
        cap = make_fake_cap(frames, video_fps=2.0)

        with patch("data_collection.preprocess_simple.cv2.VideoCapture", return_value=cap):
            result = process_video(Path("vid.mp4"), tmp_path, fps=10.0)

        assert result["frames"] == 5

    def test_default_fps_is_2(self, tmp_path):
        """process_video default fps=2.0: 10fps source, 5 frames -> saves every 5, so 1 frame."""
        frames = [bgr_frame() for _ in range(5)]
        cap = make_fake_cap(frames, video_fps=10.0)

        with patch("data_collection.preprocess_simple.cv2.VideoCapture", return_value=cap):
            result = process_video(Path("vid.mp4"), tmp_path)  # default fps=2.0

        assert result["frames"] == 1


# ---------------------------------------------------------------------------
# process_video — metadata
# ---------------------------------------------------------------------------


class TestProcessVideoMetadata:
    def test_metadata_file_created(self, tmp_path):
        cap = make_fake_cap([bgr_frame()], video_fps=1.0)

        with patch("data_collection.preprocess_simple.cv2.VideoCapture", return_value=cap):
            process_video(Path("vid.mp4"), tmp_path, fps=1.0)

        assert (tmp_path / "vid" / "metadata.json").exists()

    def test_metadata_contents(self, tmp_path):
        frames = [bgr_frame() for _ in range(2)]
        cap = make_fake_cap(frames, video_fps=1.0)

        with patch("data_collection.preprocess_simple.cv2.VideoCapture", return_value=cap):
            process_video(Path("myvid.mp4"), tmp_path, fps=1.0)

        meta = json.loads((tmp_path / "myvid" / "metadata.json").read_text())
        assert meta["video_id"] == "myvid"
        assert meta["frames"] == 2
        assert meta["fps"] == 1.0
        assert meta["resolution"] == f"{GBA_WIDTH}x{GBA_HEIGHT}"
        assert meta["crop"] is None  # no crop config supplied

    def test_metadata_crop_field_when_config_provided(self, tmp_path):
        frames = [bgr_frame(60, 80) for _ in range(1)]
        cap = make_fake_cap(frames, video_fps=1.0)
        cfg = {"usable_videos": {"myvid": {"crop": [0, 0, 80, 60]}}, "skip_videos": {}}

        with patch("data_collection.preprocess_simple.cv2.VideoCapture", return_value=cap):
            process_video(Path("myvid.mp4"), tmp_path, fps=1.0, crop_config=cfg)

        meta = json.loads((tmp_path / "myvid" / "metadata.json").read_text())
        assert meta["crop"] == [0, 0, 80, 60]

    def test_return_dict_matches_metadata(self, tmp_path):
        frames = [bgr_frame() for _ in range(3)]
        cap = make_fake_cap(frames, video_fps=1.0)

        with patch("data_collection.preprocess_simple.cv2.VideoCapture", return_value=cap):
            result = process_video(Path("vid.mp4"), tmp_path, fps=1.0)

        assert result["frames"] == 3
        assert result["video_id"] == "vid"


# ---------------------------------------------------------------------------
# process_video — crop config integration
# ---------------------------------------------------------------------------


class TestProcessVideoCrop:
    def test_crop_reduces_area_before_resize(self, tmp_path):
        """
        Provide a 60x80 frame where the top-left 30x40 is red and the rest is
        blue.  Crop to [0,0,40,30] (x2=40,y2=30 so 30 rows x 40 cols = red).
        After resize to GBA all pixels should be red-ish in RGB (R≈255, B≈0).
        """
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        frame[:30, :40] = (0, 0, 255)   # BGR: top-left patch pure red
        frame[30:, 40:] = (255, 0, 0)   # BGR: bottom-right patch pure blue

        cap = make_fake_cap([frame], video_fps=1.0)
        cfg = {"usable_videos": {"vid": {"crop": [0, 0, 40, 30]}}, "skip_videos": {}}

        with patch("data_collection.preprocess_simple.cv2.VideoCapture", return_value=cap):
            process_video(Path("vid.mp4"), tmp_path, fps=1.0, crop_config=cfg)

        saved = np.load(tmp_path / "vid" / "frame_000000.npy")
        # After BGR->RGB: red (0,0,255 BGR) becomes (255,0,0 RGB)
        assert saved[0, 0, 0] > 200   # R channel high
        assert saved[0, 0, 2] < 50    # B channel low

    def test_no_crop_entry_processes_full_frame(self, tmp_path):
        frames = [bgr_frame() for _ in range(2)]
        cap = make_fake_cap(frames, video_fps=1.0)
        cfg = {"usable_videos": {}, "skip_videos": {}}  # no entry for "vid"

        with patch("data_collection.preprocess_simple.cv2.VideoCapture", return_value=cap):
            result = process_video(Path("vid.mp4"), tmp_path, fps=1.0, crop_config=cfg)

        assert result.get("frames") == 2

    def test_crop_clamps_to_frame_bounds(self, tmp_path):
        """Crop box larger than frame should not crash."""
        frame = bgr_frame(60, 80)
        cap = make_fake_cap([frame], video_fps=1.0)
        cfg = {"usable_videos": {"vid": {"crop": [0, 0, 9999, 9999]}}, "skip_videos": {}}

        with patch("data_collection.preprocess_simple.cv2.VideoCapture", return_value=cap):
            result = process_video(Path("vid.mp4"), tmp_path, fps=1.0, crop_config=cfg)

        assert result.get("frames") == 1


class TestProcessVideoSkip:
    def test_skip_returns_skipped_status(self, tmp_path):
        cfg = {"usable_videos": {}, "skip_videos": {"bad": "low quality"}}

        result = process_video(Path("bad_run.mp4"), tmp_path, fps=1.0, crop_config=cfg)

        assert result["status"] == "skipped"
        assert "low quality" in result["reason"]

    def test_skip_creates_no_files(self, tmp_path):
        cfg = {"usable_videos": {}, "skip_videos": {"bad": "corrupted"}}

        process_video(Path("bad_run.mp4"), tmp_path, fps=1.0, crop_config=cfg)

        assert not list(tmp_path.rglob("*.npy"))
        assert not list(tmp_path.rglob("metadata.json"))

    def test_no_skip_when_config_is_none(self, tmp_path):
        """No crop_config means skip list is never consulted."""
        cap = make_fake_cap([bgr_frame()], video_fps=1.0)

        with patch("data_collection.preprocess_simple.cv2.VideoCapture", return_value=cap):
            result = process_video(Path("bad_run.mp4"), tmp_path, fps=1.0, crop_config=None)

        assert result.get("status") != "skipped"


# ---------------------------------------------------------------------------
# process_video — error handling
# ---------------------------------------------------------------------------


class TestProcessVideoErrors:
    def test_returns_error_status_if_cannot_open(self, tmp_path):
        cap = MagicMock()
        cap.isOpened.return_value = False

        with patch("data_collection.preprocess_simple.cv2.VideoCapture", return_value=cap):
            result = process_video(Path("bad.mp4"), tmp_path, fps=1.0)

        assert result["status"] == "error"

    def test_no_npy_files_on_error(self, tmp_path):
        cap = MagicMock()
        cap.isOpened.return_value = False

        with patch("data_collection.preprocess_simple.cv2.VideoCapture", return_value=cap):
            process_video(Path("bad.mp4"), tmp_path, fps=1.0)

        npy_files = list(tmp_path.rglob("*.npy"))
        assert len(npy_files) == 0
