"""
Unit tests for tools/crop_tool.py

Strategy
--------
* Helper functions (find_videos, load_config, save_config, scale_frame,
  make_preview) are tested directly.  PIL/ImageTk is mocked so no display
  is required.
* CropTool logic methods (_clamp_canvas, _hit_corner, _canvas_to_orig_update)
  are exercised by calling the unbound methods on a minimal attribute stub,
  avoiding any real tkinter widgets.
"""

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Patch tkinter and PIL.ImageTk before importing crop_tool so that the
# import succeeds even without a display / Tcl/Tk installation.
# ---------------------------------------------------------------------------

# Build a minimal fake tkinter module
_tk_stub = types.ModuleType("tkinter")
_tk_stub.Tk = MagicMock
_tk_stub.Canvas = MagicMock
_tk_stub.Frame = MagicMock
_tk_stub.Label = MagicMock
_tk_stub.Button = MagicMock
_tk_stub.NW = "nw"
_tk_stub.X = "x"
_tk_stub.Y = "y"
_tk_stub.BOTH = "both"
_tk_stub.LEFT = "left"
_tk_stub.RIGHT = "right"
_tk_stub.FLAT = "flat"
_tk_stub.SOLID = "solid"
_tk_stub.NORMAL = "normal"
_tk_stub.DISABLED = "disabled"
_tk_stub.END = "end"
_messagebox = types.ModuleType("tkinter.messagebox")
_messagebox.showwarning = MagicMock()
sys.modules.setdefault("tkinter", _tk_stub)
sys.modules.setdefault("tkinter.messagebox", _messagebox)
_tk_stub.messagebox = _messagebox

# Fake ImageTk
_fake_photo = MagicMock()
_fake_photo.width.return_value = 320
_fake_photo.height.return_value = 240

_imagetk = MagicMock()
_imagetk.PhotoImage.return_value = _fake_photo
sys.modules.setdefault("PIL.ImageTk", _imagetk)

# Now safe to import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

with patch("PIL.ImageTk.PhotoImage", return_value=_fake_photo):
    from tools.crop_tool import (
        GBA_HEIGHT,
        GBA_WIDTH,
        HANDLE_RADIUS,
        CropTool,
        find_videos,
        load_config,
        make_preview,
        save_config,
        scale_frame,
    )


# ===========================================================================
# find_videos
# ===========================================================================


class TestFindVideos:
    def test_finds_mp4_files(self, tmp_path):
        (tmp_path / "a.mp4").touch()
        (tmp_path / "b.mkv").touch()
        results = find_videos(tmp_path)
        assert any(v.name == "a.mp4" for v in results)
        assert any(v.name == "b.mkv" for v in results)

    def test_excludes_frames_directories(self, tmp_path):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        (frames_dir / "clip.mp4").touch()
        (tmp_path / "good.mp4").touch()
        results = find_videos(tmp_path)
        names = [v.name for v in results]
        assert "clip.mp4" not in names
        assert "good.mp4" in names

    def test_ignores_non_video_files(self, tmp_path):
        (tmp_path / "notes.txt").touch()
        (tmp_path / "data.npy").touch()
        results = find_videos(tmp_path)
        assert results == []

    def test_returns_sorted_list(self, tmp_path):
        (tmp_path / "c.mp4").touch()
        (tmp_path / "a.mp4").touch()
        (tmp_path / "b.mp4").touch()
        results = find_videos(tmp_path)
        names = [v.name for v in results]
        assert names == sorted(names)

    def test_recurses_into_subdirectories(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.mp4").touch()
        results = find_videos(tmp_path)
        assert any(v.name == "deep.mp4" for v in results)

    def test_empty_directory_returns_empty_list(self, tmp_path):
        assert find_videos(tmp_path) == []


# ===========================================================================
# load_config
# ===========================================================================


class TestLoadConfig:
    def test_returns_defaults_when_file_missing(self, tmp_path):
        cfg = load_config(tmp_path / "no_such.json")
        assert cfg == {"usable_videos": {}, "skip_videos": {}}

    def test_returns_defaults_for_empty_file(self, tmp_path):
        p = tmp_path / "empty.json"
        p.touch()
        cfg = load_config(p)
        assert cfg == {"usable_videos": {}, "skip_videos": {}}

    def test_loads_valid_json(self, tmp_path):
        data = {"usable_videos": {"vid1": {"crop": [0, 0, 100, 80]}}, "skip_videos": {}}
        p = tmp_path / "config.json"
        p.write_text(json.dumps(data))
        assert load_config(p) == data

    def test_preserves_skip_videos(self, tmp_path):
        data = {"usable_videos": {}, "skip_videos": {"bad": "blurry"}}
        p = tmp_path / "config.json"
        p.write_text(json.dumps(data))
        assert load_config(p)["skip_videos"] == {"bad": "blurry"}


# ===========================================================================
# save_config
# ===========================================================================


class TestSaveConfig:
    def test_creates_parent_directories(self, tmp_path):
        p = tmp_path / "deep" / "nested" / "config.json"
        save_config({"usable_videos": {}, "skip_videos": {}}, p)
        assert p.exists()

    def test_written_json_is_valid(self, tmp_path):
        data = {"usable_videos": {"v": {"crop": [1, 2, 3, 4]}}, "skip_videos": {}}
        p = tmp_path / "out.json"
        save_config(data, p)
        assert json.loads(p.read_text()) == data

    def test_overwrites_existing_file(self, tmp_path):
        p = tmp_path / "cfg.json"
        save_config({"usable_videos": {"old": {}}, "skip_videos": {}}, p)
        save_config({"usable_videos": {"new": {}}, "skip_videos": {}}, p)
        loaded = json.loads(p.read_text())
        assert "new" in loaded["usable_videos"]
        assert "old" not in loaded["usable_videos"]

    def test_uses_indent_for_readability(self, tmp_path):
        p = tmp_path / "cfg.json"
        save_config({"usable_videos": {}, "skip_videos": {}}, p)
        raw = p.read_text()
        assert "\n" in raw  # indented JSON contains newlines


# ===========================================================================
# scale_frame
# ===========================================================================


class TestScaleFrame:
    def _make_frame(self, h, w):
        return np.zeros((h, w, 3), dtype=np.uint8)

    @patch("PIL.ImageTk.PhotoImage", return_value=_fake_photo)
    def test_scale_does_not_upscale(self, _mock):
        # Frame smaller than CANVAS_MAX → scale should be 1.0
        frame = self._make_frame(100, 200)
        _photo, scale = scale_frame(frame)
        assert scale == 1.0

    @patch("PIL.ImageTk.PhotoImage", return_value=_fake_photo)
    def test_scale_downscales_wide_frame(self, _mock):
        # 1800-wide frame must be shrunk below CANVAS_MAX_W=900
        frame = self._make_frame(400, 1800)
        _photo, scale = scale_frame(frame)
        assert scale < 1.0
        assert scale == pytest.approx(900 / 1800)

    @patch("PIL.ImageTk.PhotoImage", return_value=_fake_photo)
    def test_scale_downscales_tall_frame(self, _mock):
        frame = self._make_frame(1200, 400)
        _photo, scale = scale_frame(frame)
        assert scale < 1.0
        assert scale == pytest.approx(600 / 1200)

    @patch("PIL.ImageTk.PhotoImage", return_value=_fake_photo)
    def test_returns_photimage_and_float(self, _mock):
        frame = self._make_frame(100, 100)
        result = scale_frame(frame)
        assert len(result) == 2
        assert isinstance(result[1], float)


# ===========================================================================
# make_preview
# ===========================================================================


class TestMakePreview:
    def _frame(self, h=480, w=640):
        """Return a test frame with increasing pixel values."""
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(w, dtype=np.uint8)  # blue channel ramp
        return frame

    @patch("PIL.ImageTk.PhotoImage", return_value=_fake_photo)
    def test_valid_crop_returns_photimage(self, _mock):
        result = make_preview(self._frame(), 0, 0, 320, 240)
        assert result is _fake_photo

    @patch("PIL.ImageTk.PhotoImage")
    def test_output_is_gba_resolution(self, mock_photo):
        """The numpy array passed to ImageTk must be GBA_WIDTH × GBA_HEIGHT."""
        captured = []

        def capture(img):
            captured.append(np.array(img))
            return _fake_photo

        mock_photo.side_effect = capture

        make_preview(self._frame(), 0, 0, 320, 240)
        assert len(captured) == 1
        h, w, c = captured[0].shape
        assert w == GBA_WIDTH
        assert h == GBA_HEIGHT
        assert c == 3

    @patch("PIL.ImageTk.PhotoImage")
    def test_bgr_to_rgb_conversion(self, mock_photo):
        """A pure-blue BGR frame must appear pure-red in the RGB output."""
        captured = []

        def capture(img):
            captured.append(np.array(img))
            return _fake_photo

        mock_photo.side_effect = capture

        # Pure blue in BGR = (B=255, G=0, R=0)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # BGR blue channel

        make_preview(frame, 0, 0, 200, 200)
        arr = captured[0]
        # After BGR→RGB: the blue component (BGR index 0) moves to RGB index 2.
        # R channel (index 0) should be 0, B channel (index 2) should be 255.
        assert arr[0, 0, 0] == 0    # R
        assert arr[0, 0, 2] == 255  # B (was BGR blue)

    @patch("PIL.ImageTk.PhotoImage", return_value=_fake_photo)
    def test_degenerate_crop_returns_blank(self, _mock):
        """Zero-area crop (x1==x2) should return a blank black image."""
        result = make_preview(self._frame(), 100, 100, 100, 200)
        assert result is _fake_photo

    @patch("PIL.ImageTk.PhotoImage", return_value=_fake_photo)
    def test_inverted_coords_are_corrected(self, _mock):
        """x1>x2 or y1>y2 should be sorted internally without error."""
        result = make_preview(self._frame(), 320, 240, 0, 0)
        assert result is _fake_photo

    @patch("PIL.ImageTk.PhotoImage", return_value=_fake_photo)
    def test_out_of_bounds_coords_clamped(self, _mock):
        """Coords beyond frame dimensions must be clamped without raising."""
        result = make_preview(self._frame(480, 640), -50, -50, 9999, 9999)
        assert result is _fake_photo


# ===========================================================================
# CropTool pure logic (no GUI required)
# All methods are called as unbound functions on a minimal attribute stub.
# ===========================================================================


class _State:
    """Minimal stand-in for CropTool instance state."""

    def __init__(self, canvas_w=800, canvas_h=600, scale=0.5,
                 orig_w=1280, orig_h=720, rect_canvas=None):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.scale = scale
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.rect_canvas = rect_canvas
        self.rect_orig = None
        # Stub out lbl_coords so _update_coords_display doesn't crash
        self.lbl_coords = MagicMock()
        self._update_coords_display = MagicMock()


class TestClampCanvas:
    def test_clamps_negative_x(self):
        s = _State()
        x, y = CropTool._clamp_canvas(s, -10, 100)
        assert x == 0

    def test_clamps_negative_y(self):
        s = _State()
        x, y = CropTool._clamp_canvas(s, 100, -5)
        assert y == 0

    def test_clamps_x_beyond_width(self):
        s = _State(canvas_w=800)
        x, y = CropTool._clamp_canvas(s, 900, 100)
        assert x == 800

    def test_clamps_y_beyond_height(self):
        s = _State(canvas_h=600)
        x, y = CropTool._clamp_canvas(s, 100, 700)
        assert y == 600

    def test_passthrough_within_bounds(self):
        s = _State(canvas_w=800, canvas_h=600)
        x, y = CropTool._clamp_canvas(s, 400, 300)
        assert (x, y) == (400, 300)

    def test_corner_at_exact_boundary(self):
        s = _State(canvas_w=800, canvas_h=600)
        x, y = CropTool._clamp_canvas(s, 800, 600)
        assert (x, y) == (800, 600)


class TestHitCorner:
    def _state_with_rect(self, cx1=100, cy1=100, cx2=400, cy2=300):
        return _State(rect_canvas=(cx1, cy1, cx2, cy2))

    def test_no_rect_returns_none(self):
        s = _State(rect_canvas=None)
        assert CropTool._hit_corner(s, 100, 100) is None

    def test_exact_tl_corner(self):
        s = self._state_with_rect()
        assert CropTool._hit_corner(s, 100, 100) == "tl"

    def test_exact_tr_corner(self):
        s = self._state_with_rect()
        assert CropTool._hit_corner(s, 400, 100) == "tr"

    def test_exact_bl_corner(self):
        s = self._state_with_rect()
        assert CropTool._hit_corner(s, 100, 300) == "bl"

    def test_exact_br_corner(self):
        s = self._state_with_rect()
        assert CropTool._hit_corner(s, 400, 300) == "br"

    def test_within_handle_radius(self):
        s = self._state_with_rect()
        # HANDLE_RADIUS pixels away from TL
        assert CropTool._hit_corner(s, 100 + HANDLE_RADIUS, 100) == "tl"

    def test_just_outside_handle_radius(self):
        s = self._state_with_rect()
        assert CropTool._hit_corner(s, 100 + HANDLE_RADIUS + 1, 100) is None

    def test_center_of_rect_returns_none(self):
        s = self._state_with_rect()
        assert CropTool._hit_corner(s, 250, 200) is None


class TestCanvasToOrigUpdate:
    def test_basic_conversion(self):
        s = _State(scale=0.5, orig_w=1280, orig_h=720)
        # canvas (100, 50) → orig (200, 100) at scale 0.5
        CropTool._canvas_to_orig_update(s, 0, 0, 100, 50)
        assert s.rect_orig == (0, 0, 200, 100)

    def test_swapped_coords_are_sorted(self):
        s = _State(scale=1.0, orig_w=1000, orig_h=800)
        CropTool._canvas_to_orig_update(s, 300, 200, 100, 50)
        x1, y1, x2, y2 = s.rect_orig
        assert x1 <= x2 and y1 <= y2

    def test_clamps_to_frame_bounds(self):
        s = _State(scale=1.0, orig_w=640, orig_h=480)
        CropTool._canvas_to_orig_update(s, 0, 0, 9000, 9000)
        x1, y1, x2, y2 = s.rect_orig
        assert x2 <= 640
        assert y2 <= 480

    def test_negative_canvas_coords_clamped_to_zero(self):
        s = _State(scale=1.0, orig_w=640, orig_h=480)
        CropTool._canvas_to_orig_update(s, -50, -50, 200, 150)
        x1, y1, x2, y2 = s.rect_orig
        assert x1 >= 0 and y1 >= 0

    def test_updates_rect_orig(self):
        s = _State(scale=1.0, orig_w=640, orig_h=480)
        CropTool._canvas_to_orig_update(s, 10, 20, 200, 100)
        assert s.rect_orig is not None

    def test_zero_scale_does_nothing(self):
        s = _State(scale=0.0)
        s.rect_orig = (1, 2, 3, 4)
        CropTool._canvas_to_orig_update(s, 0, 0, 100, 100)
        # Should return early without changing rect_orig
        assert s.rect_orig == (1, 2, 3, 4)
