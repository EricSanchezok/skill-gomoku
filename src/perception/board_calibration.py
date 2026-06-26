"""Board-frame calibration loading and validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.constants import CALIB_CORNER_ORDER


@dataclass(frozen=True)
class BoardFrameCalibration:
    """Camera-image coordinates of the four board-frame corners."""

    corners: dict[str, tuple[float, float]]
    dst_size: int = 600

    def as_array(self) -> np.ndarray:
        """Return corners in OpenCV perspective-transform order."""
        return np.array([self.corners[name] for name in CALIB_CORNER_ORDER], dtype=np.float32)


def load_board_frame_calibration(
    config: Mapping[str, Any],
    *,
    required: bool = False,
) -> BoardFrameCalibration | None:
    """Load and validate ``board.calibration.corners`` from config.

    The expected corner order is top-left, top-right, bottom-right, bottom-left.
    These are image pixel coordinates recorded by ``scripts/calibrate_board.py``.
    """
    board_cfg = config.get("board", {})
    if not isinstance(board_cfg, Mapping):
        if required:
            raise ValueError("board config must be a mapping")
        return None

    calibration = board_cfg.get("calibration", {})
    if not isinstance(calibration, Mapping):
        if required:
            raise ValueError(_missing_message())
        return None

    method = str(calibration.get("method", "manual"))
    if method != "manual":
        if required:
            raise ValueError(f"Unsupported board.calibration.method for manual mode: {method}")
        return None

    raw_corners = calibration.get("corners")
    if not isinstance(raw_corners, Mapping):
        if required:
            raise ValueError(_missing_message())
        return None

    missing = [name for name in CALIB_CORNER_ORDER if name not in raw_corners]
    if missing:
        if required:
            raise ValueError(f"{_missing_message()} Missing corners: {', '.join(missing)}")
        return None

    corners = {name: _parse_corner(name, raw_corners[name]) for name in CALIB_CORNER_ORDER}
    _validate_corner_area(corners)

    return BoardFrameCalibration(
        corners=corners,
        dst_size=int(calibration.get("dst_size", 600)),
    )


def board_frame_required(config: Mapping[str, Any]) -> bool:
    """Return whether the configured perception path requires manual board corners."""
    board_cfg = config.get("board", {})
    det_cfg = config.get("board_detection", {})

    calibration = board_cfg.get("calibration", {}) if isinstance(board_cfg, Mapping) else {}
    method = None
    if isinstance(det_cfg, Mapping):
        method = det_cfg.get("method")
    if method is None and isinstance(calibration, Mapping):
        method = calibration.get("method")

    return str(method or "auto") == "manual"


def _parse_corner(name: str, value: Any) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 2:
        raise ValueError(f"Board corner {name!r} must be a two-number sequence")
    x = float(value[0])
    y = float(value[1])
    if not np.isfinite(x) or not np.isfinite(y):
        raise ValueError(f"Board corner {name!r} contains non-finite coordinates")
    return (x, y)


def _validate_corner_area(corners: Mapping[str, tuple[float, float]]) -> None:
    pts = [corners[name] for name in CALIB_CORNER_ORDER]
    twice_area = 0.0
    for idx, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(idx + 1) % len(pts)]
        twice_area += x1 * y2 - x2 * y1
    if abs(twice_area) < 1.0:
        raise ValueError(
            "Board calibration corners are degenerate; re-run scripts/calibrate_board.py"
        )


def _missing_message() -> str:
    return (
        "Manual board detection requires board.calibration.corners. "
        "Run scripts/calibrate_board.py to record the board frame first."
    )
