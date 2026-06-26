"""Bundled LeRobot calibration files for the SO101 arm."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

DEFAULT_SO101_ROBOT_ID = "so101_follower_0610"
DEFAULT_LEROBOT_ROBOT_TYPE = "so_follower"


def project_root() -> Path:
    """Return the skill-gomoku project root."""
    return Path(__file__).resolve().parents[2]


def bundled_lerobot_calibration_path(
    robot_id: str = DEFAULT_SO101_ROBOT_ID,
    robot_type: str = DEFAULT_LEROBOT_ROBOT_TYPE,
) -> Path:
    """Return the repo-tracked calibration JSON path for a LeRobot robot id."""
    return (
        project_root()
        / "calibration"
        / "lerobot"
        / "robots"
        / robot_type
        / f"{robot_id}.json"
    )


def lerobot_calibration_root() -> Path:
    """Mirror LeRobot's default calibration root without importing lerobot."""
    explicit_root = os.getenv("HF_LEROBOT_CALIBRATION")
    if explicit_root:
        return Path(explicit_root).expanduser()

    hf_lerobot_home = os.getenv("HF_LEROBOT_HOME")
    if hf_lerobot_home:
        return Path(hf_lerobot_home).expanduser() / "calibration"

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "lerobot" / "calibration"

    return Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"


def lerobot_calibration_cache_path(
    robot_id: str = DEFAULT_SO101_ROBOT_ID,
    robot_type: str = DEFAULT_LEROBOT_ROBOT_TYPE,
    calibration_root: str | Path | None = None,
) -> Path:
    """Return the LeRobot cache path for a robot calibration file."""
    root = (
        Path(calibration_root).expanduser()
        if calibration_root is not None
        else lerobot_calibration_root()
    )
    return root / "robots" / robot_type / f"{robot_id}.json"


def install_bundled_lerobot_calibration(
    robot_id: str = DEFAULT_SO101_ROBOT_ID,
    robot_type: str = DEFAULT_LEROBOT_ROBOT_TYPE,
    *,
    overwrite: bool = False,
    calibration_root: str | Path | None = None,
) -> Path | None:
    """Copy a repo-tracked LeRobot calibration into LeRobot's expected cache.

    Returns the installed or existing cache path. If this repo does not bundle
    the requested robot id, returns ``None`` so callers can continue with
    LeRobot's normal "missing calibration" error.
    """
    source = bundled_lerobot_calibration_path(robot_id, robot_type)
    if not source.is_file():
        return None

    target = lerobot_calibration_cache_path(robot_id, robot_type, calibration_root)
    if target.is_file() and not overwrite:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target
