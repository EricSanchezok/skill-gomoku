"""Manual robot board calibration utilities.

This module handles the per-game workflow where an operator releases the arm,
guides the end effector to the four board corners, and records the current
robot/controller pose at each corner.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import yaml

from src.robot.controller import CalibrationPoints, PoseLike, RobotPose
from src.utils.constants import CALIB_CORNER_ORDER

PrintFn = Callable[[str], None]
InputFn = Callable[[str], str]

CORNER_DISPLAY_NAMES = {
    "top_left": "top-left",
    "top_right": "top-right",
    "bottom_right": "bottom-right",
    "bottom_left": "bottom-left",
}


class ManualPoseSampler(Protocol):
    """Minimal interface needed by the manual corner calibration flow."""

    coordinate_space: str

    def prepare_manual_guidance(self) -> None:
        """Prepare the robot so a person can guide it by hand."""

    def read_current_pose(self) -> PoseLike:
        """Read the current robot/controller pose."""

    def finish_manual_guidance(self, hold: bool = False) -> None:
        """Finish calibration and optionally hold the current pose."""


class InputPoseSampler:
    """Terminal-only sampler useful for dry runs and tests without hardware."""

    coordinate_space = "manual_input"

    def __init__(self, input_fn: InputFn = input, print_fn: PrintFn = print) -> None:
        self._input = input_fn
        self._print = print_fn

    def prepare_manual_guidance(self) -> None:
        self._print("Manual input mode: enter numeric poses when prompted.")

    def read_current_pose(self) -> RobotPose:
        while True:
            raw = self._input("Pose (e.g. '120, 30, 45' or 'joint_a=1 joint_b=2'): ").strip()
            try:
                return parse_pose(raw)
            except ValueError as exc:
                self._print(f"Invalid pose: {exc}")

    def finish_manual_guidance(self, hold: bool = False) -> None:
        if hold:
            self._print("Manual input mode has no robot torque to hold.")


def parse_pose(raw: str) -> RobotPose:
    """Parse either a numeric sequence or key=value mapping from terminal text."""
    if not raw:
        raise ValueError("empty pose")

    tokens = [token for token in raw.replace(",", " ").split() if token]
    if any("=" in token for token in tokens):
        pose: dict[str, float] = {}
        for token in tokens:
            if "=" not in token:
                raise ValueError("mapping poses must use key=value for every token")
            key, value = token.split("=", 1)
            if not key:
                raise ValueError("mapping pose contains an empty key")
            pose[key] = float(value)
        return pose

    return tuple(float(token) for token in tokens)


def run_manual_robot_calibration(
    sampler: ManualPoseSampler,
    input_fn: InputFn = input,
    print_fn: PrintFn = print,
    hold_after: bool = False,
) -> CalibrationPoints:
    """Record top-left, top-right, bottom-right, bottom-left robot poses."""
    print_fn("Robot board calibration")
    print_fn("Guide the arm tip to each board corner, then press Enter to record it.")

    points: list[PoseLike] = []
    sampler.prepare_manual_guidance()
    try:
        for idx, corner in enumerate(CALIB_CORNER_ORDER, start=1):
            display = CORNER_DISPLAY_NAMES.get(corner, corner)
            input_fn(f"[{idx}/4] Move to {display}, then press Enter...")
            pose = sampler.read_current_pose()
            points.append(pose)
            print_fn(f"Recorded {corner}: {_format_pose(pose)}")

        return CalibrationPoints.from_list(points)
    finally:
        sampler.finish_manual_guidance(hold=hold_after)


def load_robot_calibration(config: Mapping[str, Any]) -> CalibrationPoints:
    """Load robot corner calibration from a parsed config dictionary."""
    robot_cfg = config.get("robot", {})
    if not isinstance(robot_cfg, Mapping):
        raise ValueError("robot config must be a mapping")

    calibration = robot_cfg.get("calibration", {})
    if isinstance(calibration, Mapping):
        corners = calibration.get("corners")
        if isinstance(corners, Mapping) and all(name in corners for name in CALIB_CORNER_ORDER):
            return CalibrationPoints.from_corners(corners)

    legacy_points = robot_cfg.get("calibration_points")
    if isinstance(legacy_points, Sequence) and not isinstance(legacy_points, str | bytes):
        if len(legacy_points) == 4:
            return CalibrationPoints.from_list(legacy_points)

    raise ValueError(
        "No robot calibration found. Run scripts/calibrate_robot_board.py before starting a game."
    )


def get_robot_z_height(config: Mapping[str, Any], default: float = 50.0) -> float:
    robot_cfg = config.get("robot", {})
    if isinstance(robot_cfg, Mapping) and "z_height" in robot_cfg:
        return float(robot_cfg["z_height"])
    return float(default)


def update_robot_calibration_config(
    config: dict[str, Any],
    calib: CalibrationPoints,
    coordinate_space: str = "robot_pose",
    z_height: float | None = None,
) -> dict[str, Any]:
    """Mutate and return a config dictionary with the latest robot calibration."""
    robot_cfg = config.setdefault("robot", {})
    if not isinstance(robot_cfg, dict):
        raise ValueError("robot config must be a mapping")

    robot_cfg["calibration"] = {
        "method": "manual",
        "coordinate_space": coordinate_space,
        "corners": _poses_for_yaml(calib.to_corners_dict()),
    }
    robot_cfg["calibration_points"] = _poses_for_yaml(calib.to_list())
    if z_height is not None:
        robot_cfg["z_height"] = float(z_height)
    return config


def save_robot_calibration(
    config_path: str | Path,
    calib: CalibrationPoints,
    coordinate_space: str = "robot_pose",
    z_height: float | None = None,
) -> None:
    """Write robot calibration into a YAML config file."""
    path = Path(config_path)
    with open(path) as f:
        config = yaml.safe_load(f) or {}

    update_robot_calibration_config(config, calib, coordinate_space, z_height)

    with open(path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _poses_for_yaml(value: RobotPose | list[RobotPose] | dict[str, RobotPose]) -> Any:
    if isinstance(value, list):
        return [_poses_for_yaml(item) for item in value]
    if isinstance(value, dict) and all(isinstance(item, (int, float)) for item in value.values()):
        return {key: float(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: _poses_for_yaml(item) for key, item in value.items()}
    return [float(item) for item in value]


def _format_pose(pose: PoseLike) -> str:
    if isinstance(pose, Mapping):
        return "{" + ", ".join(f"{key}: {float(value):.3f}" for key, value in pose.items()) + "}"
    return "(" + ", ".join(f"{float(value):.3f}" for value in pose) + ")"
