"""Robot controller helpers for board-to-arm pose mapping.

The teammate-owned motion layer can represent a target either as a Cartesian
``(x, y, z)`` tuple or as a LeRobot action dictionary such as
``{"shoulder_pan.pos": 3.2, ...}``.  The calibration and interpolation helpers
below support both forms so the game loop does not care which low-level
controller is plugged in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.utils.constants import CALIB_CORNER_ORDER

PoseSequence = tuple[float, ...]
PoseMapping = dict[str, float]
PoseLike = Sequence[float] | Mapping[str, float]
RobotPose = PoseSequence | PoseMapping


def _normalise_pose(pose: PoseLike) -> RobotPose:
    """Convert a user/config pose into the internal immutable-ish shape."""
    if isinstance(pose, Mapping):
        if not pose:
            raise ValueError("Robot pose mapping cannot be empty")
        return {str(key): float(value) for key, value in pose.items()}

    if isinstance(pose, str | bytes):
        raise TypeError("Robot pose must be a numeric sequence or mapping")

    values = tuple(float(value) for value in pose)
    if not values:
        raise ValueError("Robot pose sequence cannot be empty")
    return values


def _clone_pose(pose: RobotPose) -> RobotPose:
    if isinstance(pose, Mapping):
        return dict(pose)
    return tuple(pose)


def _validate_same_structure(points: Sequence[RobotPose]) -> None:
    ref = points[0]
    if isinstance(ref, Mapping):
        ref_keys = set(ref.keys())
        for point in points[1:]:
            if not isinstance(point, Mapping):
                raise ValueError("All calibration poses must use the same structure")
            if set(point.keys()) != ref_keys:
                raise ValueError("All calibration pose mappings must have the same keys")
        return

    ref_len = len(ref)
    for point in points[1:]:
        if isinstance(point, Mapping) or len(point) != ref_len:
            raise ValueError("All calibration pose sequences must have the same length")


@dataclass(frozen=True)
class CalibrationPoints:
    """Four board-corner poses in robot/controller space.

    Corner order is top-left, top-right, bottom-right, bottom-left.  Each corner
    may be a Cartesian tuple or a LeRobot action mapping, but all four corners
    must use the same structure.
    """

    top_left: RobotPose
    top_right: RobotPose
    bottom_right: RobotPose
    bottom_left: RobotPose

    @classmethod
    def from_list(cls, points: Sequence[PoseLike]) -> CalibrationPoints:
        if len(points) != 4:
            raise ValueError("Expected 4 calibration points")
        normalised = [_normalise_pose(point) for point in points]
        _validate_same_structure(normalised)
        return cls(
            top_left=normalised[0],
            top_right=normalised[1],
            bottom_right=normalised[2],
            bottom_left=normalised[3],
        )

    @classmethod
    def from_corners(cls, corners: Mapping[str, PoseLike]) -> CalibrationPoints:
        missing = [name for name in CALIB_CORNER_ORDER if name not in corners]
        if missing:
            raise ValueError(f"Missing calibration corners: {', '.join(missing)}")
        return cls.from_list([corners[name] for name in CALIB_CORNER_ORDER])

    def to_list(self) -> list[RobotPose]:
        return [
            _clone_pose(self.top_left),
            _clone_pose(self.top_right),
            _clone_pose(self.bottom_right),
            _clone_pose(self.bottom_left),
        ]

    def to_corners_dict(self) -> dict[str, RobotPose]:
        return {
            name: _clone_pose(pose)
            for name, pose in zip(CALIB_CORNER_ORDER, self.to_list(), strict=True)
        }


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _bilinear(tl: float, tr: float, br: float, bl: float, u: float, v: float) -> float:
    top = _lerp(tl, tr, u)
    bottom = _lerp(bl, br, u)
    return _lerp(top, bottom, v)


def _interpolate_pose(calib: CalibrationPoints, u: float, v: float) -> RobotPose:
    tl = calib.top_left
    tr = calib.top_right
    br = calib.bottom_right
    bl = calib.bottom_left

    if isinstance(tl, Mapping):
        if not (
            isinstance(tr, Mapping) and isinstance(br, Mapping) and isinstance(bl, Mapping)
        ):
            raise ValueError("Calibration pose structures do not match")
        return {
            key: _bilinear(float(tl[key]), float(tr[key]), float(br[key]), float(bl[key]), u, v)
            for key in tl
        }

    if isinstance(tr, Mapping) or isinstance(br, Mapping) or isinstance(bl, Mapping):
        raise ValueError("Calibration pose structures do not match")

    return tuple(
        _bilinear(float(tl[idx]), float(tr[idx]), float(br[idx]), float(bl[idx]), u, v)
        for idx in range(len(tl))
    )


def _apply_z_height(pose: RobotPose, z_height: float | None) -> RobotPose:
    if z_height is None:
        return pose

    z_value = float(z_height)
    if isinstance(pose, Mapping):
        with_z = dict(pose)
        if "z" in with_z:
            with_z["z"] = z_value
        elif {"x", "y"}.issubset(with_z.keys()):
            with_z["z"] = z_value
        return with_z

    if len(pose) >= 3:
        return (pose[0], pose[1], z_value, *pose[3:])
    if len(pose) == 2:
        return (pose[0], pose[1], z_value)
    return pose


def board_to_robot_pose(
    row: int,
    col: int,
    board_rows: int,
    board_cols: int,
    calib: CalibrationPoints,
    z_height: float | None = None,
) -> RobotPose:
    """Map a board cell to a robot target pose via four-corner interpolation."""
    if not (0 <= row < board_rows and 0 <= col < board_cols):
        raise IndexError(f"Position ({row}, {col}) out of bounds")

    u = col / (board_cols - 1) if board_cols > 1 else 0.5
    v = row / (board_rows - 1) if board_rows > 1 else 0.5

    pose = _interpolate_pose(calib, u, v)
    return _apply_z_height(pose, z_height)


def board_to_robot_coords(
    row: int,
    col: int,
    board_rows: int,
    board_cols: int,
    calib: CalibrationPoints,
    z_height: float,
) -> tuple[float, float, float]:
    """将棋盘行列坐标转换为机械臂坐标系坐标。

    通过四点标定做双线性插值。

    Args:
        row: 棋盘行 (0-based)。
        col: 棋盘列 (0-based)。
        board_rows: 总行数 (15)。
        board_cols: 总列数 (15)。
        calib: 四点标定数据。
        z_height: 机械臂执行高度。

    Returns:
        (x, y, z) 机械臂坐标系坐标。
    """
    pose = board_to_robot_pose(row, col, board_rows, board_cols, calib, z_height)
    if isinstance(pose, Mapping):
        raise TypeError("board_to_robot_coords requires sequence calibration, got mapping pose")
    if len(pose) < 3:
        raise ValueError("Robot coordinate pose must have at least 3 values")
    return (float(pose[0]), float(pose[1]), float(pose[2]))
