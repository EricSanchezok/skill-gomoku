"""Measured board-position to robot-pose mapping.

The current SO101 workflow uses a measured pose table instead of deriving
targets by interpolating four corners.  Each abstract board position maps
directly to the LeRobot action that was recorded for that position.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.robot.controller import CalibrationPoints, PoseLike, RobotPose
from src.utils.constants import CALIB_CORNER_ORDER


def _normalise_pose(pose: PoseLike) -> RobotPose:
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


class BoardPoseMapper(Protocol):
    """Abstract board position to robot pose mapper."""

    @property
    def board_rows(self) -> int:
        """Number of mapped board rows."""

    @property
    def board_cols(self) -> int:
        """Number of mapped board columns."""

    @property
    def coordinate_space(self) -> str:
        """Coordinate space of returned robot poses."""

    def target_for_cell(self, row: int, col: int) -> RobotPose:
        """Return the robot pose for a 0-based board cell."""


class RobotPoseMover(Protocol):
    """Minimal movement interface used by corner replay helpers."""

    def move_to(self, target_pose: Mapping[str, float]) -> Any:
        """Move the robot to ``target_pose``."""


@dataclass(frozen=True)
class PoseTarget:
    """A measured robot target for one abstract board position."""

    label: str
    row: int
    col: int
    pose: RobotPose


class MeasuredBoardPoseMapper:
    """Direct lookup mapper backed by measured robot poses.

    The source JSON format matches ``so101_board_81_positions.json``:

    ``positions.r1c1.action`` is the recorded robot pose for row 1, column 1.
    Public lookup methods use 0-based row/col by default to match the rest of
    the codebase.
    """

    def __init__(
        self,
        records: Iterable[PoseTarget],
        board_rows: int,
        board_cols: int,
        coordinate_space: str = "robot_pose",
        source_path: str | Path | None = None,
    ) -> None:
        if board_rows <= 0 or board_cols <= 0:
            raise ValueError("board_rows and board_cols must be positive")

        self._board_rows = int(board_rows)
        self._board_cols = int(board_cols)
        self._coordinate_space = coordinate_space
        self._source_path = Path(source_path) if source_path is not None else None
        self._records: dict[tuple[int, int], PoseTarget] = {}
        self._labels: dict[str, tuple[int, int]] = {}

        for record in records:
            self._add_record(record)

        self._validate_complete()

    @classmethod
    def from_json_file(cls, path: str | Path) -> MeasuredBoardPoseMapper:
        """Load a measured pose table from JSON."""
        source = Path(path)
        with open(source, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_json_data(data, source_path=source)

    @classmethod
    def from_json_data(
        cls,
        data: Mapping[str, Any],
        source_path: str | Path | None = None,
    ) -> MeasuredBoardPoseMapper:
        """Create a mapper from a parsed measured-pose JSON dictionary."""
        positions = data.get("positions")
        if not isinstance(positions, Mapping) or not positions:
            raise ValueError("Measured pose JSON must contain a non-empty 'positions' mapping")

        board_info = data.get("board", {})
        if not isinstance(board_info, Mapping):
            board_info = {}

        rows, cols = _infer_board_shape(board_info, positions)
        coordinate_space = str(data.get("coordinate_space", "robot_pose"))

        records: list[PoseTarget] = []
        for label, raw_record in positions.items():
            if not isinstance(raw_record, Mapping):
                raise ValueError(f"Position {label!r} must be a mapping")
            row, col = _record_indices(str(label), raw_record)
            action = raw_record.get("action", raw_record.get("pose"))
            if action is None:
                raise ValueError(f"Position {label!r} is missing 'action' or 'pose'")
            records.append(
                PoseTarget(
                    label=str(label),
                    row=row,
                    col=col,
                    pose=_normalise_pose(action),
                )
            )

        return cls(
            records=records,
            board_rows=rows,
            board_cols=cols,
            coordinate_space=coordinate_space,
            source_path=source_path,
        )

    @property
    def board_rows(self) -> int:
        return self._board_rows

    @property
    def board_cols(self) -> int:
        return self._board_cols

    @property
    def coordinate_space(self) -> str:
        return self._coordinate_space

    @property
    def source_path(self) -> Path | None:
        return self._source_path

    def target_for_cell(self, row: int, col: int) -> RobotPose:
        """Return the measured robot pose for a 0-based board cell."""
        self._check_bounds(row, col)
        try:
            return _clone_pose(self._records[(row, col)].pose)
        except KeyError as exc:
            raise KeyError(f"No measured robot pose for cell ({row}, {col})") from exc

    def target_for_position(self, row: int, col: int, one_based: bool = False) -> RobotPose:
        """Return the measured pose for a row/col pair.

        Args:
            row: Row index. 0-based unless ``one_based`` is true.
            col: Column index. 0-based unless ``one_based`` is true.
            one_based: Interpret row/col as human-facing 1-based coordinates.
        """
        if one_based:
            row -= 1
            col -= 1
        return self.target_for_cell(row, col)

    def target_for_label(self, label: str) -> RobotPose:
        """Return the measured pose for a label such as ``r1c1``."""
        try:
            row, col = self._labels[label]
        except KeyError as exc:
            raise KeyError(f"Unknown measured board position label: {label}") from exc
        return self.target_for_cell(row, col)

    def pose_target_for_cell(self, row: int, col: int) -> PoseTarget:
        """Return metadata and pose for a 0-based board cell."""
        self._check_bounds(row, col)
        record = self._records[(row, col)]
        return PoseTarget(
            label=record.label,
            row=record.row,
            col=record.col,
            pose=_clone_pose(record.pose),
        )

    def corner_cells(self) -> dict[str, tuple[int, int]]:
        """Return the four board-corner cells in calibration order names."""
        return {
            "top_left": (0, 0),
            "top_right": (0, self._board_cols - 1),
            "bottom_right": (self._board_rows - 1, self._board_cols - 1),
            "bottom_left": (self._board_rows - 1, 0),
        }

    def corner_targets(
        self,
        order: Sequence[str] = CALIB_CORNER_ORDER,
    ) -> list[PoseTarget]:
        """Return measured corner targets in the requested order."""
        cells = self.corner_cells()
        targets = []
        for name in order:
            if name not in cells:
                raise ValueError(f"Unknown corner name: {name}")
            row, col = cells[name]
            targets.append(self.pose_target_for_cell(row, col))
        return targets

    def corner_calibration_points(self) -> CalibrationPoints:
        """Expose measured corners as ``CalibrationPoints`` for legacy callers."""
        return CalibrationPoints.from_corners(
            {
                name: self.target_for_cell(row, col)
                for name, (row, col) in self.corner_cells().items()
            }
        )

    def replay_corners(
        self,
        mover: RobotPoseMover,
        order: Sequence[str] = CALIB_CORNER_ORDER,
        before_move: Callable[[str, PoseTarget], None] | None = None,
    ) -> list[Any]:
        """Move through the measured four corners in order.

        This is the "restore the four corners" step used to physically locate
        the board before camera board calibration.  The mapper only emits the
        measured poses; safety prompts and connection/torque handling stay in
        the concrete mover or script.
        """
        results = []
        for name, target in zip(order, self.corner_targets(order), strict=True):
            if before_move is not None:
                before_move(name, target)
            pose = target.pose
            if not isinstance(pose, Mapping):
                raise TypeError("replay_corners requires mapping poses for the configured mover")
            results.append(mover.move_to(pose))
        return results

    def _add_record(self, record: PoseTarget) -> None:
        self._check_bounds(record.row, record.col)
        key = (record.row, record.col)
        if key in self._records:
            raise ValueError(f"Duplicate measured pose for cell {key}")
        if record.label in self._labels:
            raise ValueError(f"Duplicate measured pose label: {record.label}")
        normalised = PoseTarget(
            label=record.label,
            row=record.row,
            col=record.col,
            pose=_normalise_pose(record.pose),
        )
        self._records[key] = normalised
        self._labels[normalised.label] = key

    def _check_bounds(self, row: int, col: int) -> None:
        if not (0 <= row < self._board_rows and 0 <= col < self._board_cols):
            raise IndexError(
                f"Cell ({row}, {col}) out of range for "
                f"{self._board_rows}x{self._board_cols} measured pose map"
            )

    def _validate_complete(self) -> None:
        missing = [
            (row, col)
            for row in range(self._board_rows)
            for col in range(self._board_cols)
            if (row, col) not in self._records
        ]
        if missing:
            preview = ", ".join(f"({row}, {col})" for row, col in missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            raise ValueError(f"Measured pose map is missing cells: {preview}{suffix}")


def load_pose_mapper_from_config(
    config: Mapping[str, Any],
    base_dir: str | Path = ".",
) -> MeasuredBoardPoseMapper | None:
    """Load the measured pose mapper configured under ``robot.pose_map``.

    Supported forms:

    ```yaml
    robot:
      pose_map:
        method: measured
        path: so101_board_81_positions.json
    ```

    A legacy shorthand ``robot.pose_map_path`` is also accepted.
    """
    robot_cfg = config.get("robot", {})
    if not isinstance(robot_cfg, Mapping):
        return None

    pose_map_cfg = robot_cfg.get("pose_map")
    if isinstance(pose_map_cfg, Mapping):
        if pose_map_cfg.get("enabled", True) is False:
            return None
        method = str(pose_map_cfg.get("method", "measured"))
        if method != "measured":
            raise ValueError(f"Unsupported robot.pose_map method: {method}")
        path_value = pose_map_cfg.get("path")
    else:
        path_value = robot_cfg.get("pose_map_path")

    if not path_value:
        return None

    path = Path(str(path_value))
    if not path.is_absolute():
        path = Path(base_dir) / path
    return MeasuredBoardPoseMapper.from_json_file(path)


def _infer_board_shape(
    board_info: Mapping[str, Any],
    positions: Mapping[str, Any],
) -> tuple[int, int]:
    rows_value = board_info.get("rows")
    cols_value = board_info.get("cols")
    size_value = board_info.get("size")

    if rows_value is not None and cols_value is not None:
        return int(rows_value), int(cols_value)
    if size_value is not None:
        size = int(size_value)
        return size, size

    rows = []
    cols = []
    for label, raw_record in positions.items():
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"Position {label!r} must be a mapping")
        row, col = _record_indices(str(label), raw_record)
        rows.append(row)
        cols.append(col)

    return max(rows) + 1, max(cols) + 1


def _record_indices(label: str, raw_record: Mapping[str, Any]) -> tuple[int, int]:
    row_index = raw_record.get("row_index")
    col_index = raw_record.get("col_index")
    if row_index is not None and col_index is not None:
        return int(row_index), int(col_index)

    row_value = raw_record.get("row")
    col_value = raw_record.get("col")
    if row_value is not None and col_value is not None:
        return int(row_value) - 1, int(col_value) - 1

    if label.startswith("r") and "c" in label:
        row_label, col_label = label[1:].split("c", 1)
        return int(row_label) - 1, int(col_label) - 1

    raise ValueError(f"Cannot determine row/col indices for position {label!r}")
