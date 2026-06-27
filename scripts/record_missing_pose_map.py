#!/usr/bin/env python3
"""Record missing measured SO101 poses in a board pose-map JSON file.

Default behavior:
  - Load robot.pose_map.path from config/default.yaml.
  - Find positions with missing/empty action entries.
  - Disable SO101 torque, then prompt for each missing position.
  - Save after every recorded position.

Examples:
  conda run -n lerobot python scripts/record_missing_pose_map.py --port /dev/ttyACM0

  conda run -n lerobot python scripts/record_missing_pose_map.py \
    --positions r3c7 r5c9 --overwrite --port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.game.play_area import parse_play_area_config  # noqa: E402
from src.robot.calibration import InputPoseSampler  # noqa: E402
from src.robot.so101_adapter import SO101PoseSampler  # noqa: E402
from src.robot.so101_mover import DEFAULT_MAX_RELATIVE_TARGET, DEFAULT_ROBOT_ID  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL_RE = re.compile(r"^r(?P<row>\d+)c(?P<col>\d+)$", re.IGNORECASE)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    config_path = _project_path(args.config)
    config = load_config(config_path)
    pose_map_path = _resolve_pose_map_path(args.pose_map, config)

    data = _load_pose_map(pose_map_path)
    rows, cols = _board_shape(data)
    positions = _ensure_positions(data)
    targets = _target_positions(args.positions, positions, rows, cols, overwrite=args.overwrite)

    play_area = parse_play_area_config(config.get("game", {}).get("play_area"))
    _print_summary(pose_map_path, rows, cols, targets, play_area)

    if args.list_only or not targets:
        return 0

    if not args.yes:
        answer = input("Press Enter to start recording, or type q to cancel > ").strip().lower()
        if answer == "q":
            return 130

    if not args.no_backup:
        backup_path = _backup_file(pose_map_path)
        print(f"Backup written to {backup_path}")

    sampler = _build_sampler(args, config)
    try:
        sampler.prepare_manual_guidance()
        for row, col in targets:
            label = _label(row, col)
            global_position = _global_position_text(play_area, row, col)
            input(
                f"\nMove the arm tip to {label} "
                f"(local row={row + 1}, col={col + 1}{global_position}), "
                "then press Enter to record > "
            )
            action = sampler.read_current_pose()
            positions[label] = {
                "row": row + 1,
                "col": col + 1,
                "row_index": row,
                "col_index": col,
                "action": _normalise_action(action),
                "note": args.note,
                "recorded_at": time.time(),
            }
            data["updated_at"] = time.time()
            if args.port:
                data["port"] = args.port
            _write_pose_map(pose_map_path, data)
            print(f"Recorded and saved {label}")

        print("\nDone.")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Already-recorded positions have been saved.")
        return 130
    finally:
        sampler.finish_manual_guidance(hold=args.hold_after)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML to read")
    parser.add_argument("--pose-map", default=None, help="Pose-map JSON path; default from config")
    parser.add_argument("--backend", choices=("so101", "input"), default="so101")
    parser.add_argument("--port", default=None, help="SO101 serial port")
    parser.add_argument("--robot-id", default=None, help="LeRobot calibration id")
    parser.add_argument("--max-relative-target", type=float, default=DEFAULT_MAX_RELATIVE_TARGET)
    parser.add_argument("--no-degrees", action="store_true", help="Do not request degree units")
    parser.add_argument(
        "--positions",
        nargs="*",
        default=None,
        help="Specific labels, e.g. r3c7 r5c9",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing actions")
    parser.add_argument("--note", default="", help="Note stored on each new record")
    parser.add_argument("--hold-after", action="store_true", help="Hold the final pose before exit")
    parser.add_argument("--no-backup", action="store_true", help="Do not create a .bak file first")
    parser.add_argument("--list-only", action="store_true", help="Only print target positions")
    parser.add_argument("--yes", action="store_true", help="Skip initial confirmation")
    return parser


def _build_sampler(args: argparse.Namespace, config: Mapping[str, Any]):
    if args.backend == "input":
        return InputPoseSampler()

    robot_cfg = config.get("robot", {})
    if not isinstance(robot_cfg, Mapping):
        robot_cfg = {}
    port = args.port or robot_cfg.get("port")
    robot_id = args.robot_id or robot_cfg.get("id", DEFAULT_ROBOT_ID)
    if not port:
        raise SystemExit("--port is required when robot.port is missing")

    args.port = str(port)
    return SO101PoseSampler(
        port=str(port),
        robot_id=str(robot_id),
        max_relative_target=args.max_relative_target,
        use_degrees=not args.no_degrees,
    )


def _resolve_pose_map_path(value: str | None, config: Mapping[str, Any]) -> Path:
    if value:
        return _project_path(value)

    robot_cfg = config.get("robot", {})
    if not isinstance(robot_cfg, Mapping):
        raise SystemExit("Missing robot config; pass --pose-map explicitly")
    pose_map_cfg = robot_cfg.get("pose_map", {})
    path_value = None
    if isinstance(pose_map_cfg, Mapping):
        path_value = pose_map_cfg.get("path")
    path_value = path_value or robot_cfg.get("pose_map_path")
    if not path_value:
        raise SystemExit("Missing robot.pose_map.path; pass --pose-map explicitly")
    return _project_path(str(path_value))


def _load_pose_map(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Pose map not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Pose map JSON root must be an object")
    return data


def _write_pose_map(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _backup_file(path: Path) -> Path:
    backup = path.with_suffix(path.suffix + f".bak-{int(time.time())}")
    shutil.copy2(path, backup)
    return backup


def _ensure_positions(data: dict[str, Any]) -> dict[str, Any]:
    positions = data.setdefault("positions", {})
    if not isinstance(positions, dict):
        raise ValueError("Pose map 'positions' must be an object")
    return positions


def _board_shape(data: Mapping[str, Any]) -> tuple[int, int]:
    board = data.get("board", {})
    if not isinstance(board, Mapping):
        board = {}
    rows = board.get("rows")
    cols = board.get("cols")
    size = board.get("size")
    if rows is not None and cols is not None:
        return int(rows), int(cols)
    if size is not None:
        size_int = int(size)
        return size_int, size_int
    return _infer_shape_from_positions(data.get("positions", {}))


def _infer_shape_from_positions(positions: Any) -> tuple[int, int]:
    if not isinstance(positions, Mapping) or not positions:
        return 9, 9
    max_row = 0
    max_col = 0
    for label, record in positions.items():
        row, col = _record_indices(str(label), record)
        max_row = max(max_row, row)
        max_col = max(max_col, col)
    return max_row + 1, max_col + 1


def _target_positions(
    requested: list[str] | None,
    positions: Mapping[str, Any],
    rows: int,
    cols: int,
    *,
    overwrite: bool,
) -> list[tuple[int, int]]:
    if requested:
        targets = [_parse_position_token(token, rows, cols) for token in requested]
    else:
        targets = [
            (row, col)
            for row in range(rows)
            for col in range(cols)
            if not _has_action(positions.get(_label(row, col)))
        ]

    deduped: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for target in targets:
        if target in seen:
            continue
        seen.add(target)
        if requested and not overwrite and _has_action(positions.get(_label(*target))):
            print(f"Skipping {_label(*target)} because it already has an action; use --overwrite")
            continue
        deduped.append(target)
    return deduped


def _has_action(record: Any) -> bool:
    return isinstance(record, Mapping) and bool(record.get("action") or record.get("pose"))


def _parse_position_token(token: str, rows: int, cols: int) -> tuple[int, int]:
    stripped = token.strip()
    match = LABEL_RE.match(stripped)
    if match:
        row = int(match.group("row")) - 1
        col = int(match.group("col")) - 1
    elif "," in stripped:
        row_text, col_text = stripped.split(",", 1)
        row = int(row_text) - 1
        col = int(col_text) - 1
    else:
        raise ValueError(f"Invalid position {token!r}; use r3c7 or 3,7")

    if not (0 <= row < rows and 0 <= col < cols):
        raise ValueError(f"Position {token!r} is outside {rows}x{cols}")
    return row, col


def _record_indices(label: str, record: Any) -> tuple[int, int]:
    if isinstance(record, Mapping):
        row_index = record.get("row_index")
        col_index = record.get("col_index")
        if row_index is not None and col_index is not None:
            return int(row_index), int(col_index)
        row_value = record.get("row")
        col_value = record.get("col")
        if row_value is not None and col_value is not None:
            return int(row_value) - 1, int(col_value) - 1

    match = LABEL_RE.match(label)
    if match:
        return int(match.group("row")) - 1, int(match.group("col")) - 1
    raise ValueError(f"Cannot infer row/col for {label!r}")


def _normalise_action(action: Any) -> dict[str, float]:
    if not isinstance(action, Mapping):
        raise TypeError("Recorded action must be a mapping for SO101 pose maps")
    normalised = {str(key): float(value) for key, value in action.items()}
    invalid = sorted(key for key in normalised if not key.endswith(".pos"))
    if invalid:
        raise ValueError(f"Recorded action keys must end with '.pos': {invalid}")
    return normalised


def _print_summary(
    path: Path,
    rows: int,
    cols: int,
    targets: list[tuple[int, int]],
    play_area,
) -> None:
    print(f"Pose map: {path}")
    print(f"Pose map board: {rows}x{cols}")
    print(f"Config play_area: {play_area.describe()}")
    if not targets:
        print("No target positions to record.")
        return
    labels = " ".join(_label(row, col) for row, col in targets)
    print(f"Will record {len(targets)} position(s): {labels}")


def _global_position_text(play_area, row: int, col: int) -> str:
    if play_area.rows == row + 1 and play_area.cols == col + 1:
        return ""
    try:
        global_row, global_col = play_area.to_global(row, col)
    except ValueError:
        return ""
    return f"; full-board row={global_row + 1}, col={global_col + 1}"


def _label(row: int, col: int) -> str:
    return f"r{row + 1}c{col + 1}"


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
