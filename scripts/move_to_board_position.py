#!/usr/bin/env python3
"""Move SO101 to a chosen measured board position via the waiting pose.

Examples:
  python scripts/move_to_board_position.py r5c5
  python scripts/move_to_board_position.py 5,5 --yes
  python scripts/move_to_board_position.py 8,8 --space global
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.game.play_area import PlayArea, parse_play_area_config  # noqa: E402
from src.robot.pose_mapper import (  # noqa: E402
    MeasuredBoardPoseMapper,
    load_pose_mapper_from_config,
)
from src.robot.so101_mover import (  # noqa: E402
    DEFAULT_MAX_RELATIVE_TARGET,
    DEFAULT_ROBOT_ID,
    PRESET_ACTIONS,
    MotionProfile,
    SO101SmoothMover,
    suppress_lerobot_clamp_warnings,
)
from src.utils.config_loader import load_config  # noqa: E402
from src.utils.constants import BOARD_COLS, BOARD_ROWS  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSITION_LABEL_RE = re.compile(r"^r(?P<row>\d+)c(?P<col>\d+)$", re.IGNORECASE)
POSITION_PAIR_RE = re.compile(r"^(?P<row>\d+)\s*,\s*(?P<col>\d+)$")


@dataclass(frozen=True)
class ResolvedMoveTarget:
    token: str
    map_row: int
    map_col: int
    global_row: int | None
    global_col: int | None
    action: dict[str, float]

    @property
    def map_label(self) -> str:
        return f"r{self.map_row + 1}c{self.map_col + 1}"

    @property
    def global_label(self) -> str:
        if self.global_row is None or self.global_col is None:
            return "n/a"
        return f"r{self.global_row + 1}c{self.global_col + 1}"


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    suppress_lerobot_clamp_warnings(enabled=not args.show_clamp_warnings)

    config_path = _project_path(args.config)
    config = load_config(config_path)
    mapper = load_pose_mapper_from_config(config, base_dir=PROJECT_ROOT)
    if mapper is None:
        parser.error("robot.pose_map.path is required")

    play_area = parse_play_area_config(config.get("game", {}).get("play_area"))
    waiting_action = _resolve_waiting_action(config)
    port, robot_id = _resolve_robot_connection(config, args)

    profile = MotionProfile(
        duration_seconds=args.duration,
        dt_seconds=args.dt,
        max_relative_target=args.max_relative_target,
    )

    if args.positions:
        targets = [
            _resolve_position_target(token, mapper, play_area, args.space)
            for token in args.positions
        ]
        if args.dry_run:
            for target in targets:
                _print_target_summary(target)
            return 0
        return _run_positions(args, port, robot_id, profile, waiting_action, targets)

    if args.dry_run:
        parser.error("--dry-run needs at least one position")
    return _run_interactive(args, port, robot_id, profile, waiting_action, mapper, play_area)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("positions", nargs="*", help="Position tokens such as r5c5 or 5,5")
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML path")
    parser.add_argument("--port", default=None, help="Override robot.port")
    parser.add_argument("--robot-id", default=None, help="Override robot.id")
    parser.add_argument(
        "--space",
        choices=("local", "global"),
        default="local",
        help="local=1-based pose-map/play-window coordinates; global=1-based 15x15 coordinates",
    )
    parser.add_argument("--duration", type=float, default=5.0, help="SO101 move duration")
    parser.add_argument("--dt", type=float, default=0.01, help="SO101 move command interval")
    parser.add_argument("--max-relative-target", type=float, default=DEFAULT_MAX_RELATIVE_TARGET)
    parser.add_argument("--return-to-waiting", action="store_true")
    parser.add_argument("--release-after", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print targets only")
    parser.add_argument("--yes", action="store_true", help="Skip per-position confirmation")
    parser.add_argument(
        "--show-clamp-warnings",
        action="store_true",
        help="Show LeRobot max-relative-target clamp warnings.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print full target actions")
    return parser


def _run_interactive(
    args: argparse.Namespace,
    port: str,
    robot_id: str,
    profile: MotionProfile,
    waiting_action: Mapping[str, float],
    mapper: MeasuredBoardPoseMapper,
    play_area: PlayArea,
) -> int:
    mover = SO101SmoothMover(port=port, robot_id=robot_id, profile=profile)
    try:
        mover.connect()
        print("Holding current SO101 pose before manual move tests...")
        mover.hold_current(waiting_action)
        print(
            f"Interactive mode ({args.space}, 1-based). "
            "Type r5c5 / 5,5, or q to finish."
        )
        while True:
            token = input("position> ").strip()
            if token.lower() in {"q", "quit", "exit"}:
                break
            if not token:
                continue
            try:
                target = _resolve_position_target(token, mapper, play_area, args.space)
                _move_one(mover, waiting_action, target, args)
            except Exception as exc:
                print(f"ERROR: {exc}")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Disabling torque...")
        mover.release()
        return 130
    finally:
        if args.release_after:
            mover.release()
            print("Torque disabled.")
        mover.disconnect()


def _run_positions(
    args: argparse.Namespace,
    port: str,
    robot_id: str,
    profile: MotionProfile,
    waiting_action: Mapping[str, float],
    targets: list[ResolvedMoveTarget],
) -> int:
    mover = SO101SmoothMover(port=port, robot_id=robot_id, profile=profile)
    try:
        mover.connect()
        print("Holding current SO101 pose before board-position move...")
        mover.hold_current(waiting_action)
        for target in targets:
            _move_one(mover, waiting_action, target, args)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Disabling torque...")
        mover.release()
        return 130
    finally:
        if args.release_after:
            mover.release()
            print("Torque disabled.")
        mover.disconnect()


def _move_one(
    mover: SO101SmoothMover,
    waiting_action: Mapping[str, float],
    target: ResolvedMoveTarget,
    args: argparse.Namespace,
) -> None:
    _print_target_summary(target)
    if args.verbose:
        print(_format_action(target.action))
    if not args.yes:
        answer = input("Press Enter to move via waiting pose, or type q to skip > ")
        if answer.strip().lower() == "q":
            return

    print("Moving to waiting pose...")
    mover.move_to(waiting_action)
    print(f"Moving to {target.map_label}...")
    mover.move_to(target.action)
    if args.return_to_waiting:
        print("Returning to waiting pose...")
        mover.move_to(waiting_action)


def _resolve_position_target(
    token: str,
    mapper: MeasuredBoardPoseMapper,
    play_area: PlayArea,
    space: str,
) -> ResolvedMoveTarget:
    row, col = _parse_position_token(token)
    if space == "global":
        global_row, global_col = row, col
        if mapper.board_rows == BOARD_ROWS and mapper.board_cols == BOARD_COLS:
            map_row, map_col = global_row, global_col
        elif mapper.board_rows == play_area.rows and mapper.board_cols == play_area.cols:
            map_row, map_col = play_area.to_local(global_row, global_col)
        else:
            raise ValueError(
                f"Cannot map global position through {mapper.board_rows}x{mapper.board_cols} "
                f"pose map and play area {play_area.describe()}"
            )
    elif space == "local":
        map_row, map_col = row, col
        if mapper.board_rows == play_area.rows and mapper.board_cols == play_area.cols:
            global_row, global_col = play_area.to_global(map_row, map_col)
        elif mapper.board_rows == BOARD_ROWS and mapper.board_cols == BOARD_COLS:
            global_row, global_col = map_row, map_col
        else:
            global_row = None
            global_col = None
    else:
        raise ValueError(f"Unknown position space: {space}")

    action = mapper.target_for_cell(map_row, map_col)
    if not isinstance(action, Mapping):
        raise TypeError("SO101 board-position movement requires mapping action targets")
    return ResolvedMoveTarget(
        token=token,
        map_row=map_row,
        map_col=map_col,
        global_row=global_row,
        global_col=global_col,
        action={str(key): float(value) for key, value in action.items()},
    )


def _parse_position_token(token: str) -> tuple[int, int]:
    stripped = token.strip()
    match = POSITION_LABEL_RE.match(stripped)
    if match is None:
        match = POSITION_PAIR_RE.match(stripped)
    if match is None:
        raise ValueError(f"Position must look like r5c5 or 5,5, got {token!r}")

    row = int(match.group("row"))
    col = int(match.group("col"))
    if row <= 0 or col <= 0:
        raise ValueError("Positions are 1-based and must be positive")
    return row - 1, col - 1


def _resolve_waiting_action(config: Mapping[str, Any]) -> dict[str, float]:
    robot_cfg = config.get("robot", {})
    if not isinstance(robot_cfg, Mapping):
        robot_cfg = {}
    waiting_value = robot_cfg.get("waiting_pose", "waiting")
    if waiting_value is None:
        raise ValueError("robot.waiting_pose is required")
    return _resolve_action(waiting_value, "robot.waiting_pose")


def _resolve_action(value: Any, field_name: str) -> dict[str, float]:
    if isinstance(value, str):
        try:
            return dict(PRESET_ACTIONS[value])
        except KeyError as exc:
            raise ValueError(
                f"Unknown {field_name} preset {value!r}; available presets: "
                f"{sorted(PRESET_ACTIONS)}"
            ) from exc
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field_name} must be a preset name or non-empty action mapping")
    return {str(key): float(item) for key, item in value.items()}


def _resolve_robot_connection(
    config: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[str, str]:
    robot_cfg = config.get("robot", {})
    if not isinstance(robot_cfg, Mapping):
        robot_cfg = {}
    port = args.port or robot_cfg.get("port")
    robot_id = args.robot_id or robot_cfg.get("id", DEFAULT_ROBOT_ID)
    if not port:
        raise ValueError("--port is required when robot.port is missing")
    return str(port), str(robot_id)


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _print_target_summary(target: ResolvedMoveTarget) -> None:
    print(
        f"Target {target.token}: pose_map={target.map_label}, "
        f"global_15x15={target.global_label}, joints={len(target.action)}"
    )


def _format_action(action: Mapping[str, float]) -> str:
    return "\n".join(f"  {key}: {float(action[key]):.3f}" for key in sorted(action))


if __name__ == "__main__":
    raise SystemExit(main())
