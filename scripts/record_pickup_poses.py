#!/usr/bin/env python3
"""Record black/white stone pickup poses and pickup-top poses into the config YAML.

The live game uses ``robot.pickup_poses.<black|white>`` when present, and falls
back to legacy ``robot.pickup_pose`` only when the colour-specific pose is not
configured.

Example:
  conda run -n lerobot python scripts/record_pickup_poses.py --port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.robot.calibration import InputPoseSampler  # noqa: E402
from src.robot.so101_adapter import SO101PoseSampler  # noqa: E402
from src.robot.so101_mover import DEFAULT_MAX_RELATIVE_TARGET, DEFAULT_ROBOT_ID  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STONE_ORDER = ("black", "white")
STONE_LABELS = {
    "black": "black stone pickup pose / 黑棋取子位",
    "white": "white stone pickup pose / 白棋取子位",
}


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    config_path = _project_path(args.config)
    config = load_config(config_path)

    stones = _requested_stones(args.stones)
    _print_summary(config_path, stones, config)
    if args.list_only:
        return 0

    if not args.yes:
        answer = input("Press Enter to start recording pickup poses, or type q to cancel > ")
        if answer.strip().lower() == "q":
            return 130

    if not args.no_backup:
        backup_path = _backup_file(config_path)
        print(f"Backup written to {backup_path}")

    sampler = _build_sampler(args, config)
    try:
        sampler.prepare_manual_guidance()
        for stone in stones:
            input(
                f"\nMove the suction tip to {STONE_LABELS[stone]}, "
                "then press Enter to record > "
            )
            pose = _normalise_action(sampler.read_current_pose())
            _set_pickup_pose(config, stone, pose)
            input(
                f"Move the suction tip to {stone} pickup top pose / "
                f"{'黑棋' if stone == 'black' else '白棋'}取子上方安全位, "
                "then press Enter to record > "
            )
            top_pose = _normalise_action(sampler.read_current_pose())
            _set_pickup_top_pose(config, stone, top_pose)
            _write_config(config_path, config)
            print(f"Recorded {stone} pickup pose + top pose and saved config.")

        print("\nDone. Live game will pick the pose matching game.robot_stone.")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Already-recorded pickup poses have been saved.")
        return 130
    finally:
        sampler.finish_manual_guidance(hold=args.hold_after)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML to update")
    parser.add_argument("--backend", choices=("so101", "input"), default="so101")
    parser.add_argument("--port", default=None, help="SO101 serial port")
    parser.add_argument("--robot-id", default=None, help="LeRobot calibration id")
    parser.add_argument("--max-relative-target", type=float, default=DEFAULT_MAX_RELATIVE_TARGET)
    parser.add_argument("--no-degrees", action="store_true", help="Do not request degree units")
    parser.add_argument(
        "--stones",
        nargs="+",
        choices=STONE_ORDER,
        default=list(STONE_ORDER),
        help="Which pickup poses to record",
    )
    parser.add_argument("--hold-after", action="store_true", help="Hold final pose before exit")
    parser.add_argument("--no-backup", action="store_true", help="Do not create a .bak file first")
    parser.add_argument("--list-only", action="store_true", help="Only print current pickup status")
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

    return SO101PoseSampler(
        port=str(port),
        robot_id=str(robot_id),
        max_relative_target=args.max_relative_target,
        use_degrees=not args.no_degrees,
    )


def _requested_stones(values: list[str]) -> list[str]:
    requested = []
    for value in values:
        if value not in STONE_ORDER:
            raise ValueError(f"Unknown stone {value!r}")
        if value not in requested:
            requested.append(value)
    return requested


def _print_summary(config_path: Path, stones: list[str], config: Mapping[str, Any]) -> None:
    robot_cfg = config.get("robot", {})
    pickup_poses = robot_cfg.get("pickup_poses", {}) if isinstance(robot_cfg, Mapping) else {}
    if not isinstance(pickup_poses, Mapping):
        pickup_poses = {}

    print(f"Config: {config_path}")
    print(f"Will record: {', '.join(stones)}")
    for stone in STONE_ORDER:
        status = "configured" if isinstance(pickup_poses.get(stone), Mapping) else "missing"
        top_status = _pickup_top_status(robot_cfg, stone)
        print(f"  {stone}: pickup={status}, pickup_top={top_status}")


def _set_pickup_pose(config: dict[str, Any], stone: str, pose: Mapping[str, float]) -> None:
    robot_cfg = config.setdefault("robot", {})
    if not isinstance(robot_cfg, dict):
        raise ValueError("config.robot must be a mapping")
    pickup_poses = robot_cfg.setdefault("pickup_poses", {})
    if not isinstance(pickup_poses, dict):
        raise ValueError("config.robot.pickup_poses must be a mapping")
    pickup_poses[stone] = dict(pose)


def _set_pickup_top_pose(config: dict[str, Any], stone: str, pose: Mapping[str, float]) -> None:
    robot_cfg = config.setdefault("robot", {})
    if not isinstance(robot_cfg, dict):
        raise ValueError("config.robot must be a mapping")
    pickup_top_poses = robot_cfg.setdefault("pickup_top_poses", {})
    if not isinstance(pickup_top_poses, dict):
        raise ValueError("config.robot.pickup_top_poses must be a mapping")
    pickup_top_poses[stone] = dict(pose)


def _pickup_top_status(robot_cfg: Any, stone: str) -> str:
    if not isinstance(robot_cfg, Mapping):
        return "missing"
    pickup_top_poses = robot_cfg.get("pickup_top_poses", {})
    if not isinstance(pickup_top_poses, Mapping):
        return "missing"
    return "configured" if isinstance(pickup_top_poses.get(stone), Mapping) else "missing"


def _normalise_action(action: Any) -> dict[str, float]:
    if not isinstance(action, Mapping):
        raise TypeError("Recorded pickup pose must be a mapping")
    normalised = {str(key): float(value) for key, value in action.items()}
    invalid = sorted(key for key in normalised if not key.endswith(".pos"))
    if invalid:
        raise ValueError(f"Recorded action keys must end with '.pos': {invalid}")
    return normalised


def _write_config(path: Path, config: Mapping[str, Any]) -> None:
    yaml_text = yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True)
    path.write_text(yaml_text, encoding="utf-8")


def _backup_file(path: Path) -> Path:
    backup = path.with_suffix(path.suffix + f".bak-{int(time.time())}")
    shutil.copy2(path, backup)
    return backup


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
