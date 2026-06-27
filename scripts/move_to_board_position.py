#!/usr/bin/env python3
"""Move SO101 to one measured pose-map position via waiting pose."""

import argparse
import re
import sys
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.robot.pose_mapper import load_pose_mapper_from_config  # noqa: E402
from src.robot.so101_lowlevel_mover import (  # noqa: E402
    DEFAULT_LOWLEVEL_DT_SECONDS,
    DEFAULT_LOWLEVEL_DURATION_SECONDS,
    DEFAULT_ROBOT_ID,
    SO101LowLevelMover,
    make_lowlevel_profile,
)
from src.robot.so101_mover import (  # noqa: E402
    PRESET_ACTIONS,
)
from src.utils.config_loader import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LABEL_RE = re.compile(r"^r(?P<row>\d+)c(?P<col>\d+)$", re.IGNORECASE)
PAIR_RE = re.compile(r"^(?P<row>\d+)\s*,\s*(?P<col>\d+)$")


def main() -> int:
    args = parse_args()
    config = load_config(ROOT / args.config)
    mapper = load_pose_mapper_from_config(config, base_dir=ROOT)
    if mapper is None:
        raise SystemExit("robot.pose_map.path is required")

    row, col = parse_position(args.position)
    target = mapper.target_for_cell(row, col)
    if not isinstance(target, Mapping):
        raise TypeError("SO101 target must be a mapping action")
    target = {str(k): float(v) for k, v in target.items()}

    robot_cfg = config.get("robot", {})
    waiting_value = robot_cfg.get("waiting_pose", "waiting")
    waiting = dict(PRESET_ACTIONS[waiting_value]) if isinstance(waiting_value, str) else {
        str(k): float(v) for k, v in waiting_value.items()
    }
    print(f"{args.position} -> pose_map=r{row + 1}c{col + 1}")
    if args.dry_run:
        return 0

    port = args.port or robot_cfg.get("port")
    if not port:
        raise ValueError("--port is required when robot.port is missing")
    robot_id = args.robot_id or robot_cfg.get("id", DEFAULT_ROBOT_ID)
    profile = make_lowlevel_profile(
        duration_seconds=args.duration,
        dt_seconds=args.dt,
        lookahead_ticks=args.lookahead,
        pan_lookahead_ticks=args.pan_lookahead,
        lift_lookahead_ticks=args.lift_lookahead,
        elbow_lookahead_ticks=args.elbow_lookahead,
        wrist_flex_lookahead_ticks=args.wrist_flex_lookahead,
    )
    mover = SO101LowLevelMover(port=str(port), robot_id=str(robot_id), profile=profile)
    try:
        mover.connect()
        print("lock current")
        mover.hold_current(waiting)
        print("send waiting")
        mover.move_to(waiting, progress=progress)
        print("send target")
        mover.move_to(target, progress=progress)
        print("done")
        return 0
    finally:
        if args.release_after:
            mover.release()
        mover.disconnect()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("position", help="1-based pose-map position, e.g. r5c5 or 5,5")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--port")
    p.add_argument("--robot-id")
    p.add_argument("--duration", type=float, default=DEFAULT_LOWLEVEL_DURATION_SECONDS)
    p.add_argument("--dt", type=float, default=DEFAULT_LOWLEVEL_DT_SECONDS)
    p.add_argument("--lookahead", type=int, default=None)
    p.add_argument("--pan-lookahead", type=int, default=None)
    p.add_argument("--lift-lookahead", type=int, default=None)
    p.add_argument("--elbow-lookahead", type=int, default=None)
    p.add_argument("--wrist-flex-lookahead", type=int, default=None)
    p.add_argument("--release-after", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def parse_position(text):
    match = LABEL_RE.match(text.strip()) or PAIR_RE.match(text.strip())
    if match is None:
        raise ValueError("position must look like r5c5 or 5,5")
    row, col = int(match.group("row")), int(match.group("col"))
    if row < 1 or col < 1:
        raise ValueError("position is 1-based, so row and col must be >= 1")
    return row - 1, col - 1


def progress(i, total, alpha):
    if i == 1 or i % 50 == 0 or i == total:
        print(f"  step={i}/{total}, alpha={alpha:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
