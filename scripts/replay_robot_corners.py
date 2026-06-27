#!/usr/bin/env python3
"""Replay measured robot poses for the four board corners.

Use this before camera board calibration to locate the physical board from the
already-measured SO101 corner poses.
"""

from __future__ import annotations

import argparse
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
from src.utils.config_loader import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML to read")
    parser.add_argument("--port", default=None, help="SO101 serial port")
    parser.add_argument("--robot-id", default=None, help="LeRobot calibration id")
    parser.add_argument("--duration", type=float, default=DEFAULT_LOWLEVEL_DURATION_SECONDS)
    parser.add_argument("--dt", type=float, default=DEFAULT_LOWLEVEL_DT_SECONDS)
    parser.add_argument("--lookahead", type=int, default=None)
    parser.add_argument("--pan-lookahead", type=int, default=None)
    parser.add_argument("--lift-lookahead", type=int, default=None)
    parser.add_argument("--elbow-lookahead", type=int, default=None)
    parser.add_argument("--wrist-flex-lookahead", type=int, default=None)
    parser.add_argument("--release-after", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Do not prompt before each corner")
    args = parser.parse_args()

    config = load_config(args.config)
    mapper = load_pose_mapper_from_config(config)
    if mapper is None:
        parser.error("robot.pose_map.path is required for measured corner replay")

    robot_cfg = config.get("robot", {})
    if not isinstance(robot_cfg, Mapping):
        robot_cfg = {}

    port = args.port or robot_cfg.get("port")
    robot_id = args.robot_id or robot_cfg.get("id", DEFAULT_ROBOT_ID)
    if not port:
        parser.error("--port is required when robot.port is missing")

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

    targets = mapper.corner_targets()
    try:
        mover.connect()
        first_pose = targets[0].pose
        if not isinstance(first_pose, Mapping):
            raise TypeError("SO101 corner replay requires mapping poses")

        print("Holding current pose before corner replay...")
        mover.hold_current(first_pose)

        for corner_name, target in zip(mapper.corner_cells(), targets, strict=True):
            pose = target.pose
            if not isinstance(pose, Mapping):
                raise TypeError("SO101 corner replay requires mapping poses")
            if not args.yes:
                input(f"Press Enter to move to {corner_name} ({target.label})...")
            print(f"Moving to {corner_name}: {target.label}")
            mover.move_to(pose)

        if args.release_after:
            mover.release()
            print("Torque disabled.")
        else:
            print("Corner replay complete; torque remains enabled.")

        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Disabling torque...")
        mover.release()
        return 130
    finally:
        mover.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
