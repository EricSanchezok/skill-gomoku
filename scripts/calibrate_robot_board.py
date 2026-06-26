#!/usr/bin/env python3
"""Record robot poses for the four Gomoku board corners.

Typical SO101 usage:

    conda run -n lerobot python scripts/calibrate_robot_board.py \
      --backend so101 \
      --port /dev/tty.usbmodem5A4B0487101 \
      --robot-id so101_follower_0610

Dry run without hardware:

    python scripts/calibrate_robot_board.py --backend input
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.robot.calibration import (  # noqa: E402
    InputPoseSampler,
    run_manual_robot_calibration,
    save_robot_calibration,
)
from src.utils.config_loader import load_config  # noqa: E402


def build_sampler(args: argparse.Namespace):
    if args.backend == "input":
        return InputPoseSampler()

    from src.robot.so101_adapter import SO101PoseSampler

    return SO101PoseSampler(
        port=args.port,
        robot_id=args.robot_id,
        max_relative_target=args.max_relative_target,
        use_degrees=not args.no_degrees,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual robot board-corner calibration")
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML to update")
    parser.add_argument("--backend", choices=("so101", "input"), default="so101")
    parser.add_argument("--port", default=None, help="SO101 serial port")
    parser.add_argument("--robot-id", default=None, help="LeRobot calibration id")
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    parser.add_argument("--no-degrees", action="store_true", help="Do not request degree units")
    parser.add_argument("--coordinate-space", default=None)
    parser.add_argument("--z-height", type=float, default=None)
    parser.add_argument("--hold-after", action="store_true", help="Enable torque after recording")
    parser.add_argument("--no-save", action="store_true", help="Print calibration but do not save")
    args = parser.parse_args()

    config = load_config(args.config)
    robot_cfg = config.get("robot", {})
    if args.backend == "so101":
        args.port = args.port or robot_cfg.get("port")
        args.robot_id = args.robot_id or robot_cfg.get("id", "so101_follower_0610")
        if not args.port:
            parser.error("--port is required for --backend so101 when robot.port is missing")

    sampler = build_sampler(args)
    coordinate_space = args.coordinate_space or getattr(sampler, "coordinate_space", "robot_pose")
    z_height = args.z_height
    if z_height is None and "z_height" in robot_cfg:
        z_height = float(robot_cfg["z_height"])

    calib = run_manual_robot_calibration(sampler, hold_after=args.hold_after)

    print("\nCalibration complete:")
    for name, pose in calib.to_corners_dict().items():
        print(f"  {name}: {pose}")

    if not args.no_save:
        save_robot_calibration(args.config, calib, coordinate_space, z_height)
        print(f"\nSaved robot calibration to {Path(args.config).resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
