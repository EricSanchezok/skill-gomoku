#!/usr/bin/env python3
"""Install bundled LeRobot calibration files into the local LeRobot cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.robot.lerobot_calibration import (  # noqa: E402
    DEFAULT_LEROBOT_ROBOT_TYPE,
    DEFAULT_SO101_ROBOT_ID,
    bundled_lerobot_calibration_path,
    install_bundled_lerobot_calibration,
    lerobot_calibration_cache_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-id", default=DEFAULT_SO101_ROBOT_ID)
    parser.add_argument("--robot-type", default=DEFAULT_LEROBOT_ROBOT_TYPE)
    parser.add_argument("--calibration-root", default=None, help="Defaults to LeRobot's cache root")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing cached file")
    args = parser.parse_args()

    source = bundled_lerobot_calibration_path(args.robot_id, args.robot_type)
    expected_target = lerobot_calibration_cache_path(
        args.robot_id,
        args.robot_type,
        calibration_root=args.calibration_root,
    )
    existed = expected_target.is_file()
    target = install_bundled_lerobot_calibration(
        args.robot_id,
        args.robot_type,
        overwrite=args.overwrite,
        calibration_root=args.calibration_root,
    )
    if target is None:
        parser.error(f"No bundled calibration file found at {source}")

    action = "Installed" if args.overwrite or not existed else "Found existing"
    print(f"{action} LeRobot calibration:")
    print(f"  source: {source}")
    print(f"  target: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
