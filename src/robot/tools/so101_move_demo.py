#!/usr/bin/env python3
"""Move the SO101 arm to a tested preset with the smooth mover."""

from __future__ import annotations

import argparse

from src.robot.so101_lowlevel_mover import (
    DEFAULT_LOWLEVEL_DT_SECONDS,
    DEFAULT_LOWLEVEL_DURATION_SECONDS,
    DEFAULT_PORT,
    DEFAULT_ROBOT_ID,
    SO101LowLevelMover,
    make_lowlevel_profile,
)
from src.robot.so101_mover import (
    PRESET_ACTIONS,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", choices=sorted(PRESET_ACTIONS), help="Tested target preset")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    parser.add_argument("--duration", type=float, default=DEFAULT_LOWLEVEL_DURATION_SECONDS)
    parser.add_argument("--dt", type=float, default=DEFAULT_LOWLEVEL_DT_SECONDS)
    parser.add_argument("--lookahead", type=int, default=None)
    parser.add_argument("--pan-lookahead", type=int, default=None)
    parser.add_argument("--lift-lookahead", type=int, default=None)
    parser.add_argument("--elbow-lookahead", type=int, default=None)
    parser.add_argument("--wrist-flex-lookahead", type=int, default=None)
    parser.add_argument("--release-after", action="store_true")
    args = parser.parse_args()

    profile = make_lowlevel_profile(
        duration_seconds=args.duration,
        dt_seconds=args.dt,
        lookahead_ticks=args.lookahead,
        pan_lookahead_ticks=args.pan_lookahead,
        lift_lookahead_ticks=args.lift_lookahead,
        elbow_lookahead_ticks=args.elbow_lookahead,
        wrist_flex_lookahead_ticks=args.wrist_flex_lookahead,
    )
    target = PRESET_ACTIONS[args.preset]
    mover = SO101LowLevelMover(port=args.port, robot_id=args.robot_id, profile=profile)

    try:
        mover.connect()
        print("Current pose:")
        current = mover.read_action(target)
        for key, value in current.items():
            print(f"  {key}: {value:.3f}")

        input("\nPress Enter to hold the current pose...")
        mover.hold_current(target)

        input(f"Press Enter to move slowly to preset '{args.preset}'...")
        final = mover.move_to(
            target,
            progress=lambda idx, steps, alpha: print(
                f"step={idx:03d}/{steps}, alpha={alpha:.3f}"
            ),
        )

        print("\nFinal pose:")
        for key, value in final.items():
            print(f"  {key}: {value:.3f}")

        if args.release_after:
            mover.release()
            print("\nTorque disabled.")
        else:
            print("\nTorque remains enabled; the arm should hold the target.")

        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Disabling torque...")
        mover.release()
        return 130
    finally:
        mover.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
