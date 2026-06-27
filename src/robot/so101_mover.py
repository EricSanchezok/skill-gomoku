"""SO101 smooth movement helpers built around LeRobot action commands.

This is the reusable part of the locally tested ``gomoku_so101`` motion script.
It intentionally avoids writing motor speed/acceleration registers.  Smoothness
comes from sending small intermediate LeRobot action targets along a cubic
easing curve.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from src.robot.lerobot_calibration import install_bundled_lerobot_calibration

Action = dict[str, float]
ProgressCallback = Callable[[int, int, float], None]

DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_ROBOT_ID = "so101_follower_0610"
DEFAULT_MAX_RELATIVE_TARGET = 5.0
DEFAULT_DURATION_SECONDS = 5.0
DEFAULT_DT_SECONDS = 0.01

WAITING_ACTION: Action = {
    "elbow_flex.pos": 9.714285714285714,
    "gripper.pos": 6.731436502428869,
    "shoulder_lift.pos": -81.93406593406593,
    "shoulder_pan.pos": 5.450549450549451,
    "wrist_flex.pos": 76.08791208791209,
    "wrist_roll.pos": -0.04395604395604396,
}

CENTER_ACTION: Action = {
    "elbow_flex.pos": 51.472527472527474,
    "gripper.pos": 36.01665510062457,
    "shoulder_lift.pos": -13.802197802197803,
    "shoulder_pan.pos": 3.6923076923076925,
    "wrist_flex.pos": 14.10989010989011,
    "wrist_roll.pos": 1.4505494505494505,
}

PRESET_ACTIONS: dict[str, Action] = {
    "waiting": WAITING_ACTION,
    "center": CENTER_ACTION,
}


def smoothstep(x: float) -> float:
    """Cubic easing curve with zero slope at the start and end."""
    return x * x * (3.0 - 2.0 * x)


def interpolate_action(
    start: Mapping[str, float],
    target: Mapping[str, float],
    alpha: float,
) -> Action:
    """Interpolate a LeRobot action mapping key-by-key."""
    missing = set(target) - set(start)
    if missing:
        raise KeyError(f"Start action is missing target keys: {sorted(missing)}")
    return {
        key: float(start[key]) + (float(target[key]) - float(start[key])) * alpha
        for key in target
    }


def ensure_action(value: Mapping[str, float]) -> Action:
    """Validate and normalize a LeRobot action mapping."""
    if not value:
        raise ValueError("SO101 action cannot be empty")
    action = {str(key): float(item) for key, item in value.items()}
    invalid = sorted(key for key in action if not key.endswith(".pos"))
    if invalid:
        raise ValueError(f"SO101 action keys must end with '.pos': {invalid}")
    return action


@dataclass(frozen=True)
class MotionProfile:
    """Timing and safety parameters for point-to-point movement."""

    duration_seconds: float = DEFAULT_DURATION_SECONDS
    dt_seconds: float = DEFAULT_DT_SECONDS
    max_relative_target: float = DEFAULT_MAX_RELATIVE_TARGET

    def steps(self) -> int:
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive")
        return max(1, int(self.duration_seconds / self.dt_seconds))


class SO101SmoothMover:
    """Smooth point-to-point SO101 movement via ``SO101Follower.send_action``.

    The safe startup pattern is:

    1. Connect only the motor bus.
    2. Read the current pose.
    3. Send the current pose back as the hold target.
    4. Enable torque.
    5. Move with small eased action steps.
    """

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        robot_id: str = DEFAULT_ROBOT_ID,
        profile: MotionProfile | None = None,
        use_degrees: bool = True,
    ) -> None:
        install_bundled_lerobot_calibration(robot_id)
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError as exc:
            raise ImportError(
                "SO101SmoothMover requires the lerobot package. "
                "Run it inside the environment used to control the arm."
            ) from exc

        self.profile = profile or MotionProfile()
        config = SO101FollowerConfig(
            port=port,
            id=robot_id,
            max_relative_target=self.profile.max_relative_target,
            disable_torque_on_disconnect=False,
            use_degrees=use_degrees,
        )
        self.robot: Any = SO101Follower(config)
        self._connected = False

    def connect(self) -> None:
        """Connect only the motor bus used for observations and actions."""
        if not self._connected:
            self.robot.bus.connect()
            self._connected = True

    def disconnect(self) -> None:
        """Disconnect while leaving torque state under explicit caller control."""
        if self._connected and self.robot.bus.is_connected:
            self.robot.bus.disconnect(disable_torque=False)
            self._connected = False

    def read_action(self, target_action: Mapping[str, float] | None = None) -> Action:
        """Read current joint positions for target keys, or all ``.pos`` keys."""
        obs = self.robot.get_observation()
        keys = list(target_action.keys()) if target_action is not None else [
            key for key in obs if key.endswith(".pos")
        ]
        missing = sorted(key for key in keys if key not in obs)
        if missing:
            raise RuntimeError(f"Target has keys not found in robot observation: {missing}")
        return {key: float(obs[key]) for key in keys}

    def hold_current(self, target_action: Mapping[str, float] | None = None) -> Action:
        """Send the current pose as the goal and enable torque to hold it."""
        current = self.read_action(target_action)
        self.robot.send_action(current)
        time.sleep(0.2)
        self.robot.bus.enable_torque(num_retry=5)
        time.sleep(0.5)
        return current

    def release(self) -> None:
        """Disable torque so the arm can be hand-guided."""
        self.robot.bus.disable_torque(num_retry=5)

    def move_to(
        self,
        target_action: Mapping[str, float],
        profile: MotionProfile | None = None,
        progress: ProgressCallback | None = None,
    ) -> Action:
        """Move smoothly from the current pose to ``target_action``."""
        target = ensure_action(target_action)
        active_profile = profile or self.profile
        start = self.read_action(target)
        steps = active_profile.steps()

        for idx in range(1, steps + 1):
            alpha = smoothstep(idx / steps)
            action = interpolate_action(start, target, alpha)
            self.robot.send_action(action)
            if progress is not None:
                progress(idx, steps, alpha)
            time.sleep(active_profile.dt_seconds)

        self.robot.send_action(target)
        time.sleep(0.5)
        return self.read_action(target)


def move_to_smoothly(
    target_action: Mapping[str, float],
    port: str = DEFAULT_PORT,
    robot_id: str = DEFAULT_ROBOT_ID,
    profile: MotionProfile | None = None,
    hold_after: bool = True,
) -> Action:
    """Convenience wrapper for demos and one-off tests."""
    mover = SO101SmoothMover(port=port, robot_id=robot_id, profile=profile)
    try:
        mover.connect()
        mover.hold_current(target_action)
        final = mover.move_to(target_action)
        if not hold_after:
            mover.release()
        return final
    except KeyboardInterrupt:
        mover.release()
        raise
    finally:
        mover.disconnect()


SO101SlowMover = SO101SmoothMover
