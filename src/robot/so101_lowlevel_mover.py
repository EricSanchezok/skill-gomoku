"""Low-level SO101 movement using raw STS3215 goal positions.

The public mover interface mirrors ``SO101SmoothMover`` so the game layer can
keep using existing LeRobot action poses.  Internally those poses are converted
to raw STS3215 ticks and streamed with the v3 lookahead controller.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from src.robot.lerobot_calibration import (
    DEFAULT_SO101_ROBOT_ID,
    bundled_lerobot_calibration_path,
    lerobot_calibration_cache_path,
)
from src.robot.so101_mover import DEFAULT_PORT, Action

ProgressCallback = Callable[[int, int, float], None]

DEFAULT_ROBOT_ID = DEFAULT_SO101_ROBOT_ID
BAUDRATE = 1_000_000
PROTOCOL_VERSION = 0
STS3215_MODEL = 777
STS3215_MAX_RAW = 4095

MOTORS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}

BODY_MOTORS = set(MOTORS) - {"gripper"}

ADDR_MODEL = 3
ADDR_TORQUE = 40
ADDR_GOAL_POS = 42
ADDR_PRESENT_POS = 56
TORQUE_OFF = 0
TORQUE_ON = 1

DEFAULT_LOWLEVEL_DURATION_SECONDS = 6.0
DEFAULT_LOWLEVEL_DT_SECONDS = 0.02
DEFAULT_LOWLEVEL_SETTLE_SECONDS = 2.0
DEFAULT_LOWLEVEL_TOLERANCE_TICKS = 4
DEFAULT_LOWLEVEL_SMALL_MOVE_TICKS = 80
DEFAULT_LOWLEVEL_NORMAL_MOVE_TICKS = 500
DEFAULT_LOWLEVEL_SMALL_MOVE_SECONDS = 1.0
DEFAULT_LOWLEVEL_NORMAL_MOVE_SECONDS = 4.0
DEFAULT_LOWLEVEL_LOOKAHEAD_TICKS = {
    "shoulder_pan": 24,
    "shoulder_lift": 80,
    "elbow_flex": 60,
    "wrist_flex": 24,
    "wrist_roll": 8,
    "gripper": 8,
}


@dataclass(frozen=True)
class SO101MotorCalibration:
    id: int
    drive_mode: int
    range_min: int
    range_max: int


@dataclass(frozen=True)
class LowLevelMotionProfile:
    """Timing and v3 lookahead parameters verified on the SO101 arm."""

    duration_seconds: float = DEFAULT_LOWLEVEL_DURATION_SECONDS
    dt_seconds: float = DEFAULT_LOWLEVEL_DT_SECONDS
    settle_seconds: float = DEFAULT_LOWLEVEL_SETTLE_SECONDS
    tolerance_ticks: int = DEFAULT_LOWLEVEL_TOLERANCE_TICKS
    adaptive_duration: bool = True
    small_move_ticks: int = DEFAULT_LOWLEVEL_SMALL_MOVE_TICKS
    normal_move_ticks: int = DEFAULT_LOWLEVEL_NORMAL_MOVE_TICKS
    small_move_seconds: float = DEFAULT_LOWLEVEL_SMALL_MOVE_SECONDS
    normal_move_seconds: float = DEFAULT_LOWLEVEL_NORMAL_MOVE_SECONDS
    lookahead_ticks: Mapping[str, int] = field(
        default_factory=lambda: dict(DEFAULT_LOWLEVEL_LOOKAHEAD_TICKS)
    )

    def steps(self) -> int:
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive")
        return max(1, math.ceil(self.duration_seconds / self.dt_seconds))

    def settle_steps(self) -> int:
        if self.settle_seconds < 0:
            raise ValueError("settle_seconds cannot be negative")
        return max(1, round(self.settle_seconds / self.dt_seconds))

    def lookahead_for(self, motor: str) -> int:
        return int(self.lookahead_ticks.get(motor, DEFAULT_LOWLEVEL_LOOKAHEAD_TICKS[motor]))

    def duration_for_error(self, raw_error_ticks: int) -> float:
        if not self.adaptive_duration:
            return self.duration_seconds
        if raw_error_ticks <= self.small_move_ticks:
            return min(self.duration_seconds, self.small_move_seconds)
        if raw_error_ticks <= self.normal_move_ticks:
            return min(self.duration_seconds, self.normal_move_seconds)
        return self.duration_seconds

    def for_error(self, raw_error_ticks: int) -> LowLevelMotionProfile:
        duration = self.duration_for_error(raw_error_ticks)
        if duration == self.duration_seconds:
            return self
        return replace(self, duration_seconds=duration)


def load_lerobot_motor_calibration(
    robot_id: str = DEFAULT_ROBOT_ID,
    calibration_path: str | Path | None = None,
) -> dict[str, SO101MotorCalibration]:
    """Load the SO101 LeRobot calibration JSON without importing LeRobot."""

    if calibration_path is not None:
        path = Path(calibration_path).expanduser()
    else:
        path = bundled_lerobot_calibration_path(robot_id)
        if not path.is_file():
            path = lerobot_calibration_cache_path(robot_id)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    missing = sorted(set(MOTORS) - set(data))
    if missing:
        raise ValueError(f"Calibration file {path} is missing motors: {missing}")

    calibration = {}
    for motor, item in data.items():
        if motor not in MOTORS:
            continue
        calibration[motor] = SO101MotorCalibration(
            id=int(item["id"]),
            drive_mode=int(item.get("drive_mode", 0)),
            range_min=int(item["range_min"]),
            range_max=int(item["range_max"]),
        )
    return calibration


def lerobot_action_to_raw(
    action: Mapping[str, float],
    calibration: Mapping[str, SO101MotorCalibration],
) -> dict[str, int]:
    """Convert ``{motor}.pos`` LeRobot action values to raw STS3215 ticks."""

    raw = {}
    for key, value in ensure_action(action).items():
        motor = key.removesuffix(".pos")
        if motor not in calibration:
            raise KeyError(f"No SO101 calibration for motor {motor!r}")
        cal = calibration[motor]
        if motor == "gripper":
            norm = 100.0 - value if cal.drive_mode else value
            bounded = min(100.0, max(0.0, norm))
            raw_value = int((bounded / 100.0) * (cal.range_max - cal.range_min) + cal.range_min)
        else:
            mid = (cal.range_min + cal.range_max) / 2.0
            raw_value = int((value * STS3215_MAX_RAW / 360.0) + mid)
        if not (0 <= raw_value <= STS3215_MAX_RAW):
            raise ValueError(f"{key}={value} converts to out-of-range raw tick {raw_value}")
        raw[motor] = raw_value
    return raw


def raw_to_lerobot_action(
    raw_pose: Mapping[str, int],
    calibration: Mapping[str, SO101MotorCalibration],
) -> Action:
    """Convert raw STS3215 ticks back to LeRobot action values."""

    action = {}
    for motor, raw_value in raw_pose.items():
        cal = calibration[motor]
        if motor == "gripper":
            norm = ((int(raw_value) - cal.range_min) / (cal.range_max - cal.range_min)) * 100.0
            action[f"{motor}.pos"] = 100.0 - norm if cal.drive_mode else norm
        else:
            mid = (cal.range_min + cal.range_max) / 2.0
            action[f"{motor}.pos"] = (int(raw_value) - mid) * 360.0 / STS3215_MAX_RAW
    return action


def ensure_action(value: Mapping[str, float]) -> Action:
    if not value:
        raise ValueError("SO101 action cannot be empty")
    action = {str(key): float(item) for key, item in value.items()}
    invalid = sorted(key for key in action if not key.endswith(".pos"))
    if invalid:
        raise ValueError(f"SO101 action keys must end with '.pos': {invalid}")
    unknown = sorted(
        key.removesuffix(".pos")
        for key in action
        if key.removesuffix(".pos") not in MOTORS
    )
    if unknown:
        raise ValueError(f"Unknown SO101 action motors: {unknown}")
    return action


def minjerk(t: float) -> float:
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def interpolate_raw(
    start: Mapping[str, int],
    target: Mapping[str, int],
    alpha: float,
) -> dict[str, int]:
    return {motor: round(start[motor] + (target[motor] - start[motor]) * alpha) for motor in target}


def near_raw_target(
    actual: Mapping[str, int],
    target: Mapping[str, int],
    profile: LowLevelMotionProfile,
) -> dict[str, int]:
    limited = {}
    for motor, target_value in target.items():
        lead = profile.lookahead_for(motor)
        limited[motor] = max(actual[motor] - lead, min(actual[motor] + lead, target_value))
    return limited


def max_raw_error(actual: Mapping[str, int], target: Mapping[str, int]) -> int:
    return max(abs(actual[motor] - target[motor]) for motor in target)


def make_lowlevel_profile(
    *,
    duration_seconds: float = DEFAULT_LOWLEVEL_DURATION_SECONDS,
    dt_seconds: float = DEFAULT_LOWLEVEL_DT_SECONDS,
    settle_seconds: float = DEFAULT_LOWLEVEL_SETTLE_SECONDS,
    tolerance_ticks: int = DEFAULT_LOWLEVEL_TOLERANCE_TICKS,
    lookahead_ticks: int | None = None,
    pan_lookahead_ticks: int | None = None,
    lift_lookahead_ticks: int | None = None,
    elbow_lookahead_ticks: int | None = None,
    wrist_flex_lookahead_ticks: int | None = None,
    adaptive_duration: bool = True,
    small_move_ticks: int = DEFAULT_LOWLEVEL_SMALL_MOVE_TICKS,
    normal_move_ticks: int = DEFAULT_LOWLEVEL_NORMAL_MOVE_TICKS,
    small_move_seconds: float = DEFAULT_LOWLEVEL_SMALL_MOVE_SECONDS,
    normal_move_seconds: float = DEFAULT_LOWLEVEL_NORMAL_MOVE_SECONDS,
) -> LowLevelMotionProfile:
    lookahead = dict(DEFAULT_LOWLEVEL_LOOKAHEAD_TICKS)
    if lookahead_ticks is not None:
        lookahead = {motor: int(lookahead_ticks) for motor in MOTORS}
    overrides = {
        "shoulder_pan": pan_lookahead_ticks,
        "shoulder_lift": lift_lookahead_ticks,
        "elbow_flex": elbow_lookahead_ticks,
        "wrist_flex": wrist_flex_lookahead_ticks,
    }
    for motor, value in overrides.items():
        if value is not None:
            lookahead[motor] = int(value)
    return LowLevelMotionProfile(
        duration_seconds=duration_seconds,
        dt_seconds=dt_seconds,
        settle_seconds=settle_seconds,
        tolerance_ticks=tolerance_ticks,
        adaptive_duration=adaptive_duration,
        small_move_ticks=int(small_move_ticks),
        normal_move_ticks=int(normal_move_ticks),
        small_move_seconds=float(small_move_seconds),
        normal_move_seconds=float(normal_move_seconds),
        lookahead_ticks=lookahead,
    )


class SO101LowLevelMover:
    """SO101 point-to-point mover that bypasses ``SO101Follower.send_action``."""

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        robot_id: str = DEFAULT_ROBOT_ID,
        profile: LowLevelMotionProfile | None = None,
        calibration_path: str | Path | None = None,
    ) -> None:
        try:
            import scservo_sdk as scs
        except ImportError as exc:
            raise ImportError(
                "SO101LowLevelMover requires scservo_sdk. Run it inside the SO101 control env."
            ) from exc

        self.port_name = port
        self.robot_id = robot_id
        self.profile = profile or LowLevelMotionProfile()
        self.calibration = load_lerobot_motor_calibration(robot_id, calibration_path)
        self._scs: Any = scs
        self._port: Any = scs.PortHandler(port)
        self._packet: Any = scs.PacketHandler(PROTOCOL_VERSION)
        self._connected = False

    def connect(self) -> None:
        if self._connected:
            return
        if not self._port.openPort():
            raise RuntimeError(f"failed to open SO101 port {self.port_name}")
        self._connected = True
        try:
            if not self._port.setBaudRate(BAUDRATE):
                raise RuntimeError(f"failed to set SO101 baudrate {BAUDRATE}")
            for motor, motor_id in MOTORS.items():
                model = self._read2(motor_id, ADDR_MODEL)
                if model != STS3215_MODEL:
                    raise RuntimeError(
                        f"{motor} id={motor_id} model={model}, expected {STS3215_MODEL}"
                    )
        except Exception:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        if self._connected:
            self._port.closePort()
            self._connected = False

    def release(self) -> None:
        self._set_torque(TORQUE_OFF)

    def read_raw(self, motors: Mapping[str, float] | None = None) -> dict[str, int]:
        names = [key.removesuffix(".pos") for key in motors] if motors is not None else list(MOTORS)
        return {motor: self._read2(MOTORS[motor], ADDR_PRESENT_POS) for motor in names}

    def read_action(self, target_action: Mapping[str, float] | None = None) -> Action:
        raw = self.read_raw(target_action)
        return raw_to_lerobot_action(raw, self.calibration)

    def hold_current(self, target_action: Mapping[str, float] | None = None) -> Action:
        raw = self.read_raw()
        self._sync_goal(raw)
        self._set_torque(TORQUE_ON)
        time.sleep(0.3)
        if target_action is None:
            return raw_to_lerobot_action(raw, self.calibration)
        keys = {key.removesuffix(".pos") for key in target_action}
        return raw_to_lerobot_action({motor: raw[motor] for motor in keys}, self.calibration)

    def move_to(
        self,
        target_action: Mapping[str, float],
        profile: LowLevelMotionProfile | None = None,
        progress: ProgressCallback | None = None,
    ) -> Action:
        active_profile = profile or self.profile
        target = lerobot_action_to_raw(target_action, self.calibration)
        start = self.read_raw(target_action)
        active_profile = active_profile.for_error(max_raw_error(start, target))
        steps = active_profile.steps()
        started = time.monotonic()

        for idx in range(1, steps + 1):
            alpha = minjerk(idx / steps)
            planned = interpolate_raw(start, target, alpha)
            actual = self.read_raw(target_action)
            self._sync_goal(near_raw_target(actual, planned, active_profile))
            if progress is not None:
                progress(idx, steps, alpha)
            sleep_s = started + idx * active_profile.dt_seconds - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

        for _ in range(active_profile.settle_steps()):
            actual = self.read_raw(target_action)
            if max_raw_error(actual, target) <= active_profile.tolerance_ticks:
                break
            self._sync_goal(near_raw_target(actual, target, active_profile))
            time.sleep(active_profile.dt_seconds)

        return self.read_action(target_action)

    def _read2(self, motor_id: int, address: int) -> int:
        value, result, error = self._packet.read2ByteTxRx(self._port, motor_id, address)
        self._check(result, error, f"read id={motor_id} addr={address}")
        return int(value)

    def _write1(self, motor_id: int, address: int, value: int) -> None:
        result, error = self._packet.write1ByteTxRx(self._port, motor_id, address, int(value))
        self._check(result, error, f"write id={motor_id} addr={address}")

    def _set_torque(self, value: int) -> None:
        for motor_id in MOTORS.values():
            self._write1(motor_id, ADDR_TORQUE, value)

    def _sync_goal(self, raw_pose: Mapping[str, int]) -> None:
        writer = self._scs.GroupSyncWrite(self._port, self._packet, ADDR_GOAL_POS, 2)
        for motor, value in raw_pose.items():
            raw_value = int(value) & 0xFFFF
            ok = writer.addParam(
                MOTORS[motor],
                [self._scs.SCS_LOBYTE(raw_value), self._scs.SCS_HIBYTE(raw_value)],
            )
            if not ok:
                raise RuntimeError(f"failed to add sync-write param for {motor}")
        result = writer.txPacket()
        if result != self._scs.COMM_SUCCESS:
            raise RuntimeError(f"sync write failed: {self._packet.getTxRxResult(result)}")

    def _check(self, result: int, error: int, label: str) -> None:
        if result != self._scs.COMM_SUCCESS:
            raise RuntimeError(f"{label}: {self._packet.getTxRxResult(result)}")
        if error:
            raise RuntimeError(f"{label}: {self._packet.getRxPacketError(error)}")
