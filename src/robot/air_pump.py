"""Air-pump suction controller for the Gomoku robot end effector."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from time import sleep
from typing import Any, Protocol

logger = logging.getLogger(__name__)

DEFAULT_VALVE_GPIO = 20
DEFAULT_PUMP_GPIO = 21
DEFAULT_MIN_PULSE_US = 544
DEFAULT_MAX_PULSE_US = 2400
DEFAULT_FRAME_MS = 20

DEFAULT_PICK_SECONDS = 1.0
DEFAULT_DROP_DELAY_SECONDS = 0.05


class SuctionController(Protocol):
    """End-effector operations used by the game orchestrator."""

    def pick_stone(self) -> None:
        """Start suction and wait until the stone is held."""

    def hold_stone(self) -> None:
        """Keep suction active while moving."""

    def drop_stone(self) -> None:
        """Release the stone at the target cell."""

    def off(self) -> None:
        """Return the suction hardware to a safe idle state."""


@dataclass(frozen=True)
class AirPumpConfig:
    valve_pin: int = DEFAULT_VALVE_GPIO
    pump_pin: int = DEFAULT_PUMP_GPIO
    min_pulse_us: int = DEFAULT_MIN_PULSE_US
    max_pulse_us: int = DEFAULT_MAX_PULSE_US
    frame_ms: int = DEFAULT_FRAME_MS
    pick_seconds: float = DEFAULT_PICK_SECONDS
    drop_delay_seconds: float = DEFAULT_DROP_DELAY_SECONDS
    dry_run: bool = False
    verbose: bool = True


class FakeServo:
    """Dry-run replacement for gpiozero.Servo."""

    def __init__(self, pin: int, **_: object) -> None:
        self.pin = pin
        self.value: float | None = None

    def close(self) -> None:
        pass


class AirPumpSuctionController:
    """Control the suction cup's valve and pump via Arduino-servo-style PWM.

    Real-machine validated truth table:
        - pick / hold: valve open, pump on
        - drop: close valve first so the stone falls, then turn pump off
        - off: valve closed, pump off
    """

    def __init__(self, config: AirPumpConfig | None = None) -> None:
        self.config = config or AirPumpConfig()
        self.valve_open_state = False
        self.pump_on_state = False

        if self.config.dry_run:
            servo_cls: Any = FakeServo
        else:
            try:
                import gpiozero
            except ImportError as exc:
                raise RuntimeError(
                    "Air pump control requires gpiozero on Raspberry Pi. "
                    "Set robot.air_pump.dry_run=true for non-hardware tests."
                ) from exc
            servo_cls = gpiozero.Servo

        servo_kwargs = {
            "min_pulse_width": self.config.min_pulse_us / 1_000_000,
            "max_pulse_width": self.config.max_pulse_us / 1_000_000,
            "frame_width": self.config.frame_ms / 1000,
            "initial_value": None,
        }
        self.valve = servo_cls(self.config.valve_pin, **servo_kwargs)
        self.pump = servo_cls(self.config.pump_pin, **servo_kwargs)
        self.off()

    @staticmethod
    def _angle_to_value(angle: float) -> float:
        angle = max(0.0, min(180.0, float(angle)))
        return angle / 90.0 - 1.0

    def _write_angle(self, servo: object, angle: float) -> None:
        setattr(servo, "value", self._angle_to_value(angle))

    def _log(self, message: str) -> None:
        if self.config.verbose:
            logger.info(
                "%s | valve=%s pump=%s",
                message,
                "open" if self.valve_open_state else "closed",
                "on" if self.pump_on_state else "off",
            )

    def valve_open(self) -> None:
        self._write_angle(self.valve, 180)
        self.valve_open_state = True
        self._log("Valve opened")

    def valve_close(self) -> None:
        self._write_angle(self.valve, 0)
        self.valve_open_state = False
        self._log("Valve closed")

    def pump_on(self) -> None:
        self._write_angle(self.pump, 180)
        self.pump_on_state = True
        self._log("Pump on")

    def pump_off(self) -> None:
        self._write_angle(self.pump, 0)
        self.pump_on_state = False
        self._log("Pump off")

    def pick_stone(self) -> None:
        """吸棋子：打开电磁阀并开启气泵，等待负压建立。"""
        self._log(f"Pick stone: valve open + pump on for {self.config.pick_seconds:.2f}s")
        self.valve_open()
        self.pump_on()
        sleep(self.config.pick_seconds)
        self.hold_stone()

    def hold_stone(self) -> None:
        """保持吸住：电磁阀和气泵都保持开启。"""
        self._log("Hold stone: valve open + pump on")
        self.valve_open()
        self.pump_on()

    def drop_stone(self) -> None:
        """放棋子：先关闭电磁阀让棋子落下，再关闭气泵降低噪音。"""
        self._log(
            f"Drop stone: close valve, then turn pump off after "
            f"{self.config.drop_delay_seconds:.2f}s"
        )
        self.valve_close()
        sleep(self.config.drop_delay_seconds)
        self.pump_off()

    def off(self) -> None:
        """Safe idle: valve closed, pump off."""
        self._write_angle(self.valve, 0)
        self._write_angle(self.pump, 0)
        self.valve_open_state = False
        self.pump_on_state = False
        self._log("Air pump off")

    def close(self) -> None:
        try:
            self.off()
            sleep(0.2)
        finally:
            self.valve.close()
            self.pump.close()

    def __enter__(self) -> AirPumpSuctionController:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def create_suction_controller_from_config(
    config: Mapping[str, Any],
) -> SuctionController | None:
    """Create the configured suction controller, or ``None`` when disabled."""
    robot_cfg = config.get("robot", {})
    if not isinstance(robot_cfg, Mapping):
        return None

    pump_cfg = robot_cfg.get("air_pump", {})
    if not isinstance(pump_cfg, Mapping) or not pump_cfg.get("enabled", False):
        return None

    controller_cfg = AirPumpConfig(
        valve_pin=int(pump_cfg.get("valve_pin", DEFAULT_VALVE_GPIO)),
        pump_pin=int(pump_cfg.get("pump_pin", DEFAULT_PUMP_GPIO)),
        min_pulse_us=int(pump_cfg.get("min_pulse_us", DEFAULT_MIN_PULSE_US)),
        max_pulse_us=int(pump_cfg.get("max_pulse_us", DEFAULT_MAX_PULSE_US)),
        frame_ms=int(pump_cfg.get("frame_ms", DEFAULT_FRAME_MS)),
        pick_seconds=float(pump_cfg.get("pick_seconds", DEFAULT_PICK_SECONDS)),
        drop_delay_seconds=float(pump_cfg.get("drop_delay_seconds", DEFAULT_DROP_DELAY_SECONDS)),
        dry_run=bool(pump_cfg.get("dry_run", False)),
        verbose=bool(pump_cfg.get("verbose", True)),
    )
    return AirPumpSuctionController(controller_cfg)
