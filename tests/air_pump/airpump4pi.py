#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
airpump4pi.py

For Raspberry Pi controlling an Arduino-Servo-signal-based air pump module.

Default wiring:
    GPIO20 -> valve signal
    GPIO21 -> pump signal

Original Arduino logic:
    servo_8.write(0)    -> valve off / closed
    servo_8.write(180)  -> valve on / open

    servo_9.write(0)    -> pump off
    servo_9.write(180)  -> pump on
"""

from __future__ import annotations

import argparse
import signal
import sys
from time import sleep
from typing import Optional

try:
    from gpiozero import Servo
except ImportError as exc:
    raise SystemExit(
        "Cannot import gpiozero.\n"
        "Try installing it with:\n"
        "  sudo apt update\n"
        "  sudo apt install python3-gpiozero\n"
    ) from exc


class AirPump4Pi:
    """
    Air pump controller using RC-servo-style PWM signal.

    State definition:
        off / 关闭:
            valve closed, pump off

        start_suction / 开始吸:
            valve closed, pump on

        hold / 吸住:
            valve closed, pump on by default
            You can set keep_pump=False if your vacuum system can hold suction
            without continuously running the pump.

        release / 松口:
            pump off, valve open for a short time, then everything off
    """

    # Arduino Servo default-ish parameters
    DEFAULT_MIN_PULSE_WIDTH = 544 / 1_000_000      # 544 us
    DEFAULT_MAX_PULSE_WIDTH = 2400 / 1_000_000     # 2400 us
    DEFAULT_FRAME_WIDTH = 20 / 1000                # 20 ms, 50 Hz

    def __init__(
        self,
        valve_pin: int = 20,
        pump_pin: int = 21,
        *,
        min_pulse_width: float = DEFAULT_MIN_PULSE_WIDTH,
        max_pulse_width: float = DEFAULT_MAX_PULSE_WIDTH,
        frame_width: float = DEFAULT_FRAME_WIDTH,
        initial_off: bool = True,
        verbose: bool = True,
    ) -> None:
        self.valve_pin = valve_pin
        self.pump_pin = pump_pin
        self.verbose = verbose

        self.valve = Servo(
            valve_pin,
            min_pulse_width=min_pulse_width,
            max_pulse_width=max_pulse_width,
            frame_width=frame_width,
            initial_value=None,
        )

        self.pump = Servo(
            pump_pin,
            min_pulse_width=min_pulse_width,
            max_pulse_width=max_pulse_width,
            frame_width=frame_width,
            initial_value=None,
        )

        if initial_off:
            self.off()

    # -------------------------
    # Low-level helpers
    # -------------------------

    @staticmethod
    def _angle_to_value(angle: float) -> float:
        """
        Convert Arduino Servo angle to gpiozero Servo value.

        Arduino:
            write(0)   -> minimum pulse
            write(90)  -> middle pulse
            write(180) -> maximum pulse

        gpiozero:
            value=-1   -> minimum pulse
            value=0    -> middle pulse
            value=1    -> maximum pulse
        """
        angle = max(0.0, min(180.0, float(angle)))
        return angle / 90.0 - 1.0

    def _write_angle(self, device: Servo, angle: float) -> None:
        device.value = self._angle_to_value(angle)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[AirPump4Pi] {message}")

    # -------------------------
    # Direct device operations
    # -------------------------

    def valve_close(self) -> None:
        """Close valve. Arduino equivalent: servo_8.write(0)."""
        self._write_angle(self.valve, 0)

    def valve_open(self) -> None:
        """Open valve. Arduino equivalent: servo_8.write(180)."""
        self._write_angle(self.valve, 180)

    def pump_off(self) -> None:
        """Turn pump off. Arduino equivalent: servo_9.write(0)."""
        self._write_angle(self.pump, 0)

    def pump_on(self) -> None:
        """Turn pump on. Arduino equivalent: servo_9.write(180)."""
        self._write_angle(self.pump, 180)

    # -------------------------
    # High-level actions
    # -------------------------

    def start_suction(self, duration: Optional[float] = None) -> None:
        """
        开始吸:
            valve closed, pump on.

        If duration is given, keep this state for `duration` seconds.
        """
        self._log("开始吸：电磁阀关闭，气泵开启")
        self.valve_close()
        self.pump_on()

        if duration is not None:
            sleep(duration)

    def hold(self, *, keep_pump: bool = True, duration: Optional[float] = None) -> None:
        """
        吸住:
            valve closed.

        By default, keep pump running for stronger and safer holding.
        If your suction cup can maintain vacuum after pumping, use keep_pump=False.
        """
        if keep_pump:
            self._log("吸住：电磁阀关闭，气泵保持开启")
            self.valve_close()
            self.pump_on()
        else:
            self._log("吸住：电磁阀关闭，气泵关闭，仅靠负压保持")
            self.valve_close()
            self.pump_off()

        if duration is not None:
            sleep(duration)

    def release(self, duration: float = 0.8, *, final_off: bool = True) -> None:
        """
        松口:
            pump off, valve open for `duration` seconds.

        By default, after releasing, return to off state:
            valve closed, pump off.
        """
        self._log(f"松口：气泵关闭，电磁阀开启 {duration:.2f} 秒")
        self.pump_off()
        self.valve_open()
        sleep(duration)

        if final_off:
            self.off()

    def off(self) -> None:
        """
        关闭:
            valve closed, pump off.

        Arduino equivalent:
            servo_8.write(0)
            servo_9.write(0)
        """
        self._log("关闭：电磁阀关闭，气泵关闭")
        self.valve_close()
        self.pump_off()

    def suck_then_hold(
        self,
        suck_time: float = 1.0,
        *,
        keep_pump: bool = True,
        hold_time: Optional[float] = None,
    ) -> None:
        """
        First build vacuum, then hold.

        Typical usage:
            air.suck_then_hold(suck_time=1.0)
        """
        self.start_suction(duration=suck_time)
        self.hold(keep_pump=keep_pump, duration=hold_time)

    def suck_and_release(
        self,
        suck_time: float = 1.0,
        release_time: float = 0.8,
        *,
        count: int = 1,
        interval: float = 0.2,
    ) -> None:
        """
        Reproduce the original Arduino loop:
            suction for suck_time
            release for release_time
            off
            repeat count times
        """
        for i in range(count):
            self._log(f"循环 {i + 1}/{count}")
            self.start_suction(duration=suck_time)
            self.release(duration=release_time, final_off=True)

            if i != count - 1:
                sleep(interval)

    # -------------------------
    # Chinese aliases
    # -------------------------

    def 开始吸(self, duration: Optional[float] = None) -> None:
        self.start_suction(duration=duration)

    def 吸住(self, *, keep_pump: bool = True, duration: Optional[float] = None) -> None:
        self.hold(keep_pump=keep_pump, duration=duration)

    def 松口(self, duration: float = 0.8, *, final_off: bool = True) -> None:
        self.release(duration=duration, final_off=final_off)

    def 关闭(self) -> None:
        self.off()

    # -------------------------
    # Cleanup
    # -------------------------

    def close(self, *, safe_off: bool = True) -> None:
        """
        Release GPIO resources.

        safe_off=True will first send off signal briefly.
        """
        if safe_off:
            try:
                self.off()
                sleep(0.2)
            except Exception:
                pass

        self.valve.close()
        self.pump.close()
        self._log("GPIO resources released")

    def __enter__(self) -> "AirPump4Pi":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(safe_off=True)


def wait_forever() -> None:
    while True:
        sleep(1)


def run_shell(air: AirPump4Pi) -> None:
    print(
        "\nInteractive commands:\n"
        "  start / suck / 开始吸        开始吸，气泵开启\n"
        "  hold / 吸住                 吸住，默认气泵保持开启\n"
        "  hold-off                    吸住，但关闭气泵，仅靠负压保持\n"
        "  release / 松口              松口，电磁阀开启 0.8 秒后关闭\n"
        "  off / close / 关闭          全部关闭\n"
        "  cycle                       吸 1.0 秒，松口 0.8 秒\n"
        "  q / quit / exit             退出\n"
    )

    while True:
        try:
            cmd = input("airpump> ").strip().lower()
        except EOFError:
            break

        if cmd in {"q", "quit", "exit"}:
            break
        elif cmd in {"start", "suck", "开始吸"}:
            air.start_suction()
        elif cmd in {"hold", "吸住"}:
            air.hold(keep_pump=True)
        elif cmd in {"hold-off", "hold_off"}:
            air.hold(keep_pump=False)
        elif cmd in {"release", "松口"}:
            air.release(duration=0.8)
        elif cmd in {"off", "close", "关闭"}:
            air.off()
        elif cmd == "cycle":
            air.suck_and_release(suck_time=1.0, release_time=0.8, count=1)
        elif not cmd:
            continue
        else:
            print(f"Unknown command: {cmd}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Control air pump and valve on Raspberry Pi using servo-style PWM."
    )

    parser.add_argument(
        "command",
        choices=[
            "start",
            "suck",
            "hold",
            "release",
            "off",
            "cycle",
            "demo",
            "shell",
        ],
        help="Action to run.",
    )

    parser.add_argument("--valve-pin", type=int, default=20, help="BCM GPIO pin for valve.")
    parser.add_argument("--pump-pin", type=int, default=21, help="BCM GPIO pin for pump.")

    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help=(
            "Duration in seconds. For start/hold, if omitted, keep running until Ctrl+C. "
            "For release, default is 0.8s."
        ),
    )

    parser.add_argument("--suck-time", type=float, default=1.0, help="Suction time for cycle/demo.")
    parser.add_argument("--release-time", type=float, default=0.8, help="Release time for cycle/demo.")
    parser.add_argument("--count", type=int, default=1, help="Repeat count for cycle/demo.")

    parser.add_argument(
        "--no-keep-pump",
        action="store_true",
        help="For hold: close valve and turn pump off, relying on vacuum to hold.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable log messages.",
    )

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    interrupted = False

    def handle_signal(signum, frame):
        nonlocal interrupted
        interrupted = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        with AirPump4Pi(
            valve_pin=args.valve_pin,
            pump_pin=args.pump_pin,
            verbose=not args.quiet,
        ) as air:
            if args.command in {"start", "suck"}:
                air.start_suction(duration=args.time)

                if args.time is None:
                    print("保持开始吸状态。按 Ctrl+C 关闭。")
                    wait_forever()

            elif args.command == "hold":
                air.hold(
                    keep_pump=not args.no_keep_pump,
                    duration=args.time,
                )

                if args.time is None:
                    print("保持吸住状态。按 Ctrl+C 关闭。")
                    wait_forever()

            elif args.command == "release":
                air.release(
                    duration=0.8 if args.time is None else args.time,
                    final_off=True,
                )

            elif args.command == "off":
                air.off()
                sleep(0.3)

            elif args.command in {"cycle", "demo"}:
                air.suck_and_release(
                    suck_time=args.suck_time,
                    release_time=args.release_time,
                    count=args.count,
                )

            elif args.command == "shell":
                run_shell(air)

    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭气泵和电磁阀。")
        return 130 if interrupted else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())