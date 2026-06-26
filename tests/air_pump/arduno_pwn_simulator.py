#!/usr/bin/env python3
"""
More complete air pump / solenoid valve test for the Gomoku suction cup.

Hardware wiring uses BCM GPIO numbering:
    GPIO20 -> solenoid valve signal
    GPIO21 -> air pump signal

The module behaves like an Arduino Servo target:
    servo.write(0)   -> gpiozero Servo value -1
    servo.write(180) -> gpiozero Servo value 1

Useful suction states:
    吸棋子:
        valve closed, pump on, wait until vacuum is built.
    保持吸住:
        valve closed, pump on by default. Use --no-keep-pump only after confirming
        the suction cup can hold vacuum without continuously running the pump.
    放棋子:
        pump off, valve open for a short release pulse, then everything off.
"""

from __future__ import annotations

import argparse
import select
import signal
import sys
import termios
import tty
from dataclasses import dataclass
from time import sleep

try:
    from gpiozero import Servo as GpioServo
except ImportError:
    GpioServo = None


# Stable parameters from the existing Arduino-PWM simulator.
DEFAULT_VALVE_GPIO = 20
DEFAULT_PUMP_GPIO = 21
DEFAULT_MIN_PULSE_US = 544
DEFAULT_MAX_PULSE_US = 2400
DEFAULT_FRAME_MS = 20

DEFAULT_PICK_SECONDS = 1.0
DEFAULT_HOLD_SECONDS = 1.0
DEFAULT_RELEASE_SECONDS = 0.8
DEFAULT_INTERVAL_SECONDS = 0.2


@dataclass
class AirPumpTiming:
    pick_seconds: float = DEFAULT_PICK_SECONDS
    hold_seconds: float = DEFAULT_HOLD_SECONDS
    release_seconds: float = DEFAULT_RELEASE_SECONDS
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS


@dataclass
class AirPumpState:
    valve_open: bool = False
    pump_on: bool = False

    def as_text(self) -> str:
        valve = "开" if self.valve_open else "关"
        pump = "开" if self.pump_on else "关"
        return f"电磁阀={valve}, 气泵={pump}"


class FakeServo:
    """Small dry-run replacement so the script can be tested without GPIO."""

    def __init__(self, pin: int, **_: object) -> None:
        self.pin = pin
        self.value: float | None = None

    def close(self) -> None:
        pass


class AirPumpRig:
    """
    Test rig for direct hardware control and semantic suction actions.

    Low-level truth table:
        valve closed + pump off -> safe off
        valve closed + pump on  -> build or keep vacuum
        valve open   + pump off -> release vacuum
    """

    def __init__(
        self,
        valve_pin: int = DEFAULT_VALVE_GPIO,
        pump_pin: int = DEFAULT_PUMP_GPIO,
        *,
        min_pulse_us: int = DEFAULT_MIN_PULSE_US,
        max_pulse_us: int = DEFAULT_MAX_PULSE_US,
        frame_ms: int = DEFAULT_FRAME_MS,
        keep_pump_during_hold: bool = True,
        dry_run: bool = False,
        verbose: bool = True,
    ) -> None:
        if not dry_run and GpioServo is None:
            raise SystemExit(
                "Cannot import gpiozero. Run this on the Raspberry Pi, or use --dry-run.\n"
                "Install on Raspberry Pi with:\n"
                "  sudo apt update\n"
                "  sudo apt install python3-gpiozero"
            )

        servo_cls = FakeServo if dry_run else GpioServo
        assert servo_cls is not None

        self.valve_pin = valve_pin
        self.pump_pin = pump_pin
        self.keep_pump_during_hold = keep_pump_during_hold
        self.dry_run = dry_run
        self.verbose = verbose
        self.state = AirPumpState()

        servo_kwargs = {
            "min_pulse_width": min_pulse_us / 1_000_000,
            "max_pulse_width": max_pulse_us / 1_000_000,
            "frame_width": frame_ms / 1000,
            "initial_value": None,
        }
        self.valve = servo_cls(valve_pin, **servo_kwargs)
        self.pump = servo_cls(pump_pin, **servo_kwargs)

        self.off()

    @staticmethod
    def _angle_to_value(angle: float) -> float:
        angle = max(0.0, min(180.0, float(angle)))
        return angle / 90.0 - 1.0

    def _write_angle(self, servo: object, angle: float) -> None:
        value = self._angle_to_value(angle)
        setattr(servo, "value", value)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[air-pump] {message} | {self.state.as_text()}")

    # -------------------------
    # Direct keyboard-test layer
    # -------------------------

    def valve_close(self) -> None:
        self._write_angle(self.valve, 0)
        self.state.valve_open = False
        self._log("电磁阀关闭")

    def valve_open(self) -> None:
        self._write_angle(self.valve, 180)
        self.state.valve_open = True
        self._log("电磁阀打开")

    def pump_off(self) -> None:
        self._write_angle(self.pump, 0)
        self.state.pump_on = False
        self._log("气泵关闭")

    def pump_on(self) -> None:
        self._write_angle(self.pump, 180)
        self.state.pump_on = True
        self._log("气泵开启")

    def toggle_valve(self) -> None:
        if self.state.valve_open:
            self.valve_close()
        else:
            self.valve_open()

    def toggle_pump(self) -> None:
        if self.state.pump_on:
            self.pump_off()
        else:
            self.pump_on()

    def off(self) -> None:
        """Safe idle: pump off and valve closed."""
        self._write_angle(self.pump, 0)
        self._write_angle(self.valve, 0)
        self.state = AirPumpState(valve_open=False, pump_on=False)
        self._log("全部关闭")

    # -------------------------
    # Gomoku-stone action layer
    # -------------------------

    def suction_state(self) -> None:
        """Build vacuum: valve closed, pump on."""
        self._log("进入吸力状态：电磁阀关闭，气泵开启")
        self.valve_close()
        self.pump_on()

    def release_state(self) -> None:
        """Release vacuum: pump off, valve open."""
        self._log("进入放气状态：气泵关闭，电磁阀开启")
        self.pump_off()
        self.valve_open()

    def pick_stone(self, pick_seconds: float = DEFAULT_PICK_SECONDS) -> None:
        """吸棋子：先建立负压，再进入保持吸住状态。"""
        self._log(f"吸棋子：建立负压 {pick_seconds:.2f} 秒")
        self.suction_state()
        sleep(pick_seconds)
        self.hold_stone()

    def hold_stone(self, keep_pump: bool | None = None) -> None:
        """保持吸住：默认继续开泵，除非确认硬件可以闭阀保压。"""
        if keep_pump is None:
            keep_pump = self.keep_pump_during_hold

        if keep_pump:
            self._log("保持吸住：电磁阀关闭，气泵保持开启")
            self.valve_close()
            self.pump_on()
        else:
            self._log("保持吸住：电磁阀关闭，气泵关闭，依靠负压保持")
            self.valve_close()
            self.pump_off()

    def drop_stone(self, release_seconds: float = DEFAULT_RELEASE_SECONDS) -> None:
        """放棋子：先关泵，再打开电磁阀泄压，最后回到安全关闭。"""
        self._log(f"放棋子：泄压 {release_seconds:.2f} 秒")
        self.release_state()
        sleep(release_seconds)
        self.off()

    def pick_hold_drop(
        self,
        *,
        pick_seconds: float = DEFAULT_PICK_SECONDS,
        hold_seconds: float = DEFAULT_HOLD_SECONDS,
        release_seconds: float = DEFAULT_RELEASE_SECONDS,
    ) -> None:
        """Complete suction-cup action test: 吸棋子 -> 保持吸住 -> 放棋子."""
        self.pick_stone(pick_seconds=pick_seconds)
        self._log(f"保持吸住 {hold_seconds:.2f} 秒")
        sleep(hold_seconds)
        self.drop_stone(release_seconds=release_seconds)

    def close(self) -> None:
        try:
            self.off()
            sleep(0.2)
        finally:
            self.valve.close()
            self.pump.close()

    def __enter__(self) -> AirPumpRig:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class RawKeyboard:
    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self.old_settings: list[object] | None = None

    def __enter__(self) -> RawKeyboard:
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def read_key(self, timeout: float = 0.1) -> str | None:
        readable, _, _ = select.select([sys.stdin], [], [], timeout)
        if not readable:
            return None
        return sys.stdin.read(1)


def print_keyboard_help(timing: AirPumpTiming, keep_pump: bool) -> None:
    keep_text = "保持开泵" if keep_pump else "闭阀保压"
    print(
        "\n单键测试模式（不需要回车）\n"
        "  p        切换气泵开/关\n"
        "  v        切换电磁阀开/关\n"
        "  1 / 2    气泵开 / 气泵关\n"
        "  3 / 4    电磁阀开 / 电磁阀关\n"
        "  s        吸棋子：关阀 + 开泵，建立负压后进入保持\n"
        "  h        保持吸住：关阀，按当前保持策略控制气泵\n"
        "  d        放棋子：关泵 + 开阀泄压，然后全部关闭\n"
        "  t        完整流程：吸棋子 -> 保持吸住 -> 放棋子\n"
        "  k        切换保持策略：保持开泵 / 闭阀保压\n"
        "  + / -    调整吸棋子时长，每次 0.1 秒\n"
        "  ] / [    调整放棋子泄压时长，每次 0.1 秒\n"
        "  0/空格   全部关闭\n"
        "  ?        显示帮助\n"
        "  q        退出并关闭\n"
        f"\n当前参数：吸棋子 {timing.pick_seconds:.2f}s，"
        f"保持 {timing.hold_seconds:.2f}s，"
        f"放棋子 {timing.release_seconds:.2f}s，"
        f"保持策略={keep_text}\n"
    )


def run_keyboard_test(air: AirPumpRig, timing: AirPumpTiming) -> None:
    if not sys.stdin.isatty():
        raise SystemExit("Keyboard mode needs an interactive terminal.")

    print_keyboard_help(timing, air.keep_pump_during_hold)

    with RawKeyboard() as keyboard:
        while True:
            key = keyboard.read_key()
            if key is None:
                continue

            if key == "\x03":
                raise KeyboardInterrupt
            if key in {"q", "Q"}:
                break
            if key == "?":
                print_keyboard_help(timing, air.keep_pump_during_hold)
            elif key in {"0", " "}:
                air.off()
            elif key == "p":
                air.toggle_pump()
            elif key == "v":
                air.toggle_valve()
            elif key == "1":
                air.pump_on()
            elif key == "2":
                air.pump_off()
            elif key == "3":
                air.valve_open()
            elif key == "4":
                air.valve_close()
            elif key == "s":
                air.pick_stone(pick_seconds=timing.pick_seconds)
            elif key == "h":
                air.hold_stone()
            elif key == "d":
                air.drop_stone(release_seconds=timing.release_seconds)
            elif key == "t":
                air.pick_hold_drop(
                    pick_seconds=timing.pick_seconds,
                    hold_seconds=timing.hold_seconds,
                    release_seconds=timing.release_seconds,
                )
            elif key == "k":
                air.keep_pump_during_hold = not air.keep_pump_during_hold
                mode = "保持开泵" if air.keep_pump_during_hold else "闭阀保压"
                print(f"\n保持策略切换为：{mode}")
            elif key in {"+", "="}:
                timing.pick_seconds = round(timing.pick_seconds + 0.1, 2)
                print(f"\n吸棋子时长：{timing.pick_seconds:.2f}s")
            elif key in {"-", "_"}:
                timing.pick_seconds = max(0.1, round(timing.pick_seconds - 0.1, 2))
                print(f"\n吸棋子时长：{timing.pick_seconds:.2f}s")
            elif key == "]":
                timing.release_seconds = round(timing.release_seconds + 0.1, 2)
                print(f"\n放棋子泄压时长：{timing.release_seconds:.2f}s")
            elif key == "[":
                timing.release_seconds = max(0.1, round(timing.release_seconds - 0.1, 2))
                print(f"\n放棋子泄压时长：{timing.release_seconds:.2f}s")
            else:
                print(f"\n未知按键：{repr(key)}，按 ? 查看帮助")


def wait_forever() -> None:
    while True:
        sleep(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Keyboard and action-level test for Gomoku air pump suction control."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="keyboard",
        choices=["keyboard", "off", "pick", "hold", "drop", "cycle"],
        help="Test command. Default: keyboard.",
    )
    parser.add_argument("--valve-pin", type=int, default=DEFAULT_VALVE_GPIO)
    parser.add_argument("--pump-pin", type=int, default=DEFAULT_PUMP_GPIO)
    parser.add_argument("--min-pulse-us", type=int, default=DEFAULT_MIN_PULSE_US)
    parser.add_argument("--max-pulse-us", type=int, default=DEFAULT_MAX_PULSE_US)
    parser.add_argument("--frame-ms", type=int, default=DEFAULT_FRAME_MS)
    parser.add_argument("--pick-time", type=float, default=DEFAULT_PICK_SECONDS)
    parser.add_argument("--hold-time", type=float, default=None)
    parser.add_argument("--release-time", type=float, default=DEFAULT_RELEASE_SECONDS)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument(
        "--no-keep-pump",
        action="store_true",
        help="During hold, close valve and turn pump off. Use only after hardware validation.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print state changes without GPIO.")
    parser.add_argument("--quiet", action="store_true", help="Reduce status output.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    timing = AirPumpTiming(
        pick_seconds=args.pick_time,
        hold_seconds=DEFAULT_HOLD_SECONDS if args.hold_time is None else args.hold_time,
        release_seconds=args.release_time,
        interval_seconds=args.interval,
    )

    def handle_signal(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        with AirPumpRig(
            valve_pin=args.valve_pin,
            pump_pin=args.pump_pin,
            min_pulse_us=args.min_pulse_us,
            max_pulse_us=args.max_pulse_us,
            frame_ms=args.frame_ms,
            keep_pump_during_hold=not args.no_keep_pump,
            dry_run=args.dry_run,
            verbose=not args.quiet,
        ) as air:
            if args.command == "keyboard":
                run_keyboard_test(air, timing)
            elif args.command == "off":
                air.off()
            elif args.command == "pick":
                air.pick_stone(pick_seconds=timing.pick_seconds)
                if args.hold_time is None:
                    print("已进入保持吸住状态。按 Ctrl+C 关闭。")
                    wait_forever()
                else:
                    sleep(timing.hold_seconds)
            elif args.command == "hold":
                air.hold_stone()
                if args.hold_time is None:
                    print("保持吸住中。按 Ctrl+C 关闭。")
                    wait_forever()
                else:
                    sleep(timing.hold_seconds)
            elif args.command == "drop":
                air.drop_stone(release_seconds=timing.release_seconds)
            elif args.command == "cycle":
                for index in range(max(1, args.count)):
                    print(f"\nCycle {index + 1}/{max(1, args.count)}")
                    air.pick_hold_drop(
                        pick_seconds=timing.pick_seconds,
                        hold_seconds=timing.hold_seconds,
                        release_seconds=timing.release_seconds,
                    )
                    if index != max(1, args.count) - 1:
                        sleep(timing.interval_seconds)
    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭电磁阀和气泵。")
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
