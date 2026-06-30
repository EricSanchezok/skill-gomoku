"""Human-robot interaction ports used by the live Gomoku loop."""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from src.utils.constants import BLACK, WHITE

logger = logging.getLogger(__name__)


class HumanTurnCommand(str, Enum):
    """High-level command returned while waiting for a human move."""

    MOVE_DONE = "move_done"
    QUIT = "quit"


@dataclass(frozen=True)
class HumanTurnResult:
    """Result from the human-turn controller."""

    command: HumanTurnCommand = HumanTurnCommand.MOVE_DONE


class HumanTurnController(Protocol):
    """Port for confirming that the human has finished a move."""

    def wait_for_move_done(
        self,
        *,
        expected_stone: int,
        board_state: Any | None = None,
    ) -> HumanTurnResult:
        """Block until the human move is ready for perception."""


class RobotInteractionController(Protocol):
    """Port for optional human-facing robot actions."""

    def speak(self, text: str) -> None:
        """Make the robot say something."""

    def dance(self, name: str = "default") -> None:
        """Run a named robot dance/motion routine."""

    def use_skill_gomoku(self, context: Mapping[str, Any] | None = None) -> None:
        """Trigger the external skill-gomoku interaction hook."""


class NullRobotInteraction:
    """No-op interaction controller for tests and headless runs."""

    def speak(self, text: str) -> None:
        pass

    def dance(self, name: str = "default") -> None:
        pass

    def use_skill_gomoku(self, context: Mapping[str, Any] | None = None) -> None:
        pass


class ConsoleRobotInteraction:
    """Console-backed placeholder for future speech, dance, and skill hooks."""

    def __init__(
        self,
        print_fn: Callable[[str], None] = print,
        *,
        voice_enabled: bool = True,
    ) -> None:
        self._print = print_fn
        self._voice_enabled = voice_enabled

    def speak(self, text: str) -> None:
        self._print(f"[robot:speak] {text}")
        if self._voice_enabled:
            _speak_system_voice(text)

    def dance(self, name: str = "default") -> None:
        self._print(f"[robot:dance] {name}")

    def use_skill_gomoku(self, context: Mapping[str, Any] | None = None) -> None:
        detail = "" if context is None else f" {dict(context)}"
        self._print(f"[robot:skill_gomoku]{detail}")


@dataclass(frozen=True)
class KeyboardControlKeys:
    """Keyboard words reserved for live human-robot play."""

    confirm_keys: tuple[str, ...] = ("", "enter", "space", "done", "ok")
    quit_key: str = "q"
    speak_key: str = "s"
    dance_key: str = "d"
    skill_gomoku_key: str = "g"
    min_empty_enter_seconds: float = 0.25


class KeyboardHumanTurnController:
    """Line-input controller for human move confirmation.

    The default confirmation path is pressing Enter. Typing a single space and
    pressing Enter also works, which reserves the physical Space key for later
    raw-keyboard UIs.
    """

    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] = input,
        print_fn: Callable[[str], None] = print,
        robot_interaction: RobotInteractionController | None = None,
        keys: KeyboardControlKeys | None = None,
    ) -> None:
        self._input = input_fn
        self._print = print_fn
        self._robot_interaction = robot_interaction or NullRobotInteraction()
        self._keys = keys or KeyboardControlKeys()
        self._confirm_keys = {item.lower() for item in self._keys.confirm_keys}

    def wait_for_move_done(
        self,
        *,
        expected_stone: int,
        board_state: Any | None = None,
    ) -> HumanTurnResult:
        prompt = (
            f"人类{stone_name(expected_stone)}落子完成后按 Enter/Space；"
            f"{self._keys.speak_key}=说话 "
            f"{self._keys.dance_key}=跳舞 "
            f"{self._keys.skill_gomoku_key}=技能五子棋 "
            f"{self._keys.quit_key}=退出 > "
        )
        while True:
            started = time.monotonic()
            raw = self._input(prompt)
            elapsed = time.monotonic() - started
            command = self._normalize_key(raw)
            if (
                raw == ""
                and command in self._confirm_keys
                and elapsed < self._keys.min_empty_enter_seconds
            ):
                self._print("忽略过早的 Enter，防止上一轮输入残留。请下完棋后再确认。")
                continue
            if command in self._confirm_keys:
                return HumanTurnResult(HumanTurnCommand.MOVE_DONE)
            if command == self._keys.quit_key:
                return HumanTurnResult(HumanTurnCommand.QUIT)
            if command == self._keys.speak_key:
                self._robot_interaction.speak("我在看棋盘，准备继续。")
                continue
            if command == self._keys.dance_key:
                self._robot_interaction.dance("gomoku_waiting")
                continue
            if command == self._keys.skill_gomoku_key:
                self._robot_interaction.use_skill_gomoku(
                    {"expected_stone": stone_name(expected_stone), "board_state": board_state}
                )
                continue
            self._print("未识别的指令。按 Enter/Space 确认落子，或输入 s/d/g/q。")

    @staticmethod
    def _normalize_key(raw: str) -> str:
        if raw == " ":
            return "space"
        command = raw.strip().lower()
        if command == "":
            return ""
        return command


def stone_name(stone: int) -> str:
    """Return a human-readable Chinese name for a stone constant."""

    if stone == BLACK:
        return "黑棋"
    if stone == WHITE:
        return "白棋"
    return f"未知棋子({stone})"


def _speak_system_voice(text: str) -> None:
    command = _speech_command(text)
    if command is None:
        logger.warning("No system speech command found; skipped speech: %s", text)
        return
    try:
        subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as exc:
        logger.warning("Failed to run system speech command: %s", exc)


def _speech_command(text: str) -> list[str] | None:
    if not text.strip():
        return None
    candidates = (
        ("say", [text]),
        ("spd-say", ["--wait", text]),
        ("espeak-ng", [text]),
        ("espeak", [text]),
    )
    for executable, args in candidates:
        resolved = shutil.which(executable)
        if resolved is not None:
            return [resolved, *args]
    return None
