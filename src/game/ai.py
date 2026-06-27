"""AI 决策模块 — Rapfi 引擎封装。

通过 Gomocup 协议与 Rapfi 子进程通信。
Rapfi 是 Gomocup 2024 冠军引擎，C++ 实现，ARM64 NEON 原生支持。
"""

from __future__ import annotations

import logging
import platform
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.game.play_area import parse_play_area_config
from src.utils.constants import BLACK, BOARD_ROWS, EMPTY

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAPFI_BIN_DIR = PROJECT_ROOT / "bin" / "rapfi"
DEFAULT_TIME_PER_MOVE_MS = 3000

_PLATFORM_RAPFI_DIRS = {
    ("darwin", "arm64"): "macos-arm64",
    ("linux", "aarch64"): "linux-aarch64",
    ("linux", "arm64"): "linux-aarch64",
}

# Move response pattern: two integers separated by comma
_MOVE_RE = re.compile(r"^(-?\d+),(-?\d+)$")


@dataclass(frozen=True)
class AIEngineSettings:
    """Runtime settings for the Rapfi engine singleton."""

    engine_path: Path
    time_per_move_ms: int = DEFAULT_TIME_PER_MOVE_MS
    board_size: int = BOARD_ROWS


_settings: AIEngineSettings | None = None


class RapfiEngine:
    """Rapfi 引擎子进程封装。

    通过 Gomocup 协议管理一局对弈的完整生命周期：
    ``start_game() → play_move() / set_board() → stop()``。
    """

    def __init__(
        self,
        engine_path: str | Path,
        time_per_move_ms: int = 3000,
        board_size: int = 15,
    ) -> None:
        """启动 Rapfi 子进程。

        Args:
            engine_path: ``rapfi`` 可执行文件路径。
            time_per_move_ms: 每步思考时间（毫秒）。
            board_size: 引擎看到的棋盘大小，通常 15；机器人只在可靠窗口下棋时可为 9。
        """
        self._path = str(engine_path)
        self._time_ms = time_per_move_ms
        self._size = board_size
        self._proc: subprocess.Popen[str] | None = None
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_game(self, is_black: bool = True) -> tuple[int, int] | None:
        """启动新一局，告知引擎我方颜色。

        Returns:
            如果引擎先行（is_black=True）且通过 BEGIN 命令获得首步落子，
            返回 (row, col)；否则返回 None。
        """
        if self._proc is not None:
            self.stop()

        self._proc = subprocess.Popen(
            [self._path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self._started = False

        # Init
        self._send(f"START {self._size}")
        self._send(f"INFO timeout_turn {self._time_ms}")
        self._send(f"INFO timeout_match {self._time_ms * 100}")
        self._send("INFO rule 0")  # freestyle (no swap)
        self._started = True

        # If engine plays black, let it make the opening move
        if is_black:
            self._send("BEGIN")
            move = self._read_move()
            if move is not None:
                return self._coord_to_rowcol(move)
        return None

    def set_board(self, board: np.ndarray, my_stone: int) -> tuple[int, int]:
        """将当前棋盘状态发送给引擎，等待引擎回复落子。

        Args:
            board: NxN int8 矩阵，EMPTY=0 / BLACK=1 / WHITE=2。
            my_stone: 我方棋子颜色（BLACK 或 WHITE）。

        Returns:
            (row, col) 引擎选择的落子位置。
        """
        if self._proc is None:
            raise RuntimeError("引擎未启动，请先调用 start_game()")

        field_own = "1"
        field_opp = "2"

        if board.ndim != 2 or board.shape[0] != board.shape[1]:
            raise ValueError(f"Rapfi board must be square, got shape {board.shape}")
        if board.shape[0] != self._size:
            raise ValueError(
                f"Rapfi engine was started for {self._size}x{self._size}, "
                f"but got board shape {board.shape}"
            )

        rows, cols = board.shape
        stones: list[str] = []
        for r in range(rows):
            for c in range(cols):
                if board[r, c] == my_stone:
                    stones.append(f"{c},{r},{field_own}")
                elif board[r, c] != EMPTY:
                    stones.append(f"{c},{r},{field_opp}")

        self._send("BOARD")
        for s in stones:
            self._send(s)
        self._send("DONE")

        move = self._read_move()
        if move is None:
            raise RuntimeError("引擎未返回有效落子")
        return self._coord_to_rowcol(move)

    def stop(self) -> None:
        """优雅停止引擎子进程。"""
        if self._proc is None:
            return
        try:
            self._send("END")
            self._proc.wait(timeout=3)
        except Exception:
            self._proc.kill()
        finally:
            self._proc = None
            self._started = False

    # ------------------------------------------------------------------
    # Internal — protocol I/O
    # ------------------------------------------------------------------

    def _send(self, line: str) -> None:
        """向引擎发送一行命令（自动附加 \\r\\n）。"""
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(line + "\r\n")
        self._proc.stdin.flush()

    def _read_line(self) -> str | None:
        """从引擎读取一行，跳过空行。"""
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            line = self._proc.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if line and line.upper() != "OK":
                return line

    def _read_move(self) -> str | None:
        """读取引擎响应，跳过 MESSAGE/DEBUG/ERROR 等前缀行，提取 x,y 落子坐标。"""
        while True:
            line = self._read_line()
            if line is None:
                return None
            if line.upper().startswith("MESSAGE") or line.upper().startswith("DEBUG"):
                logger.debug("Rapfi: %s", line)
                continue
            if line.upper().startswith("ERROR"):
                logger.error("Rapfi error: %s", line)
                return None
            m = _MOVE_RE.match(line)
            if m:
                return line
            # Unknown line — log and continue
            logger.debug("Rapfi unknown: %s", line)

    @staticmethod
    def _coord_to_rowcol(coord: str) -> tuple[int, int]:
        """将 Rapfi 的 x,y 坐标转为 (row, col)。"""
        x_str, y_str = coord.split(",")
        return int(y_str), int(x_str)


# ------------------------------------------------------------------
# Global engine singleton
# ------------------------------------------------------------------

_engine: RapfiEngine | None = None


def _get_engine(time_per_move_ms: int | None = None) -> RapfiEngine:
    global _engine
    if _engine is None:
        settings = _active_settings()
        active_time_ms = settings.time_per_move_ms if time_per_move_ms is None else time_per_move_ms
        _engine = RapfiEngine(
            engine_path=settings.engine_path,
            time_per_move_ms=active_time_ms,
            board_size=settings.board_size,
        )
    return _engine


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def ai_decide(board: np.ndarray, my_stone: int) -> tuple[int, int]:
    """给定棋局状态，返回 AI 的最佳落子位置。

    Args:
        board: NxN int8 矩阵 (EMPTY=0 / BLACK=1 / WHITE=2)。
        my_stone: 我方棋子颜色。

    Returns:
        (row, col) 落子位置。
    """
    engine = _get_engine()
    try:
        return engine.set_board(board, my_stone)
    except RuntimeError:
        # Engine may have died; restart
        global _engine
        _engine = None
        engine = _get_engine()
        is_empty_board = not bool(np.any(board != EMPTY))
        opening_move = engine.start_game(is_black=(my_stone == BLACK and is_empty_board))
        if opening_move is not None:
            return opening_move
        return engine.set_board(board, my_stone)


def configure_ai(
    *,
    engine_path: str | Path | None = None,
    time_per_move_ms: int = DEFAULT_TIME_PER_MOVE_MS,
    board_size: int = BOARD_ROWS,
) -> AIEngineSettings:
    """Configure the process-wide Rapfi singleton.

    The existing engine process is stopped whenever settings change so the next
    call to :func:`ai_decide` starts Rapfi with the requested binary.
    """

    global _settings
    resolved_engine_path = (
        resolve_rapfi_engine_path() if engine_path is None else Path(engine_path).expanduser()
    )
    new_settings = AIEngineSettings(
        engine_path=resolved_engine_path,
        time_per_move_ms=int(time_per_move_ms),
        board_size=int(board_size),
    )
    if new_settings != _settings:
        ai_reset()
        _settings = new_settings
    return _settings


def configure_ai_from_config(
    config: Mapping[str, Any],
    base_dir: str | Path = PROJECT_ROOT,
) -> AIEngineSettings:
    """Configure Rapfi from ``game.ai`` settings in the YAML config."""

    game_cfg = config.get("game", {})
    if not isinstance(game_cfg, Mapping):
        game_cfg = {}
    ai_cfg = game_cfg.get("ai", {})
    if not isinstance(ai_cfg, Mapping):
        ai_cfg = {}

    path_value = ai_cfg.get("engine_path", ai_cfg.get("rapfi_path"))
    engine_path = (
        resolve_rapfi_engine_path()
        if path_value in (None, "")
        else _resolve_config_path(path_value, base_dir=base_dir)
    )
    time_per_move_ms = int(ai_cfg.get("time_per_move_ms", DEFAULT_TIME_PER_MOVE_MS))
    board_size_value = ai_cfg.get("board_size")
    if board_size_value in (None, ""):
        play_area = parse_play_area_config(game_cfg.get("play_area", game_cfg.get("active_area")))
        if play_area.rows != play_area.cols:
            raise ValueError(
                "Rapfi requires a square game.play_area when game.ai.board_size is unset"
            )
        board_size = play_area.rows
    else:
        board_size = int(board_size_value)
    return configure_ai(
        engine_path=engine_path,
        time_per_move_ms=time_per_move_ms,
        board_size=board_size,
    )


def resolve_rapfi_engine_path(
    *,
    system: str | None = None,
    machine: str | None = None,
) -> Path:
    """Return the default Rapfi binary path for the current platform."""

    platform_key = _rapfi_platform_key(system=system, machine=machine)
    return RAPFI_BIN_DIR / platform_key / "rapfi"


def ai_reset() -> None:
    """重置 AI 状态（新一局开始时调用）。"""
    global _engine
    if _engine is not None:
        _engine.stop()
        _engine = None


def _rapfi_platform_key(
    *,
    system: str | None = None,
    machine: str | None = None,
) -> str:
    system_name = (system or platform.system()).lower()
    machine_name = (machine or platform.machine()).lower()
    try:
        return _PLATFORM_RAPFI_DIRS[(system_name, machine_name)]
    except KeyError as exc:
        raise RuntimeError(
            f"No bundled Rapfi binary for platform {system_name}/{machine_name}. "
            "Set game.ai.engine_path to an explicit Rapfi executable."
        ) from exc


def _resolve_config_path(value: Any, *, base_dir: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def _active_settings() -> AIEngineSettings:
    global _settings
    if _settings is None:
        _settings = AIEngineSettings(engine_path=resolve_rapfi_engine_path())
    return _settings
