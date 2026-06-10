"""AI 决策模块 — Rapfi 引擎封装。

通过 Gomocup 协议与 Rapfi 子进程通信。
Rapfi 是 Gomocup 2024 冠军引擎，C++ 实现，ARM64 NEON 原生支持。
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

import numpy as np

from src.utils.constants import BLACK, BOARD_COLS, BOARD_ROWS, EMPTY, WHITE

logger = logging.getLogger(__name__)

# Default engine path relative to project root
_DEFAULT_ENGINE_PATH = Path(__file__).resolve().parent.parent.parent / "bin" / "rapfi"

# Move response pattern: two integers separated by comma
_MOVE_RE = re.compile(r"^(-?\d+),(-?\d+)$")


class RapfiEngine:
    """Rapfi 引擎子进程封装。

    通过 Gomocup 协议管理一局对弈的完整生命周期：
    ``start_game() → play_move() / set_board() → stop()``。
    """

    def __init__(
        self,
        engine_path: str | Path = str(_DEFAULT_ENGINE_PATH),
        time_per_move_ms: int = 3000,
        board_size: int = 15,
    ) -> None:
        """启动 Rapfi 子进程。

        Args:
            engine_path: ``rapfi`` 可执行文件路径。
            time_per_move_ms: 每步思考时间（毫秒）。
            board_size: 棋盘大小，通常 15。
        """
        self._path = str(engine_path)
        self._time_ms = time_per_move_ms
        self._size = board_size
        self._proc: subprocess.Popen[str] | None = None
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_game(self, is_black: bool = True) -> int | None:
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
            board: 15×15 int8 矩阵，EMPTY=0 / BLACK=1 / WHITE=2。
            my_stone: 我方棋子颜色（BLACK 或 WHITE）。

        Returns:
            (row, col) 引擎选择的落子位置。
        """
        if self._proc is None:
            raise RuntimeError("引擎未启动，请先调用 start_game()")

        field_own = "1"
        field_opp = "2"

        stones: list[str] = []
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
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


def _get_engine(time_per_move_ms: int = 3000) -> RapfiEngine:
    global _engine
    if _engine is None:
        _engine = RapfiEngine(time_per_move_ms=time_per_move_ms)
    return _engine


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def ai_decide(board: np.ndarray, my_stone: int) -> tuple[int, int]:
    """给定棋局状态，返回 AI 的最佳落子位置。

    Args:
        board: 15×15 int8 矩阵 (EMPTY=0 / BLACK=1 / WHITE=2)。
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
        engine.start_game(is_black=(my_stone == BLACK))
        return engine.set_board(board, my_stone)


def ai_reset() -> None:
    """重置 AI 状态（新一局开始时调用）。"""
    global _engine
    if _engine is not None:
        _engine.stop()
        _engine = None
