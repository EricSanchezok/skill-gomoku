"""棋盘状态与游戏规则."""

import numpy as np

from src.utils.constants import BLACK, BOARD_COLS, BOARD_ROWS, EMPTY


class Board:
    """15×15 五子棋棋盘状态。

    内部使用 numpy int8 矩阵，0=空, 1=黑, 2=白。
    """

    def __init__(self, state: np.ndarray | None = None):
        if state is None:
            self._state = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        else:
            if state.shape != (BOARD_ROWS, BOARD_COLS):
                raise ValueError(
                    f"Board state must be {BOARD_ROWS}×{BOARD_COLS}, got {state.shape}"
                )
            self._state = state.astype(np.int8)

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def rows(self) -> int:
        return BOARD_ROWS

    @property
    def cols(self) -> int:
        return BOARD_COLS

    def get(self, row: int, col: int) -> int:
        return int(self._state[row, col])

    def place(self, row: int, col: int, stone: int) -> None:
        if not (0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS):
            raise IndexError(f"Position ({row}, {col}) out of bounds")
        if self._state[row, col] != EMPTY:
            raise ValueError(f"Position ({row}, {col}) already occupied")
        self._state[row, col] = stone

    def get_legal_moves(self) -> list[tuple[int, int]]:
        return [
            (r, c)
            for r in range(BOARD_ROWS)
            for c in range(BOARD_COLS)
            if self._state[r, c] == EMPTY
        ]

    def is_full(self) -> bool:
        return not np.any(self._state == EMPTY)

    def __repr__(self) -> str:
        rows = ["    " + " ".join(f"{c:2d}" for c in range(BOARD_COLS))]
        for r in range(BOARD_ROWS):
            line = " ".join(f"{_stone_symbol(self._state[r, c]):>2}" for c in range(BOARD_COLS))
            rows.append(f"{r:2d}  {line}")
        return "\n".join(rows)


def check_win(board: np.ndarray, last_move: tuple[int, int] | None = None) -> int:
    """检查是否有五连。

    Args:
        board: BOARD_ROWS×BOARD_COLS 状态矩阵。
        last_move: 最后落子位置 (row, col)，若提供则仅检查经过该点的线。

    Returns:
        0 表示无人获胜，1=黑胜，2=白胜。
    """
    rows, cols = board.shape
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    def count_line(r: int, c: int, dr: int, dc: int) -> int:
        stone = board[r, c]
        if stone == EMPTY:
            return 0
        cnt = 1
        for sign in (1, -1):
            nr, nc = r, c
            for _ in range(4):
                nr += sign * dr
                nc += sign * dc
                if 0 <= nr < rows and 0 <= nc < cols and board[nr, nc] == stone:
                    cnt += 1
                else:
                    break
        return cnt

    if last_move is not None:
        r, c = last_move
        for dr, dc in directions:
            if count_line(r, c, dr, dc) >= 5:
                return int(board[r, c])
        return EMPTY

    for r in range(rows):
        for c in range(cols):
            if board[r, c] == EMPTY:
                continue
            for dr, dc in directions:
                if count_line(r, c, dr, dc) >= 5:
                    return int(board[r, c])
    return EMPTY


def _stone_symbol(stone: int) -> str:
    if stone == EMPTY:
        return "."
    if stone == BLACK:
        return "B"
    return "W"
