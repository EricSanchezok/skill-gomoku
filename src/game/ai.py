"""AI 策略模块 — 初版骨架."""

import random

import numpy as np

from src.game.board import Board, check_win
from src.utils.constants import BLACK, EMPTY, WHITE


def ai_decide(board: Board, my_stone: int) -> tuple[int, int]:
    """给定当前棋盘状态，返回 AI 决定落子的 (row, col)。

    Args:
        board: 当前棋盘。
        my_stone: 我方的棋子颜色 (1=黑, 2=白)。

    Returns:
        (row, col) 落子位置。
    """
    # 1. 先检查自己能否直接赢
    legal = board.get_legal_moves()
    for r, c in legal:
        board.place(r, c, my_stone)
        winner = check_win(board.state, (r, c))
        board._state[r, c] = EMPTY
        if winner == my_stone:
            return (r, c)

    # 2. 检查是否需要堵对手
    opponent = WHITE if my_stone == BLACK else BLACK
    for r, c in legal:
        board.place(r, c, opponent)
        winner = check_win(board.state, (r, c))
        board._state[r, c] = EMPTY
        if winner == opponent:
            return (r, c)

    # 3. 优先下中心附近
    center = board.rows // 2
    scored = []
    for r, c in legal:
        dist = abs(r - center) + abs(c - center)
        scored.append((dist, random.random(), r, c))
    scored.sort()

    return (scored[0][2], scored[0][3])
