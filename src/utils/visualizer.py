"""Visualization utilities for debugging."""

import cv2
import numpy as np
from src.utils.constants import BLACK, EMPTY, WHITE


def draw_board_overlay(
    image: np.ndarray,
    board_state: np.ndarray,
    grid_size: int | None = None,
) -> np.ndarray:
    """在棋盘图像上叠加棋子状态标记。

    Args:
        image: 透视矫正后的棋盘俯视图像 (BGR)。
        board_state: 15×15 状态矩阵，0=空 1=黑 2=白。
        grid_size: 每个格子像素数，若为 None 则根据 image 尺寸推断。

    Returns:
        叠加了标记的图像。
    """
    out = image.copy()
    rows, cols = board_state.shape
    h, w = out.shape[:2]
    if grid_size is None:
        grid_size = max(w // cols, h // rows)

    for r in range(rows):
        for c in range(cols):
            cx = int(c * grid_size + grid_size / 2)
            cy = int(r * grid_size + grid_size / 2)
            if board_state[r, c] == BLACK:
                cv2.circle(out, (cx, cy), grid_size // 4, (0, 0, 255), 2)
            elif board_state[r, c] == WHITE:
                cv2.circle(out, (cx, cy), grid_size // 4, (255, 0, 0), 2)
            else:
                cv2.circle(out, (cx, cy), 3, (0, 255, 0), 1)

    return out


def draw_board_graphics(
    board_state: np.ndarray,
    cell_size: int = 60,
    last_move: tuple[int, int] | None = None,
) -> np.ndarray:
    """绘制干净的棋盘图形（用于调试展示）。

    Args:
        board_state: 15×15 状态矩阵。
        cell_size: 每个格子像素大小。
        last_move: 最近一步落子 (row, col)，会高亮显示。

    Returns:
        BGR 图像。
    """
    rows, cols = board_state.shape
    margin = cell_size // 2
    img_h = rows * cell_size + 2 * margin
    img_w = cols * cell_size + 2 * margin

    # 棋盘底色
    canvas = np.full((img_h, img_w, 3), (60, 120, 180), dtype=np.uint8)

    # 格子
    for r in range(rows):
        for c in range(cols):
            x1 = margin + c * cell_size
            y1 = margin + r * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 80, 140), 1)

    # 棋子
    for r in range(rows):
        for c in range(cols):
            cx = margin + c * cell_size + cell_size // 2
            cy = margin + r * cell_size + cell_size // 2
            radius = int(cell_size * 0.38)
            if board_state[r, c] == BLACK:
                cv2.circle(canvas, (cx, cy), radius, (30, 30, 30), -1)
            elif board_state[r, c] == WHITE:
                cv2.circle(canvas, (cx, cy), radius, (240, 240, 240), -1)

    # 高亮最后落子
    if last_move is not None:
        r, c = last_move
        cx = margin + c * cell_size + cell_size // 2
        cy = margin + r * cell_size + cell_size // 2
        cv2.circle(canvas, (cx, cy), int(cell_size * 0.42), (0, 255, 255), 3)

    return canvas
