"""棋子检测器 — Hough 圆检测 + 局部对比度分类。

策略（不受棋盘颜色影响）:
    1. 在俯视矫正图上用 HoughCircles 检测所有近圆形轮廓（棋子）。
    2. 对每个候选圆，取内圈 vs 背景环的灰度对比，区分黑白：
       - 白子：内圈亮度显著高于局部背景
       - 黑子：内圈亮度低于或接近局部背景
    3. 把每个圆分配到最近的 15×15 落子交叉点。

该方法只依赖形状和局部对比度，对光照变化和棋盘底色鲁棒。
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.utils.constants import BLACK, BOARD_COLS, BOARD_ROWS, WHITE

logger = logging.getLogger(__name__)


class StoneDetector:
    """Hough 圆检测 + 局部对比度的棋子检测器。"""

    # HoughCircles 参数
    _DP: float = 1.0
    _MIN_DIST: int = 25
    _PARAM1: float = 40.0
    _PARAM2: float = 15.0
    _MIN_RADIUS: int = 10
    _MAX_RADIUS: int = 22

    # 分类阈值
    _WHITE_INNER_MEAN: float = 75.0  # 内圈灰度高于此值 → 白子候选
    _WHITE_CONTRAST: float = 10.0  # 内圈 vs 背景环灰度差值 → 确认白子

    def __init__(self, config: dict | None = None) -> None:
        cfg: dict = (config or {}).get("stone_detection", {})

        self._dp: float = float(cfg.get("hough_dp", self._DP))
        self._min_dist: float = float(cfg.get("hough_min_dist", self._MIN_DIST))
        self._param1: float = float(cfg.get("hough_param1", self._PARAM1))
        self._param2: float = float(cfg.get("hough_param2", self._PARAM2))
        self._min_radius: int = int(cfg.get("hough_min_radius", self._MIN_RADIUS))
        self._max_radius: int = int(cfg.get("hough_max_radius", self._MAX_RADIUS))
        self._white_inner: float = float(cfg.get("white_inner_mean", self._WHITE_INNER_MEAN))
        self._white_contrast: float = float(cfg.get("white_contrast", self._WHITE_CONTRAST))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        warped_board: np.ndarray,
        cells: list[list[tuple[int, int, int, int]]],
    ) -> np.ndarray:
        """检测矫正棋盘上所有落子点的棋子。

        Args:
            warped_board: 俯视矫正 BGR 棋盘图 (H, W, 3)。
            cells: 15×15 交叉点命中区 (x, y, w, h)。

        Returns:
            (15, 15) int8 矩阵: 0=空 1=黑 2=白。
        """
        if not cells or len(cells) != BOARD_ROWS:
            logger.warning("网格单元不完整，返回空棋盘")
            return np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)

        gray = cv2.cvtColor(warped_board, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(
            cv2.medianBlur(gray, 5),
            cv2.HOUGH_GRADIENT,
            dp=self._dp,
            minDist=self._min_dist,
            param1=self._param1,
            param2=self._param2,
            minRadius=self._min_radius,
            maxRadius=self._max_radius,
        )

        if circles is None:
            logger.info("棋子检测完成 — 未找到任何圆")
            return np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)

        circles = circles[0]  # shape (N, 3)
        max_assign_dist = max(cells[0][0][2] * 1.5, 30) ** 2
        cell_centers = self._compute_cell_centers(cells)

        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)

        for cx, cy, r in circles:
            cx_i, cy_i, r_i = int(round(cx)), int(round(cy)), int(round(r))

            stone = self._classify_stone(gray, cx_i, cy_i, r_i)
            row, col = self._assign_to_cell(cx_i, cy_i, cell_centers, max_assign_dist)
            if row is not None:
                board[row, col] = stone

        black_count = int(np.count_nonzero(board == BLACK))
        white_count = int(np.count_nonzero(board == WHITE))
        logger.info("棋子检测完成 — 黑子 %d 白子 %d", black_count, white_count)
        return board

    def detect_cell(self, cell_img: np.ndarray) -> int:
        """对单个格子图像做分类。"""
        gray = cell_img if cell_img.ndim == 2 else cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        return self._classify_stone(gray, w // 2, h // 2, min(w, h) // 3)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_cell_centers(
        cells: list[list[tuple[int, int, int, int]]],
    ) -> list[list[tuple[float, float]]]:
        """从交叉点命中区构建交叉点中心坐标矩阵."""
        centers = []
        for r in range(BOARD_ROWS):
            row = []
            for c in range(BOARD_COLS):
                x, y, w, h = cells[r][c]
                row.append((x + w / 2, y + h / 2))
            centers.append(row)
        return centers

    def _classify_stone(self, gray: np.ndarray, cx: int, cy: int, radius: int) -> int:
        """通过内圈 vs 背景环的局部亮度对比来分类棋子。

        Returns:
            1=黑子, 2=白子。
        """
        # 内圈 mask
        inner_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(inner_mask, (cx, cy), max(radius - 3, 2), 255, -1)
        inner_mean = float(cv2.mean(gray, mask=inner_mask)[0])

        if inner_mean < self._white_inner:
            return BLACK

        # 背景环 mask（棋子外围一圈）
        bg_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(bg_mask, (cx, cy), radius + 8, 255, -1)
        cv2.circle(bg_mask, (cx, cy), radius + 2, 0, -1)
        bg_mean = float(cv2.mean(gray, mask=bg_mask)[0])

        return WHITE if inner_mean - bg_mean > self._white_contrast else BLACK

    @staticmethod
    def _assign_to_cell(
        cx: int,
        cy: int,
        cell_centers: list[list[tuple[float, float]]],
        max_dist_sq: float,
    ) -> tuple[int, int] | tuple[None, None]:
        """将圆心坐标分配到最近的落子交叉点."""
        best_row, best_col = None, None
        best_dist = max_dist_sq
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                ix, iy = cell_centers[r][c]
                d = (cx - ix) ** 2 + (cy - iy) ** 2
                if d < best_dist:
                    best_dist = d
                    best_row, best_col = r, c
        return best_row, best_col
