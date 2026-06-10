"""棋子检测器 — 将俯视矫正棋盘图像的每个格子分类为空、黑子或白子.

算法:
  1. 对每个格子提取圆形 ROI（半径 = roi_ratio × 格子尺寸），排除格线干扰.
  2. 将 ROI 转换为 HSV 颜色空间.
  3. 基于饱和度与明度分类:
     - 低饱和度 → EMPTY（木质棋盘，无棋子）
     - 低明度 → BLACK 黑子
     - 高明度 → WHITE 白子
     - 其他情况 → EMPTY（模糊区域，默认判空）

所有参数优先从 config 的 ``stone_detection`` 段读取，否则回退到类级别默认值.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.utils.constants import BLACK, BOARD_COLS, BOARD_ROWS, EMPTY, WHITE

logger = logging.getLogger(__name__)


class StoneDetector:
    """将俯视矫正棋盘的每个格子分类为空、黑子或白子.

    参数从 *config* 的 ``stone_detection`` 段读取，未指定时回退到类常量.

    Attributes:
        _roi_ratio: 圆形 ROI 半径占格子边长的比例.
        _black_v_max: V 通道阈值，低于此值判为黑子.
        _white_v_min: V 通道阈值，高于此值判为白子.
        _empty_s_max: 平均饱和度阈值，低于此值判为空.
    """

    # ---- 默认值 ----------------------------------------------------------------
    _ROI_RATIO: float = 0.35
    _BLACK_V_MAX: int = 80
    _WHITE_V_MIN: int = 150
    _EMPTY_S_MAX: int = 30

    # ---------------------------------------------------------------------------

    def __init__(self, config: dict | None = None) -> None:
        """初始化检测器并配置阈值.

        Args:
            config: 完整配置字典，其 ``stone_detection`` 键值用于覆盖默认参数.
        """
        cfg: dict = (config or {}).get("stone_detection", {})

        self._roi_ratio: float = float(cfg.get("roi_ratio", self._ROI_RATIO))
        self._black_v_max: int = int(cfg.get("black_v_max", self._BLACK_V_MAX))
        self._white_v_min: int = int(cfg.get("white_v_min", self._WHITE_V_MIN))
        self._empty_s_max: int = int(cfg.get("empty_s_max", self._EMPTY_S_MAX))

        logger.debug(
            "StoneDetector 初始化 — roi_ratio=%.2f black_v_max=%d white_v_min=%d empty_s_max=%d",
            self._roi_ratio,
            self._black_v_max,
            self._white_v_min,
            self._empty_s_max,
        )

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def detect(
        self,
        warped_board: np.ndarray,
        cells: list[list[tuple[int, int, int, int]]],
    ) -> np.ndarray:
        """对俯视棋盘图像上的每个格子进行分类.

        Args:
            warped_board: 俯视矫正后的棋盘 BGR 图像，shape ``(H, W, 3)``.
            cells: 15×15 网格，每个元素为 ``(x, y, w, h)`` 格子边界，
                由 :meth:`BoardDetector.get_grid_cells` 返回.

        Returns:
            ``(15, 15)`` int8 矩阵:
                - 0 = :data:`~src.utils.constants.EMPTY`
                - 1 = :data:`~src.utils.constants.BLACK`
                - 2 = :data:`~src.utils.constants.WHITE`

            当 *cells* 为空或尺寸不匹配时返回全空棋盘.
        """
        board = np.full((BOARD_ROWS, BOARD_COLS), EMPTY, dtype=np.int8)

        if not cells or len(cells) != BOARD_ROWS:
            logger.warning("cells 为空或行数不等于 %d，返回全空棋盘", BOARD_ROWS)
            return board

        for r in range(BOARD_ROWS):
            row_cells = cells[r]
            if len(row_cells) != BOARD_COLS:
                logger.warning("cells[%d] 列数不等于 %d，返回全空棋盘", r, BOARD_COLS)
                return np.full((BOARD_ROWS, BOARD_COLS), EMPTY, dtype=np.int8)

            for c in range(BOARD_COLS):
                x, y, w, h = row_cells[c]
                cell_img = warped_board[y : y + h, x : x + w]
                board[r, c] = self.detect_cell(cell_img)

        black_count = int(np.count_nonzero(board == BLACK))
        white_count = int(np.count_nonzero(board == WHITE))
        logger.info("棋子检测完成 — 黑子 %d 白子 %d", black_count, white_count)
        return board

    def detect_cell(self, cell_img: np.ndarray) -> int:
        """对单个格子图像进行分类.

        Args:
            cell_img: 单个格子的 BGR 图像，shape ``(h, w, 3)``.

        Returns:
            ``0`` (EMPTY)、``1`` (BLACK) 或 ``2`` (WHITE).
        """
        h, w = cell_img.shape[:2]
        if h < 3 or w < 3:
            return EMPTY

        # 提取圆心区域的 HSV 特征
        roi = self._extract_circular_roi(cell_img)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = self._circular_mask(roi.shape[:2])
        valid = mask > 0
        if np.count_nonzero(valid) == 0:
            return EMPTY

        mean_s = np.mean(hsv[valid, 1])
        mean_v = np.mean(hsv[valid, 2])

        # 分类优先级:
        #   1. 低饱和度 → 木色棋盘底色 → 空
        #   2. 低明度 → 黑子
        #   3. 高明度 → 白子
        #   4. 其余 → 空（模糊区域，保守处理）
        if mean_s < self._empty_s_max:
            return EMPTY
        if mean_v < self._black_v_max:
            return BLACK
        if mean_v > self._white_v_min:
            return WHITE
        return EMPTY

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _extract_circular_roi(self, cell_img: np.ndarray) -> np.ndarray:
        """提取格子中心的圆形区域，圆外像素置零以排除格线."""
        h, w = cell_img.shape[:2]
        mask = self._circular_mask((h, w))
        return cv2.bitwise_and(cell_img, cell_img, mask=mask)

    def _circular_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """为给定形状生成居中的圆形 uint8 掩码.

        圆心为图像中心，半径为 ``roi_ratio * min(h, w) / 2``.
        """
        h, w = shape
        center = (w // 2, h // 2)
        radius = int(self._roi_ratio * min(h, w) / 2)
        radius = max(radius, 1)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        return mask
