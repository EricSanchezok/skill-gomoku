"""棋盘检测器 — 纯图像处理 15×15 五子棋棋盘识别.

流水线:
  1. 预处理 — 灰度化 + 高斯模糊
  2. 定位棋盘 — HSV 木色掩码 + 最大轮廓 → 四边形四角
  3. 透视矫正 — 将棋盘变换为俯视正方形
  4. 格线检测 — Canny + 霍夫线检测 + 聚类 → 16 横 16 纵
  5. 网格划分 — 从线位置计算 15×15 网格单元格

所有参数优先从 config 读取，否则回退到 :mod:`src.utils.constants` 中的默认值。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.utils.constants import (
    BOARD_COLS,
    BOARD_ROWS,
    CALIB_CORNER_ORDER,
    DEFAULT_BOARD_COLOR_HIGH,
    DEFAULT_BOARD_COLOR_LOW,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 结果类型
# ---------------------------------------------------------------------------


@dataclass
class BoardDetectionResult:
    """棋盘检测结果.

    Attributes:
        warped: 俯视矫正后的棋盘图像 (BGR).
        transform: 3×3 透视变换矩阵.
        corners: 4×2 有序角点 (左上, 右上, 右下, 左下)，坐标位于原图空间.
        grid_cell_size: 矫正图中相邻交点的像素间距 (检测失败时为 0.0).
        success: 是否成功检测并矫正棋盘.
    """

    warped: np.ndarray = field(default_factory=lambda: np.zeros((600, 600, 3), dtype=np.uint8))
    transform: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    corners: np.ndarray = field(default_factory=lambda: np.zeros((4, 2), dtype=np.float32))
    grid_cell_size: float = 0.0
    success: bool = False


# ---------------------------------------------------------------------------
# BoardDetector
# ---------------------------------------------------------------------------


class BoardDetector:
    """纯图像处理检测相机画面中的 15×15 五子棋棋盘.

    不使用 ArUco 等人工标记，仅依赖图像特征:
      1. HSV 颜色空间提取木质棋盘区域
      2. 最大四边形轮廓 → 棋盘四角
      3. 透视变换得到俯视矫正图
      4. 在矫正图上用 Canny + 概率霍夫线检测提取格线
      5. 聚类得到等距的 16 横 16 纵线 → 计算 225 个格点区域

    参数由 *config* 的 ``board_detection`` 段提供，未指定时回退到类常量.
    """

    # ---- 可配置默认值 ---------------------------------------------------------
    _WARPED_SIZE: int = 600
    _BLUR_KERNEL: tuple[int, int] = (5, 5)
    _MORPH_KERNEL_SIZE: tuple[int, int] = (5, 5)
    _MORPH_ITERATIONS: int = 2
    _CANNY_LOW: float = 50.0
    _CANNY_HIGH: float = 150.0
    _HOUGH_THRESHOLD: int = 100
    _MIN_LINE_LENGTH: float = 200.0
    _MAX_LINE_GAP: float = 50.0
    _MIN_CONTOUR_AREA: float = 5000.0

    # --------------------------------------------------------------------

    def __init__(self, config: dict | None = None) -> None:
        """初始化棋盘检测器.

        Args:
            config: 完整配置字典，其 ``board_detection`` 键值用于覆盖默认参数.
        """
        cfg: dict = (config or {}).get("board_detection", {})

        # Canny / Hough 参数
        self._canny_low: float = float(cfg.get("canny_low", self._CANNY_LOW))
        self._canny_high: float = float(cfg.get("canny_high", self._CANNY_HIGH))
        self._hough_threshold: int = int(cfg.get("hough_threshold", self._HOUGH_THRESHOLD))
        self._min_line_length: float = float(cfg.get("min_line_length", self._MIN_LINE_LENGTH))
        self._max_line_gap: float = float(cfg.get("max_line_gap", self._MAX_LINE_GAP))

        # 木质棋盘 HSV 范围（config 可覆盖 constants 中的默认值）
        raw_low = cfg.get("board_color_low", DEFAULT_BOARD_COLOR_LOW.tolist())
        raw_high = cfg.get("board_color_high", DEFAULT_BOARD_COLOR_HIGH.tolist())
        self._board_color_low: np.ndarray = np.array(raw_low, dtype=np.uint8)
        self._board_color_high: np.ndarray = np.array(raw_high, dtype=np.uint8)

        # 最新检测结果
        self._last_result: BoardDetectionResult | None = None
        self._last_h_lines: list[int] = []
        self._last_v_lines: list[int] = []
        self._last_cells: list[list[tuple[int, int, int, int]]] = []

    # --------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> BoardDetectionResult:
        """执行完整检测流水线.

        Args:
            image: BGR 相机图像，shape ``(H, W, 3)``.

        Returns:
            :class:`BoardDetectionResult`.  使用前请检查 ``result.success`` 字段.
        """
        logger.info("开始棋盘检测 — 图像尺寸 %s", image.shape)

        # Step 1: 找到棋盘四角.
        corners = self._find_board_region(image)
        if corners is None:
            logger.warning("未在图像中找到棋盘区域")
            result = BoardDetectionResult()
            self._store_empty()
            return result

        # Step 2: 透视矫正.
        dst_size = self._WARPED_SIZE
        transform, warped = self._warp(image, corners, dst_size)

        # Step 3: 在矫正图上检测格线.
        h_lines, v_lines = self._detect_grid_lines(warped)

        # Step 4: 计算格点像素间距.
        grid_cell_size = float(dst_size) / float(BOARD_ROWS)
        if len(h_lines) >= BOARD_ROWS + 1 and len(v_lines) >= BOARD_COLS + 1:
            grid_cell_size = (
                (h_lines[-1] - h_lines[0]) / BOARD_ROWS + (v_lines[-1] - v_lines[0]) / BOARD_COLS
            ) / 2.0

        # Step 5: 计算网格单元格.
        cells = self._compute_grid_cells(h_lines, v_lines, dst_size)

        result = BoardDetectionResult(
            warped=warped,
            transform=transform,
            corners=corners,
            grid_cell_size=grid_cell_size,
            success=True,
        )

        self._last_result = result
        self._last_h_lines = h_lines
        self._last_v_lines = v_lines
        self._last_cells = cells

        logger.info(
            "棋盘检测完成 — 横线 %d 纵线 %d 格点间距 %.1f px",
            len(h_lines),
            len(v_lines),
            grid_cell_size,
        )
        return result

    def get_grid_cells(self) -> list[list[tuple[int, int, int, int]]]:
        """返回最近一次检测的 15×15 网格单元格.

        每个单元格为 ``(x, y, w, h)``，坐标系为矫正图像素空间.
        Row 0 为棋盘最上方，Row 14 为最下方.

        Returns:
            二维列表 ``cells[r][c]``，未成功检测时返回空列表.
        """
        return self._last_cells

    def draw_debug(self, image: np.ndarray) -> np.ndarray:
        """在原图上叠加检测标注，返回标注后的图像（不修改原图）.

        绘制内容:
            - 绿色四边形轮廓与红色角点
            - 角点标签 (TL/TR/BR/BL)
            - 蓝色格线 (反投影回原图)
            - 检测失败时显示红色文字提示
        """
        vis = image.copy()

        if self._last_result is None or not self._last_result.success:
            h, w = vis.shape[:2]
            cv2.putText(
                vis,
                "Board NOT found",
                (w // 4, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )
            return vis

        corners = self._last_result.corners.astype(np.int32)

        # 绿色多边形连接四角
        cv2.polylines(vis, [corners], True, (0, 255, 0), 2)

        # 红色角点 + 标签
        for i, pt in enumerate(corners):
            cv2.circle(vis, tuple(pt), 8, (0, 0, 255), -1)
            cv2.putText(
                vis,
                CALIB_CORNER_ORDER[i],
                (pt[0] + 10, pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

        # 反投影格线
        if self._last_h_lines or self._last_v_lines:
            try:
                h_inv = np.linalg.inv(self._last_result.transform)
            except np.linalg.LinAlgError:
                logger.debug("透视变换矩阵不可逆，跳过格线反投影")
                return vis

            dst_sz = self._WARPED_SIZE
            for y in self._last_h_lines:
                pts = np.array([[[0, y], [dst_sz - 1, y]]], dtype=np.float32)
                src_pts = cv2.perspectiveTransform(pts, h_inv).astype(np.int32)
                cv2.line(vis, tuple(src_pts[0, 0]), tuple(src_pts[0, 1]), (255, 0, 0), 1)
            for x in self._last_v_lines:
                pts = np.array([[[x, 0], [x, dst_sz - 1]]], dtype=np.float32)
                src_pts = cv2.perspectiveTransform(pts, h_inv).astype(np.int32)
                cv2.line(vis, tuple(src_pts[0, 0]), tuple(src_pts[0, 1]), (255, 0, 0), 1)

        return vis

    # --------------------------------------------------------------------
    # 内部方法
    # --------------------------------------------------------------------

    def _store_empty(self) -> None:
        """将内部状态重置为空."""
        self._last_result = BoardDetectionResult()
        self._last_h_lines = []
        self._last_v_lines = []
        self._last_cells = []

    @staticmethod
    def _preprocess(gray: np.ndarray) -> np.ndarray:
        """灰度图高斯模糊."""
        return cv2.GaussianBlur(gray, BoardDetector._BLUR_KERNEL, 0)

    # ---- 棋盘区域定位 ---------------------------------------------------------

    def _find_board_region(self, image: np.ndarray) -> np.ndarray | None:
        """在图像中定位棋盘四边形.

        策略:
            1. 转 HSV 并做木色阈值分割.
            2. 形态学闭运算填补空隙.
            3. 取最大外轮廓.
            4. 多边形逼近 → 四边形.
            5. 按 TL → TR → BR → BL 排序角点.

        Returns:
            ``(4, 2)`` float32 有序角点，失败返回 ``None``.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._board_color_low, self._board_color_high)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self._MORPH_KERNEL_SIZE)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self._MORPH_ITERATIONS)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.debug("HSV 掩码未产生任何轮廓")
            return None

        # 取面积最大的轮廓
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < self._MIN_CONTOUR_AREA:
            logger.debug("最大轮廓面积过小 (area=%.0f < %.0f)", area, self._MIN_CONTOUR_AREA)
            return None

        # 多边形逼近
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

        if len(approx) < 4:
            logger.debug("多边形逼近得到 %d 个顶点 (< 4)", len(approx))
            return None

        # 提纯为四边形
        if len(approx) > 4:
            rect = cv2.minAreaRect(largest)
            approx = cv2.boxPoints(rect)
        else:
            approx = approx.reshape(4, 2)

        corners = self._order_corners(approx)
        logger.debug("棋盘角点: %s", corners.tolist())
        return corners.astype(np.float32)

    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        """将四个点排序为 左上 → 右上 → 右下 → 左下."""
        ordered = np.zeros((4, 2), dtype=np.float32)

        # x+y 最小 → 左上, x+y 最大 → 右下
        s = pts.sum(axis=1)
        ordered[0] = pts[np.argmin(s)]
        ordered[2] = pts[np.argmax(s)]

        # y-x 最小 → 右上, y-x 最大 → 左下
        d = np.diff(pts, axis=1)
        ordered[1] = pts[np.argmin(d)]
        ordered[3] = pts[np.argmax(d)]

        return ordered

    # ---- 透视矫正 -------------------------------------------------------------

    @staticmethod
    def _warp(
        image: np.ndarray, corners: np.ndarray, dst_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算透视变换并生成正方形俯视矫正图.

        Returns:
            (transform, warped)
        """
        dst_corners = np.array(
            [
                [0, 0],
                [dst_size - 1, 0],
                [dst_size - 1, dst_size - 1],
                [0, dst_size - 1],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(corners, dst_corners)
        warped = cv2.warpPerspective(image, transform, (dst_size, dst_size))
        return transform, warped

    # ---- 格线检测 -------------------------------------------------------------

    def _detect_grid_lines(self, warped: np.ndarray) -> tuple[list[int], list[int]]:
        """在矫正图上检测并聚类格线.

        Returns:
            ``(h_lines, v_lines)`` — 排序后的横线 y 坐标与纵线 x 坐标列表.
        """
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blurred = self._preprocess(gray)
        edges = cv2.Canny(blurred, self._canny_low, self._canny_high)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self._hough_threshold,
            minLineLength=self._min_line_length,
            maxLineGap=self._max_line_gap,
        )

        if lines is None:
            logger.debug("霍夫线检测未返回任何线段")
            return [], []

        h_mids: list[int] = []
        v_mids: list[int] = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)

            if angle < 10.0 or angle > 170.0:
                # 水平线 (±10°)
                h_mids.append(int((y1 + y2) / 2))
            elif 80.0 < angle < 100.0:
                # 垂直线 (±10°)
                v_mids.append(int((x1 + x2) / 2))

        h_lines = self._cluster_lines(h_mids, expected=BOARD_ROWS + 1)
        v_lines = self._cluster_lines(v_mids, expected=BOARD_COLS + 1)

        logger.debug(
            "格线聚类结果 — 横线 %d 纵线 %d (原始中点 %d / %d)",
            len(h_lines),
            len(v_lines),
            len(h_mids),
            len(v_mids),
        )
        return h_lines, v_lines

    @staticmethod
    def _cluster_lines(midpoints: list[int], expected: int) -> list[int]:
        """将一维坐标点聚类为 *expected* 个等量分箱，返回每箱中位数的排序列表.

        当唯一点数不足以填满期望时，直接返回排序唯一点.
        """
        if not midpoints:
            return []

        sorted_pts = sorted(set(midpoints))
        if len(sorted_pts) <= expected:
            return sorted_pts

        # 等量分箱取中位数
        n = len(sorted_pts)
        clustered: list[int] = []
        for i in range(expected):
            start = int(round(i * n / expected))
            end = int(round((i + 1) * n / expected))
            bin_pts = sorted_pts[start:end]
            if bin_pts:
                clustered.append(int(np.median(bin_pts)))
        return clustered

    # ---- 网格单元格 -----------------------------------------------------------

    def _compute_grid_cells(
        self, h_lines: list[int], v_lines: list[int], dst_size: int
    ) -> list[list[tuple[int, int, int, int]]]:
        """从检测到的格线计算 15×15 网格单元格.

        当检测到的线数足够 (≥16 横 + 16 纵) 时，单元格由相邻线围成；
        否则使用均匀网格回退.
        """
        cells: list[list[tuple[int, int, int, int]]] = []

        if len(h_lines) >= BOARD_ROWS + 1 and len(v_lines) >= BOARD_COLS + 1:
            h_positions = h_lines[: BOARD_ROWS + 1]
            v_positions = v_lines[: BOARD_COLS + 1]
        else:
            # 均匀网格回退
            step = float(dst_size) / float(BOARD_ROWS)
            h_positions = [int(round(i * step)) for i in range(BOARD_ROWS + 1)]
            v_positions = [int(round(i * step)) for i in range(BOARD_COLS + 1)]
            logger.debug("格线不足，使用均匀网格回退 (step=%.1f)", step)

        for r in range(BOARD_ROWS):
            y1, y2 = h_positions[r], h_positions[r + 1]
            row_cells: list[tuple[int, int, int, int]] = []
            for c in range(BOARD_COLS):
                x1, x2 = v_positions[c], v_positions[c + 1]
                row_cells.append((x1, y1, x2 - x1, y2 - y1))
            cells.append(row_cells)

        return cells
