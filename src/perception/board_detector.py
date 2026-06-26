"""棋盘检测器 — 支持手动标定与自动检测两种模式。

推荐模式: ``manual``
    相机与棋盘固定时，通过标定工具预先记录棋盘四角坐标。
    检测器直接使用预存角点做透视矫正，不需要每帧自动检测。

自动模式: ``auto``
    自适应阈值 + Hough 线聚类自动寻找棋盘区域。
    用于棋盘或相机位置可能变化的场景。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.perception.board_calibration import (
    board_frame_required,
    load_board_frame_calibration,
)
from src.utils.constants import (
    BOARD_COLS,
    BOARD_ROWS,
    CALIB_CORNER_ORDER,
)

logger = logging.getLogger(__name__)


@dataclass
class BoardDetectionResult:
    """棋盘检测结果。"""

    warped: np.ndarray = field(default_factory=lambda: np.zeros((600, 600, 3), dtype=np.uint8))
    transform: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    corners: np.ndarray = field(default_factory=lambda: np.zeros((4, 2), dtype=np.float32))
    grid_cell_size: float = 0.0
    success: bool = False


class BoardDetector:
    """棋盘检测器。

    两种模式：

    - ``manual``: 使用 ``board.calibration.corners`` 中的预标定四角。
    - ``auto``:  自适应阈值 + Hough 线聚类。
    """

    _WARPED_SIZE: int = 600
    _BLUR_KERNEL: tuple[int, int] = (5, 5)

    # Auto-mode defaults
    _CANNY_LOW: float = 50.0
    _CANNY_HIGH: float = 150.0
    _HOUGH_THRESHOLD: int = 100
    _MIN_LINE_LENGTH: float = 200.0
    _MAX_LINE_GAP: float = 50.0

    def __init__(self, config: dict | None = None) -> None:
        """初始化。

        Args:
            config: 完整配置字典。读取 ``board.calibration`` 和
                ``board_detection`` 段。
        """
        det_cfg: dict = (config or {}).get("board_detection", {})

        default_method = "manual" if board_frame_required(config or {}) else "auto"
        self._method: str = det_cfg.get("method", default_method)

        # ---- 加载预标定角点 ----
        self._calib_corners: np.ndarray | None = None
        frame_calibration = load_board_frame_calibration(
            config or {},
            required=self._method == "manual",
        )
        if frame_calibration is not None:
            self._calib_corners = frame_calibration.as_array()
            logger.info("使用手动标定角点模式，跳过棋盘检测")

        # Auto-mode 参数
        self._canny_low: float = float(det_cfg.get("canny_low", self._CANNY_LOW))
        self._canny_high: float = float(det_cfg.get("canny_high", self._CANNY_HIGH))
        self._hough_threshold: int = int(det_cfg.get("hough_threshold", self._HOUGH_THRESHOLD))
        self._min_line_length: float = float(det_cfg.get("min_line_length", self._MIN_LINE_LENGTH))
        self._max_line_gap: float = float(det_cfg.get("max_line_gap", self._MAX_LINE_GAP))

        self._last_result: BoardDetectionResult | None = None
        self._last_h_lines: list[int] = []
        self._last_v_lines: list[int] = []
        self._last_cells: list[list[tuple[int, int, int, int]]] = []

    # ---- Public API --------------------------------------------------------

    def detect(self, image: np.ndarray) -> BoardDetectionResult:
        """执行检测流水线。

        manual 模式直接使用预标定角点，auto 模式自动检测。
        """
        logger.info("棋盘检测 — 尺寸 %s 模式=%s", image.shape, self._method)

        dst = self._WARPED_SIZE

        if self._calib_corners is not None:
            corners = self._calib_corners
        else:
            corners = self._find_board_region(image)
            if corners is None:
                logger.warning("未找到棋盘区域")
                result = BoardDetectionResult()
                self._store_empty()
                return result

        transform, warped = self._warp(image, corners, dst)

        h_lines, v_lines = self._detect_grid_lines(warped)

        grid_cell_size = float(dst) / float(BOARD_ROWS)
        if len(h_lines) >= BOARD_ROWS + 1 and len(v_lines) >= BOARD_COLS + 1:
            grid_cell_size = (
                (h_lines[-1] - h_lines[0]) / BOARD_ROWS + (v_lines[-1] - v_lines[0]) / BOARD_COLS
            ) / 2.0

        cells = self._compute_grid_cells(h_lines, v_lines, dst)

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
            "检测完成 — 横线 %d 纵线 %d 间距 %.1f px", len(h_lines), len(v_lines), grid_cell_size
        )
        return result

    def get_grid_cells(self) -> list[list[tuple[int, int, int, int]]]:
        return self._last_cells

    def draw_debug(self, image: np.ndarray) -> np.ndarray:
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
        cv2.polylines(vis, [corners], True, (0, 255, 0), 2)
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

        if self._last_h_lines or self._last_v_lines:
            try:
                h_inv = np.linalg.inv(self._last_result.transform)
            except np.linalg.LinAlgError:
                return vis
            for y in self._last_h_lines:
                pts = np.array([[[0, y], [self._WARPED_SIZE - 1, y]]], dtype=np.float32)
                src = cv2.perspectiveTransform(pts, h_inv).astype(np.int32)
                cv2.line(vis, tuple(src[0, 0]), tuple(src[0, 1]), (255, 0, 0), 1)
            for x in self._last_v_lines:
                pts = np.array([[[x, 0], [x, self._WARPED_SIZE - 1]]], dtype=np.float32)
                src = cv2.perspectiveTransform(pts, h_inv).astype(np.int32)
                cv2.line(vis, tuple(src[0, 0]), tuple(src[0, 1]), (255, 0, 0), 1)

        return vis

    # ---- Internal: state ---------------------------------------------------

    def _store_empty(self) -> None:
        self._last_result = BoardDetectionResult()
        self._last_h_lines = []
        self._last_v_lines = []
        self._last_cells = []

    # ---- Internal: auto-detection ------------------------------------------

    def _find_board_region(self, image: np.ndarray) -> np.ndarray | None:
        """自动检测棋盘四角（自适应阈值 + Hough 线聚类）。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8
        )
        lines = cv2.HoughLinesP(th, 1, np.pi / 180, threshold=40, minLineLength=60, maxLineGap=20)
        if lines is None:
            return None

        h_mids, v_mids = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 15 or angle > 165:
                h_mids.append(int((y1 + y2) / 2))
            elif 75 < angle < 105:
                v_mids.append(int((x1 + x2) / 2))

        if len(h_mids) < 10 or len(v_mids) < 5:
            return None

        bounds = self._find_grid_cluster(h_mids, v_mids)
        if bounds is None:
            return None

        left, top, right, bottom = bounds
        return np.array(
            [[left, top], [right, top], [right, bottom], [left, bottom]],
            dtype=np.float32,
        )

    @staticmethod
    def _find_grid_cluster(h_mids, v_mids):
        def spacing(positions):
            u = sorted(set(positions))
            if len(u) < 3:
                return None
            g = [u[i + 1] - u[i] for i in range(len(u) - 1)]
            g = [x for x in g if 5 < x < 200]
            return float(np.median(g)) if len(g) >= 2 else None

        def cluster(positions, sp, min_sz=8):
            u = sorted(set(positions))
            cl, cur = [], [u[0]]
            for i in range(1, len(u)):
                if u[i] - u[i - 1] <= 3 * sp:
                    cur.append(u[i])
                else:
                    if len(cur) >= min_sz:
                        cl.append(cur)
                    cur = [u[i]]
            if len(cur) >= min_sz:
                cl.append(cur)
            return max(cl, key=len) if cl else []

        hs, vs = spacing(h_mids), spacing(v_mids)
        if hs is None or vs is None:
            return None
        hc, vc = cluster(h_mids, hs), cluster(v_mids, vs)
        if not hc or not vc:
            return None
        return min(vc), min(hc), max(vc), max(hc)

    # ---- Internal: perspective warp ----------------------------------------

    @staticmethod
    def _warp(image, corners, dst_size):
        dst_c = np.array(
            [[0, 0], [dst_size - 1, 0], [dst_size - 1, dst_size - 1], [0, dst_size - 1]],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(corners, dst_c)
        return transform, cv2.warpPerspective(image, transform, (dst_size, dst_size))

    # ---- Internal: grid lines on warped image ------------------------------

    def _detect_grid_lines(self, warped):
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, self._BLUR_KERNEL, 0)
        edges = cv2.Canny(blur, self._canny_low, self._canny_high)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=self._hough_threshold,
            minLineLength=self._min_line_length,
            maxLineGap=self._max_line_gap,
        )
        if lines is None:
            return [], []

        h_mids, v_mids = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 10 or angle > 170:
                h_mids.append(int((y1 + y2) / 2))
            elif 80 < angle < 100:
                v_mids.append(int((x1 + x2) / 2))

        return self._cluster_lines(h_mids, BOARD_ROWS + 1), self._cluster_lines(
            v_mids, BOARD_COLS + 1
        )

    @staticmethod
    def _cluster_lines(midpoints, expected):
        if not midpoints:
            return []
        pts = sorted(set(midpoints))
        if len(pts) <= expected:
            return pts
        n = len(pts)
        r = []
        for i in range(expected):
            s = int(round(i * n / expected))
            e = int(round((i + 1) * n / expected))
            b = pts[s:e]
            if b:
                r.append(int(np.median(b)))
        return r

    # ---- Internal: grid cells ----------------------------------------------

    def _compute_grid_cells(self, h_lines, v_lines, dst_size):
        cells = []
        if len(h_lines) >= BOARD_ROWS + 1 and len(v_lines) >= BOARD_COLS + 1:
            hp = h_lines[: BOARD_ROWS + 1]
            vp = v_lines[: BOARD_COLS + 1]
        else:
            step = float(dst_size) / BOARD_ROWS
            hp = [int(round(i * step)) for i in range(BOARD_ROWS + 1)]
            vp = [int(round(i * step)) for i in range(BOARD_COLS + 1)]
        for r in range(BOARD_ROWS):
            row = []
            for c in range(BOARD_COLS):
                row.append((vp[c], hp[r], vp[c + 1] - vp[c], hp[r + 1] - hp[r]))
            cells.append(row)
        return cells
