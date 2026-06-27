"""Grid mapper — bridge between warped image pixel coordinates and board (row, col).

Maps the 15×15 Gomoku intersection hit boxes from :class:`BoardDetector` to
discrete board positions, and optionally to physical robot coordinates via
calibration.

Coordinate systems:
    - **Pixel**: (x, y) in the warped square image (default 600×600).
    - **Grid**: (row, col) on the 0‑based 15×15 board.
    - **Robot**: (x, y, z) in the SO‑ARM101 workspace.  Only available after
      :meth:`GridMapper.calibrate_robot` is called.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.robot.controller import CalibrationPoints, board_to_robot_coords
from src.utils.constants import BOARD_COLS, BOARD_ROWS

logger = logging.getLogger(__name__)


@dataclass
class CellPosition:
    """A position on the board in multiple coordinate systems.

    Attributes:
        row: Board row index (0‑based, 0 = top).
        col: Board column index (0‑based, 0 = left).
        pixel_x: Center x in the warped image (px).
        pixel_y: Center y in the warped image (px).
        robot_x: Robot x coordinate, or ``None`` before calibration.
        robot_y: Robot y coordinate, or ``None`` before calibration.
        robot_z: Robot z coordinate (height), or ``None`` before calibration.
    """

    row: int
    col: int
    pixel_x: float
    pixel_y: float
    robot_x: float | None = None
    robot_y: float | None = None
    robot_z: float | None = None


class GridMapper:
    """Map between warped‑image pixel coordinates and board (row, col).

    Consumes the intersection hit boxes produced by
    :meth:`BoardDetector.get_grid_cells` and exposes conversions in both
    directions. Once robot calibration points are supplied, every board
    position also carries its physical workspace coordinate.

    Usage::

        mapper = GridMapper()
        mapper.load_cells(detector.get_grid_cells())

        row, col = mapper.pixel_to_grid(300.0, 250.0)
        px, py = mapper.grid_to_pixel(7, 7)

        mapper.calibrate_robot(calib_points, z_height=45.0)
        pos = mapper.get_cell(7, 7)
        print(pos.robot_x, pos.robot_y, pos.robot_z)
    """

    def __init__(self, board_rows: int = BOARD_ROWS, board_cols: int = BOARD_COLS) -> None:
        """Initialize the mapper.

        Args:
            board_rows: Number of rows on the board (default 15).
            board_cols: Number of columns on the board (default 15).
        """
        self._board_rows: int = board_rows
        self._board_cols: int = board_cols

        # intersection hit boxes: _cells[row][col] = (x, y, w, h) in warped image px
        self._cells: list[list[tuple[int, int, int, int]]] = []
        # intersection centers: _centers[row][col] = (cx, cy) in warped image px
        self._centers: list[list[tuple[float, float]]] = []
        # position grid: _positions[row][col] = CellPosition
        self._positions: list[list[CellPosition]] = []

        self._is_loaded: bool = False
        self._is_calibrated: bool = False

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------

    def load_cells(self, cells: list[list[tuple[int, int, int, int]]]) -> None:
        """Load intersection hit boxes from :meth:`BoardDetector.get_grid_cells`.

        Computes the pixel-space center of every board position and populates the
        internal position grid (robot coordinates remain ``None``).

        Args:
            cells: A ``cells[row][col] = (x, y, w, h)`` structure as returned
                   by the detector. Must be ``board_rows x board_cols``.
        """
        self._cells = cells

        self._centers = []
        self._positions = []

        for row in range(self._board_rows):
            center_row: list[tuple[float, float]] = []
            pos_row: list[CellPosition] = []
            for col in range(self._board_cols):
                x, y, w, h = cells[row][col]
                cx = float(x + w / 2.0)
                cy = float(y + h / 2.0)
                center_row.append((cx, cy))
                pos_row.append(CellPosition(row=row, col=col, pixel_x=cx, pixel_y=cy))
            self._centers.append(center_row)
            self._positions.append(pos_row)

        self._is_loaded = True
        self._is_calibrated = False
        logger.info(
            "Loaded intersection hit boxes for %dx%d board",
            self._board_rows,
            self._board_cols,
        )

    # ------------------------------------------------------------------
    # coordinate conversions
    # ------------------------------------------------------------------

    def pixel_to_grid(self, px: float, py: float) -> tuple[int, int]:
        """Convert a warped‑image pixel position to the nearest (row, col).

        Performs a nearest-position lookup: if the point falls inside an
        intersection hit box, that board position is returned. If the point
        lies outside the board, the closest edge position is returned instead.

        Args:
            px: X coordinate in the warped image.
            py: Y coordinate in the warped image.

        Returns:
            ``(row, col)`` — 0‑based board coordinates.

        Raises:
            RuntimeError: If :meth:`load_cells` has not been called yet.
        """
        if not self._is_loaded:
            raise RuntimeError("No cells loaded — call load_cells() first")

        # 1) Direct containment lookup
        for row in range(self._board_rows):
            for col in range(self._board_cols):
                x, y, w, h = self._cells[row][col]
                if x <= px <= x + w and y <= py <= y + h:
                    return (row, col)

        # 2) Point outside all boxes — nearest intersection by Euclidean distance
        best_row, best_col = 0, 0
        best_dist2 = float("inf")
        for row in range(self._board_rows):
            for col in range(self._board_cols):
                cx, cy = self._centers[row][col]
                d2 = (px - cx) ** 2 + (py - cy) ** 2
                if d2 < best_dist2:
                    best_dist2 = d2
                    best_row, best_col = row, col

        logger.debug(
            "Point (%.1f, %.1f) outside board, clamped to (%d, %d)",
            px,
            py,
            best_row,
            best_col,
        )
        return (best_row, best_col)

    def grid_to_pixel(self, row: int, col: int) -> tuple[float, float]:
        """Get the pixel center of board intersection ``(row, col)``.

        Args:
            row: 0‑based row index.
            col: 0‑based column index.

        Returns:
            ``(pixel_x, pixel_y)`` in the warped image.

        Raises:
            RuntimeError: If :meth:`load_cells` has not been called yet.
            IndexError: If *row* or *col* is out of bounds.
        """
        if not self._is_loaded:
            raise RuntimeError("No cells loaded — call load_cells() first")

        if not (0 <= row < self._board_rows and 0 <= col < self._board_cols):
            raise IndexError(
                f"Cell ({row}, {col}) out of range for {self._board_rows}×{self._board_cols} board"
            )
        return self._centers[row][col]

    # ------------------------------------------------------------------
    # board-position access
    # ------------------------------------------------------------------

    def get_cell(self, row: int, col: int) -> CellPosition:
        """Return the :class:`CellPosition` for a specific board position.

        Args:
            row: 0‑based row index.
            col: 0‑based column index.

        Returns:
            The position data in all available coordinate systems.

        Raises:
            RuntimeError: If :meth:`load_cells` has not been called yet.
            IndexError: If *row* or *col* is out of bounds.
        """
        if not self._is_loaded:
            raise RuntimeError("No cells loaded — call load_cells() first")

        if not (0 <= row < self._board_rows and 0 <= col < self._board_cols):
            raise IndexError(
                f"Cell ({row}, {col}) out of range for {self._board_rows}×{self._board_cols} board"
            )
        return self._positions[row][col]

    def get_all_positions(self) -> list[list[CellPosition]]:
        """Return the full 15×15 grid of :class:`CellPosition` objects.

        Returns:
            A ``positions[row][col]`` 2‑D list.  Robot coordinates are
            ``None`` on every intersection until :meth:`calibrate_robot` is called.

        Raises:
            RuntimeError: If :meth:`load_cells` has not been called yet.
        """
        if not self._is_loaded:
            raise RuntimeError("No cells loaded — call load_cells() first")
        return self._positions

    # ------------------------------------------------------------------
    # robot calibration
    # ------------------------------------------------------------------

    def calibrate_robot(self, calib: CalibrationPoints, z_height: float = 50.0) -> None:
        """Set robot calibration and populate robot coordinates for every position.

        Uses :func:`src.robot.controller.board_to_robot_coords` to compute the
        physical ``(x, y, z)`` of each of the 225 intersections via bilinear
        interpolation between the four calibration corner points.

        Args:
            calib: Four‐point calibration data in robot workspace coordinates.
            z_height: Z coordinate (height) used when approaching the board.
                      Defaults to 50.0 mm.

        Raises:
            RuntimeError: If :meth:`load_cells` has not been called yet.
        """
        if not self._is_loaded:
            raise RuntimeError("No cells loaded — call load_cells() first")

        for row in range(self._board_rows):
            for col in range(self._board_cols):
                rx, ry, rz = board_to_robot_coords(
                    row=row,
                    col=col,
                    board_rows=self._board_rows,
                    board_cols=self._board_cols,
                    calib=calib,
                    z_height=z_height,
                )
                pos = self._positions[row][col]
                pos.robot_x = rx
                pos.robot_y = ry
                pos.robot_z = rz

        self._is_calibrated = True
        logger.info(
            "Robot calibration applied — z_height=%.1f mm, %d cells populated",
            z_height,
            self._board_rows * self._board_cols,
        )
