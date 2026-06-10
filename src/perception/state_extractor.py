"""Board state extractor — unified perception pipeline entry point.

Integrates camera capture, board detection, and stone detection into a single
call that returns the 15×15 board state matrix plus an optional move delta
identifying the most recent stone placement.
"""

from __future__ import annotations

import logging

import numpy as np

from src.perception.board_detector import BoardDetector
from src.perception.camera import MockCamera, WebCamera, create_camera
from src.perception.stone_detector import StoneDetector
from src.utils.config_loader import load_config
from src.utils.constants import EMPTY, EMPTY_BOARD

logger = logging.getLogger(__name__)


class StateExtractor:
    """Unified perception pipeline that produces the full board state plus a
    move delta from a single camera frame.

    Internally wires together :class:`BoardDetector` and :class:`StoneDetector`
    so downstream code (orchestrator, game) only calls one method.

    Usage::

        extractor = StateExtractor(config=cfg)
        # or with an existing camera:
        extractor = StateExtractor(camera=cam, config=cfg)

        board, delta = extractor.extract()
        # board: (15, 15) int8 matrix
        # delta: (row, col, stone) or None
    """

    def __init__(
        self,
        camera: WebCamera | MockCamera | None = None,
        config: dict | None = None,
    ) -> None:
        """Initialize the full perception pipeline.

        Args:
            camera: An existing camera object (:class:`WebCamera` or
                :class:`MockCamera`). When *None*, a camera is auto-created
                via :func:`create_camera`.
            config: Optional config dictionary.  When *None* and no camera is
                supplied, loaded from the default YAML via :func:`load_config`.
        """
        if config is None:
            config = load_config()

        self._config: dict = config

        if camera is not None:
            self._camera = camera
        else:
            self._camera = create_camera(config)

        self._board_detector = BoardDetector(config)
        self._stone_detector = StoneDetector(config)

        self._previous_state: np.ndarray | None = None

        logger.info("StateExtractor initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> tuple[np.ndarray, tuple[int, int, int] | None]:
        """Execute the full perception pipeline on a freshly captured frame.

        1. Capture a colour frame from the camera.
        2. Detect the board and compute the perspective warp.
        3. Run stone detection on the warped board.
        4. Compare with the previous board state to find the move delta.

        Returns:
            ``(board_state, delta)`` where:

            * **board_state** — ``(15, 15)`` int8 matrix (0=empty, 1=black, 2=white).
            * **delta** — ``(row, col, stone)`` if a single new stone was
              detected since the last call, otherwise ``None``.
        """
        image = self._camera.get_color()
        return self.extract_from_image(image)

    def extract_from_image(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int, int] | None]:
        """Run the pipeline on a pre-captured BGR image.

        Identical to :meth:`extract` except the image is supplied directly
        instead of captured from the camera.  Useful for testing with saved
        frames or offline replay.

        Args:
            image: BGR camera image, shape ``(H, W, 3)``.

        Returns:
            ``(board_state, delta)`` — see :meth:`extract`.
        """
        # 1. Detect the board region and compute perspective warp.
        result = self._board_detector.detect(image)

        if not result.success:
            logger.warning("Board detection failed — returning fallback state")
            fallback = (
                self._previous_state.copy()
                if self._previous_state is not None
                else EMPTY_BOARD.copy()
            )
            return fallback, None

        # 2. Retrieve grid cells and run stone detection.
        cells = self._board_detector.get_grid_cells()
        board_state = self._stone_detector.detect(result.warped, cells)

        # 3. Compute move delta against the previous state.
        delta = self._compute_delta(self._previous_state, board_state)

        # 4. Persist current state for next delta comparison.
        self._previous_state = board_state.copy()

        return board_state, delta

    def get_previous_state(self) -> np.ndarray:
        """Return the board state from the previous :meth:`extract` call.

        Returns:
            ``(15, 15)`` int8 matrix.  On first call (no extraction has run)
            returns the empty board with all zeros.

            The returned array is a **new copy** — mutating it will not affect
            the extractor's internal state.
        """
        if self._previous_state is None:
            return EMPTY_BOARD.copy()
        return self._previous_state.copy()

    def reset(self) -> None:
        """Clear all internal state so the next :meth:`extract` call has no
        prior board to compare against (delta will be ``None``)."""
        self._previous_state = None
        logger.info("StateExtractor internal state reset")

    # ------------------------------------------------------------------
    # Properties — expose underlying detectors for debugging
    # ------------------------------------------------------------------

    @property
    def board_detector(self) -> BoardDetector:
        """Access the underlying :class:`BoardDetector` for debugging and
        visualisation (e.g. :meth:`BoardDetector.draw_debug`)."""
        return self._board_detector

    @property
    def stone_detector(self) -> StoneDetector:
        """Access the underlying :class:`StoneDetector` for debugging and
        per-cell inspection (e.g. :meth:`StoneDetector.detect_cell`)."""
        return self._stone_detector

    @property
    def camera(self) -> WebCamera | MockCamera:
        """Access the camera object for interactive debugging."""
        return self._camera

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_delta(
        previous: np.ndarray | None,
        current: np.ndarray,
    ) -> tuple[int, int, int] | None:
        """Compare the previous and current board states to identify the move.

        A delta is returned only when exactly one stone was added (stone count
        increased by 1 since the previous extraction).  If stones were removed
        or multiple positions changed, ``None`` is returned.

        Args:
            previous: The ``(15, 15)`` board matrix from the previous call,
                or ``None`` on the first extraction.
            current: The newly detected ``(15, 15)`` board matrix.

        Returns:
            ``(row, col, stone)`` for a single new move, or ``None``.
        """
        if previous is None:
            return None

        prev_count = int(np.count_nonzero(previous != EMPTY))
        curr_count = int(np.count_nonzero(current != EMPTY))

        if curr_count != prev_count + 1:
            logger.debug(
                "Delta skipped — stone count changed %d → %d (expected +1)",
                prev_count,
                curr_count,
            )
            return None

        # Find the cell that changed from EMPTY to a stone.
        changed_mask = (previous == EMPTY) & (current != EMPTY)
        changed_rows, changed_cols = np.where(changed_mask)

        if changed_rows.size != 1:
            logger.debug(
                "Delta ambiguous — %d cells changed from EMPTY (expected 1)",
                changed_rows.size,
            )
            return None

        row = int(changed_rows[0])
        col = int(changed_cols[0])
        stone = int(current[row, col])

        logger.info("Move delta detected — (%d, %d) stone=%d", row, col, stone)
        return (row, col, stone)
