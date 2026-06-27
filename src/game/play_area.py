"""Playable area helpers for a full Gomoku board.

The camera and board state can remain 15x15 while the robot only plays inside a
smaller reliable window, for example the centered 9x9 area covered by the
measured SO101 pose map.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.constants import BOARD_COLS, BOARD_ROWS, EMPTY


@dataclass(frozen=True)
class PlayArea:
    """A rectangular playable window inside the full 15x15 board state."""

    row_offset: int = 0
    col_offset: int = 0
    rows: int = BOARD_ROWS
    cols: int = BOARD_COLS
    board_rows: int = BOARD_ROWS
    board_cols: int = BOARD_COLS

    def __post_init__(self) -> None:
        values = {
            "row_offset": self.row_offset,
            "col_offset": self.col_offset,
            "rows": self.rows,
            "cols": self.cols,
            "board_rows": self.board_rows,
            "board_cols": self.board_cols,
        }
        for name, value in values.items():
            int_value = int(value)
            if int_value != value:
                raise ValueError(f"PlayArea {name} must be an integer")
            object.__setattr__(self, name, int_value)

        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("PlayArea rows and cols must be positive")
        if self.board_rows <= 0 or self.board_cols <= 0:
            raise ValueError("PlayArea board size must be positive")
        if self.row_offset < 0 or self.col_offset < 0:
            raise ValueError("PlayArea offsets must be non-negative")
        if self.row_offset + self.rows > self.board_rows:
            raise ValueError("PlayArea rows exceed board bounds")
        if self.col_offset + self.cols > self.board_cols:
            raise ValueError("PlayArea cols exceed board bounds")

    @classmethod
    def full(cls, board_rows: int = BOARD_ROWS, board_cols: int = BOARD_COLS) -> PlayArea:
        """Return a play area that covers the full board."""

        return cls(rows=board_rows, cols=board_cols, board_rows=board_rows, board_cols=board_cols)

    @classmethod
    def centered(
        cls,
        rows: int,
        cols: int | None = None,
        *,
        board_rows: int = BOARD_ROWS,
        board_cols: int = BOARD_COLS,
    ) -> PlayArea:
        """Return a centered play area inside the full board."""

        active_cols = rows if cols is None else cols
        return cls(
            row_offset=(board_rows - rows) // 2,
            col_offset=(board_cols - active_cols) // 2,
            rows=rows,
            cols=active_cols,
            board_rows=board_rows,
            board_cols=board_cols,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols

    @property
    def is_full_board(self) -> bool:
        return (
            self.row_offset == 0
            and self.col_offset == 0
            and self.rows == self.board_rows
            and self.cols == self.board_cols
        )

    def contains(self, row: int, col: int) -> bool:
        """Return whether a full-board coordinate is inside the play area."""

        return (
            self.row_offset <= row < self.row_offset + self.rows
            and self.col_offset <= col < self.col_offset + self.cols
        )

    def to_local(self, row: int, col: int) -> tuple[int, int]:
        """Convert a full-board coordinate to play-area local coordinates."""

        if not self.contains(row, col):
            raise ValueError(
                f"Position ({row}, {col}) is outside play area "
                f"{self.describe(include_board=False)}"
            )
        return row - self.row_offset, col - self.col_offset

    def to_global(self, row: int, col: int) -> tuple[int, int]:
        """Convert a play-area local coordinate to full-board coordinates."""

        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Local position ({row}, {col}) is outside {self.rows}x{self.cols}")
        return row + self.row_offset, col + self.col_offset

    def crop(self, board: np.ndarray) -> np.ndarray:
        """Return a copy of the play-area state from a full-board matrix."""

        self._check_board_shape(board)
        return board[
            self.row_offset : self.row_offset + self.rows,
            self.col_offset : self.col_offset + self.cols,
        ].copy()

    def filter_board_state(self, board: np.ndarray) -> np.ndarray:
        """Return a full-board copy with positions outside the play area empty."""

        self._check_board_shape(board)
        filtered = np.zeros((self.board_rows, self.board_cols), dtype=np.int8)
        filtered[
            self.row_offset : self.row_offset + self.rows,
            self.col_offset : self.col_offset + self.cols,
        ] = self.crop(board)
        return filtered

    def is_full(self, board: np.ndarray) -> bool:
        """Return whether all playable positions are occupied."""

        return not np.any(self.crop(board) == EMPTY)

    def describe(self, *, include_board: bool = True) -> str:
        """Return a compact human-readable description."""

        desc = (
            f"{self.rows}x{self.cols} at "
            f"row_offset={self.row_offset}, col_offset={self.col_offset}"
        )
        if include_board:
            desc += f" on {self.board_rows}x{self.board_cols}"
        return desc

    def _check_board_shape(self, board: np.ndarray) -> None:
        if board.shape != (self.board_rows, self.board_cols):
            raise ValueError(
                f"Expected board shape {(self.board_rows, self.board_cols)}, got {board.shape}"
            )


def parse_play_area_config(
    value: Any,
    *,
    board_rows: int = BOARD_ROWS,
    board_cols: int = BOARD_COLS,
) -> PlayArea:
    """Parse ``game.play_area`` from config.

    Supported examples:

    ``"full"``
    ``"center_9x9"``
    ``{"rows": 9, "cols": 9, "row_offset": 3, "col_offset": 3}``
    ``{"size": 9, "centered": true}``
    """

    if value in (None, ""):
        return PlayArea.full(board_rows=board_rows, board_cols=board_cols)

    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        if normalized == "full":
            return PlayArea.full(board_rows=board_rows, board_cols=board_cols)
        if normalized in {"center_9x9", "centered_9x9", "9x9"}:
            return PlayArea.centered(9, board_rows=board_rows, board_cols=board_cols)
        raise ValueError(f"Unknown game.play_area value: {value!r}")

    if not isinstance(value, Mapping):
        raise ValueError("game.play_area must be a string or mapping")

    size = value.get("size")
    rows = int(value.get("rows", size if size is not None else board_rows))
    cols = int(value.get("cols", size if size is not None else board_cols))
    centered = (
        bool(value.get("centered", False))
        or str(value.get("origin", "")).lower() == "center"
    )

    row_offset_value = value.get("row_offset", value.get("top"))
    col_offset_value = value.get("col_offset", value.get("left"))
    if centered and row_offset_value is None and col_offset_value is None:
        return PlayArea.centered(rows, cols, board_rows=board_rows, board_cols=board_cols)

    row_offset = 0 if row_offset_value is None else int(row_offset_value)
    col_offset = 0 if col_offset_value is None else int(col_offset_value)
    return PlayArea(
        row_offset=row_offset,
        col_offset=col_offset,
        rows=rows,
        cols=cols,
        board_rows=board_rows,
        board_cols=board_cols,
    )
