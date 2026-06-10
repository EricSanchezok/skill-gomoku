"""Robot controller stub — SO-ARM101 机械臂控制.

Implemented by another teammate.
"""

from dataclasses import dataclass
from typing import Sequence


@dataclass
class CalibrationPoints:
    """四点标定数据：机械臂坐标系下棋盘四角的 (x, y, z)。"""

    top_left: tuple[float, float, float]
    top_right: tuple[float, float, float]
    bottom_right: tuple[float, float, float]
    bottom_left: tuple[float, float, float]

    @classmethod
    def from_list(cls, points: Sequence[tuple[float, float, float]]) -> "CalibrationPoints":
        if len(points) != 4:
            raise ValueError("Expected 4 calibration points")
        return cls(
            top_left=tuple(points[0]),
            top_right=tuple(points[1]),
            bottom_right=tuple(points[2]),
            bottom_left=tuple(points[3]),
        )

    def to_list(self) -> list[tuple[float, float, float]]:
        return [self.top_left, self.top_right, self.bottom_right, self.bottom_left]


def board_to_robot_coords(
    row: int,
    col: int,
    board_rows: int,
    board_cols: int,
    calib: CalibrationPoints,
    z_height: float,
) -> tuple[float, float, float]:
    """将棋盘行列坐标转换为机械臂坐标系坐标。

    通过四点标定做双线性插值。

    Args:
        row: 棋盘行 (0-based)。
        col: 棋盘列 (0-based)。
        board_rows: 总行数 (15)。
        board_cols: 总列数 (15)。
        calib: 四点标定数据。
        z_height: 机械臂执行高度。

    Returns:
        (x, y, z) 机械臂坐标系坐标。
    """
    # 归一化坐标 (0~1)
    u = col / (board_cols - 1) if board_cols > 1 else 0.5
    v = row / (board_rows - 1) if board_rows > 1 else 0.5

    # 双线性插值
    def lerp(a, b, t):
        return a + (b - a) * t

    top_x = lerp(calib.top_left[0], calib.top_right[0], u)
    top_y = lerp(calib.top_left[1], calib.top_right[1], u)
    bot_x = lerp(calib.bottom_left[0], calib.bottom_right[0], u)
    bot_y = lerp(calib.bottom_left[1], calib.bottom_right[1], u)

    x = lerp(top_x, bot_x, v)
    y = lerp(top_y, bot_y, v)

    return (x, y, z_height)
