"""全局常量."""

import numpy as np

# 棋盘规格
BOARD_ROWS = 15
BOARD_COLS = 15

# 棋子颜色编码
EMPTY = 0
BLACK = 1
WHITE = 2

# 标准棋盘状态 shape
BOARD_SHAPE = (BOARD_ROWS, BOARD_COLS)

# 空棋盘（用于初始化）
EMPTY_BOARD: np.ndarray = np.zeros(BOARD_SHAPE, dtype=np.int8)

# 四点标定角点顺序：左上、右上、右下、左下
CALIB_CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")

# 棋盘检测相关
DEFAULT_BOARD_COLOR_LOW = np.array([0, 0, 0])  # 棋盘底色 HSV 下界
DEFAULT_BOARD_COLOR_HIGH = np.array([30, 80, 200])  # 棋盘底色 HSV 上界
