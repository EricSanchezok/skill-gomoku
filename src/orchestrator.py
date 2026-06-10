"""主流程编排器 — 相机→分析→AI→机械臂循环."""

import logging

from src.game.ai import ai_decide
from src.game.board import Board, check_win
from src.perception.state_extractor import StateExtractor
from src.robot.controller import CalibrationPoints, board_to_robot_coords
from src.utils.constants import BLACK, EMPTY, WHITE

logger = logging.getLogger(__name__)


class GameOrchestrator:
    """五子棋对局编排器。

    流程：
    1. 我方回合：拍照 → 分析棋盘 → AI 决策 → 机械臂落子
    2. 对方回合：拍照 → 检测对方落子 → 判定胜负 → 轮到我方
    """

    def __init__(
        self,
        state_extractor: StateExtractor,
        calib: CalibrationPoints,
        z_height: float,
        my_stone: int = BLACK,
    ):
        self.extractor = state_extractor
        self.calib = calib
        self.z_height = z_height
        self.my_stone = my_stone
        self.opponent = WHITE if my_stone == BLACK else BLACK
        self.board = Board()
        self.move_count = 0

    def get_board_state(self) -> Board:
        """拍照并获取当前棋盘状态。"""
        board_matrix, _ = self.extractor.extract()
        return Board(board_matrix)

    def wait_for_opponent(self) -> Board | None:
        """等待对方落子，返回更新后的 Board；超时或错误返回 None。"""
        # 简化版：拍照两次，对比差异找到新棋子
        import time

        prev_state = self.board.state.copy()
        max_attempts = 30
        for _ in range(max_attempts):
            time.sleep(1.0)
            try:
                new_matrix, delta = self.extractor.extract()
                if delta is not None:
                    r, c, stone = delta
                    if r is not None and c is not None and stone == self.opponent:
                        self.board.place(r, c, stone)
                        return self.board
            except Exception as e:
                logger.warning(f"Wait for opponent failed: {e}")
        return None

    def execute_my_move(self, row: int, col: int) -> tuple[float, float, float]:
        """计算机械臂目标坐标并执行落子。

        Returns:
            机械臂坐标系的目标坐标。
        """
        target = board_to_robot_coords(
            row, col, self.board.rows, self.board.cols, self.calib, self.z_height
        )
        # TODO: 调用机械臂控制接口
        # robot_controller.move_to(*target)
        # robot_controller.gripper.activate()
        self.board.place(row, col, self.my_stone)
        return target

    def run_once(self) -> int:
        """执行一回合（AI 判断落子 + 输出坐标），不实际操控机械臂。

        Returns:
            获胜方 (BLACK/WHITE)，0 表示继续。
        """
        board_matrix, _ = self.extractor.extract()
        self.board = Board(board_matrix)

        winner = check_win(self.board.state)
        if winner != EMPTY:
            return winner

        row, col = ai_decide(self.board.state, self.my_stone)
        target = board_to_robot_coords(
            row, col, self.board.rows, self.board.cols, self.calib, self.z_height
        )
        logger.info(f"AI decides: ({row}, {col}), robot target: {target}")

        self.board.place(row, col, self.my_stone)
        winner = check_win(self.board.state, (row, col))
        return winner
