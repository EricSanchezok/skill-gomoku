"""主流程编排器 — 相机→分析→AI→机械臂循环."""

import logging
from collections.abc import Mapping
from typing import Any

from src.game.ai import ai_decide, ai_reset
from src.game.board import Board, check_win
from src.perception.state_extractor import StateExtractor
from src.robot.calibration import (
    ManualPoseSampler,
    get_robot_z_height,
    load_robot_calibration,
    run_manual_robot_calibration,
    save_robot_calibration,
)
from src.robot.controller import CalibrationPoints, RobotPose, board_to_robot_pose
from src.robot.pose_mapper import BoardPoseMapper, RobotPoseMover, load_pose_mapper_from_config
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
        calib: CalibrationPoints | None = None,
        z_height: float | None = None,
        pose_mapper: BoardPoseMapper | None = None,
        robot_mover: RobotPoseMover | None = None,
        my_stone: int = BLACK,
    ):
        self.extractor = state_extractor
        self.calib = calib
        self.z_height = z_height
        self.pose_mapper = pose_mapper
        self.robot_mover = robot_mover
        self.my_stone = my_stone
        self.opponent = WHITE if my_stone == BLACK else BLACK
        self.board = Board()
        self.move_count = 0

    @classmethod
    def from_config(
        cls,
        state_extractor: StateExtractor,
        config: dict[str, Any],
    ) -> "GameOrchestrator":
        """Create an orchestrator from config, including saved robot calibration."""
        robot_cfg = config.get("robot", {})
        pose_mapper = load_pose_mapper_from_config(config)

        if pose_mapper is not None:
            calib = None
            logger.info(
                "Loaded measured robot pose map: %dx%d, coordinate_space=%s",
                pose_mapper.board_rows,
                pose_mapper.board_cols,
                pose_mapper.coordinate_space,
            )
        else:
            try:
                calib = load_robot_calibration(config)
            except ValueError:
                if not robot_cfg.get("calibrate_before_game", False):
                    raise
                calib = None

        return cls(
            state_extractor=state_extractor,
            calib=calib,
            z_height=get_robot_z_height(config),
            pose_mapper=pose_mapper,
            my_stone=_parse_stone(config.get("game", {}).get("my_stone", "black")),
        )

    def calibrate_robot_before_game(
        self,
        sampler: ManualPoseSampler,
        config_path: str | None = None,
        coordinate_space: str | None = None,
        hold_after: bool = False,
    ) -> CalibrationPoints:
        """Hand-guide the arm to four board corners and update this game session."""
        calib = run_manual_robot_calibration(sampler, hold_after=hold_after)
        self.calib = calib

        if config_path is not None:
            save_robot_calibration(
                config_path,
                calib,
                coordinate_space or getattr(sampler, "coordinate_space", "robot_pose"),
                self.z_height,
            )

        return calib

    def start_new_game(
        self,
        sampler: ManualPoseSampler | None = None,
        calibrate_robot: bool = False,
        config_path: str | None = None,
        coordinate_space: str | None = None,
        hold_after: bool = False,
    ) -> Board:
        """Reset game state and optionally run manual robot calibration first."""
        if calibrate_robot:
            if sampler is None:
                raise ValueError("sampler is required when calibrate_robot=True")
            self.calibrate_robot_before_game(
                sampler=sampler,
                config_path=config_path,
                coordinate_space=coordinate_space,
                hold_after=hold_after,
            )

        self.extractor.reset()
        ai_reset()
        self.board = self.get_board_state()
        self.move_count = 0
        return self.board

    def get_board_state(self) -> Board:
        """拍照并获取当前棋盘状态。"""
        board_matrix, _ = self.extractor.extract()
        return Board(board_matrix)

    def wait_for_opponent(self) -> Board | None:
        """等待对方落子，返回更新后的 Board；超时或错误返回 None。"""
        # 简化版：拍照两次，对比差异找到新棋子
        import time

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

    def execute_my_move(self, row: int, col: int) -> RobotPose:
        """计算机械臂目标坐标并执行落子。

        Returns:
            机械臂控制器使用的目标姿态。
        """
        target = self._target_for_cell(row, col)
        if self.robot_mover is not None:
            if not isinstance(target, Mapping):
                raise TypeError("robot_mover integration requires a mapping robot target")
            self.robot_mover.move_to(target)
            # TODO: 触发末端落子机构 / gripper sequence
        self.board.place(row, col, self.my_stone)
        self.move_count += 1
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
        target = self._target_for_cell(row, col)
        logger.info(f"AI decides: ({row}, {col}), robot target: {target}")

        self.board.place(row, col, self.my_stone)
        self.move_count += 1
        winner = check_win(self.board.state, (row, col))
        return winner

    def _target_for_cell(self, row: int, col: int) -> RobotPose:
        if self.pose_mapper is not None:
            return self.pose_mapper.target_for_cell(row, col)

        if self.calib is None:
            raise RuntimeError(
                "Robot target mapping is missing. Configure robot.pose_map or run "
                "calibrate_robot_before_game() for the legacy interpolation path."
            )
        return board_to_robot_pose(
            row, col, self.board.rows, self.board.cols, self.calib, self.z_height
        )


def _parse_stone(value: Any) -> int:
    if isinstance(value, str):
        normalized = value.lower()
        if normalized == "black":
            return BLACK
        if normalized == "white":
            return WHITE
    if value in (BLACK, WHITE):
        return int(value)
    raise ValueError(f"Unknown stone value: {value!r}")
