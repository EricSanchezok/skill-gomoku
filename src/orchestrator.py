"""主流程编排器 — 相机→分析→AI→机械臂循环."""

import logging
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.game.ai import PROJECT_ROOT, ai_decide_verbose, ai_reset, configure_ai_from_config
from src.game.board import Board, check_win
from src.game.decision import AIDecision, AIMoveError, AIRefusedMoveError
from src.game.play_area import PlayArea, parse_play_area_config
from src.interaction import (
    HumanTurnCommand,
    HumanTurnController,
    NullRobotInteraction,
    RobotInteractionController,
    stone_name,
)
from src.perception.board_calibration import board_frame_required, load_board_frame_calibration
from src.perception.state_extractor import StateExtractor
from src.robot.air_pump import SuctionController, create_suction_controller_from_config
from src.robot.calibration import (
    ManualPoseSampler,
    get_robot_z_height,
    load_robot_calibration,
    run_manual_robot_calibration,
    save_robot_calibration,
)
from src.robot.controller import CalibrationPoints, RobotPose, board_to_robot_pose
from src.robot.pose_mapper import BoardPoseMapper, RobotPoseMover, load_pose_mapper_from_config
from src.robot.so101_mover import PRESET_ACTIONS
from src.utils.constants import BLACK, EMPTY, WHITE

logger = logging.getLogger(__name__)
SKILL_AUDIO_PATH = PROJECT_ROOT / "docs" / "MP3" / "1.mp3"


@dataclass(frozen=True)
class RobotMoveTargets:
    """Resolved robot poses used by one physical placement sequence."""

    pickup_pose: dict[str, float] | None = None
    pickup_poses: dict[int, dict[str, float]] | None = None
    pickup_top_pose: dict[str, float] | None = None
    pickup_top_poses: dict[int, dict[str, float]] | None = None
    waiting_pose: dict[str, float] | None = None


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
        suction_controller: SuctionController | None = None,
        pickup_pose: Mapping[str, float] | None = None,
        pickup_poses: Mapping[int, Mapping[str, float]] | None = None,
        pickup_top_pose: Mapping[str, float] | None = None,
        pickup_top_poses: Mapping[int, Mapping[str, float]] | None = None,
        waiting_pose: Mapping[str, float] | None = None,
        play_area: PlayArea | None = None,
        human_turn_controller: HumanTurnController | None = None,
        interaction_controller: RobotInteractionController | None = None,
        trash_talk_enabled: bool = False,
        my_stone: int = BLACK,
    ):
        self.extractor = state_extractor
        self.calib = calib
        self.z_height = z_height
        self.pose_mapper = pose_mapper
        self.robot_mover = robot_mover
        self.suction_controller = suction_controller
        self.move_targets = RobotMoveTargets(
            pickup_pose=dict(pickup_pose) if pickup_pose is not None else None,
            pickup_poses=(
                {int(stone): dict(pose) for stone, pose in pickup_poses.items()}
                if pickup_poses is not None
                else None
            ),
            pickup_top_pose=dict(pickup_top_pose) if pickup_top_pose is not None else None,
            pickup_top_poses=(
                {int(stone): dict(pose) for stone, pose in pickup_top_poses.items()}
                if pickup_top_poses is not None
                else None
            ),
            waiting_pose=dict(waiting_pose) if waiting_pose is not None else None,
        )
        self.pickup_pose = self.move_targets.pickup_pose
        self.pickup_poses = self.move_targets.pickup_poses or {}
        self.pickup_top_pose = self.move_targets.pickup_top_pose
        self.pickup_top_poses = self.move_targets.pickup_top_poses or {}
        self.waiting_pose = self.move_targets.waiting_pose
        self.play_area = play_area or PlayArea.full()
        self.robot_stone = my_stone
        self.my_stone = self.robot_stone
        self.human_stone = WHITE if self.robot_stone == BLACK else BLACK
        self.opponent = self.human_stone
        self.human_turn_controller = human_turn_controller
        self.interaction_controller = interaction_controller or NullRobotInteraction()
        self.trash_talk_enabled = bool(trash_talk_enabled)
        self.board = Board()
        self.move_count = 0
        self._next_turn_override: int | None = None

    @classmethod
    def from_config(
        cls,
        state_extractor: StateExtractor,
        config: dict[str, Any],
        *,
        config_base_dir: str | Path = PROJECT_ROOT,
        robot_mover: RobotPoseMover | None = None,
        human_turn_controller: HumanTurnController | None = None,
        interaction_controller: RobotInteractionController | None = None,
    ) -> "GameOrchestrator":
        """Create an orchestrator from config, including saved robot calibration."""
        robot_cfg = config.get("robot", {})
        game_cfg = config.get("game", {})
        if board_frame_required(config):
            load_board_frame_calibration(config, required=True)

        configure_ai_from_config(config, base_dir=config_base_dir)
        pose_mapper = load_pose_mapper_from_config(config, base_dir=config_base_dir)

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
            robot_mover=robot_mover,
            suction_controller=create_suction_controller_from_config(config),
            pickup_pose=_parse_optional_robot_pose(
                robot_cfg.get("pickup_pose"),
                "robot.pickup_pose",
            ),
            pickup_poses=_parse_pickup_poses(robot_cfg.get("pickup_poses")),
            pickup_top_pose=_parse_optional_robot_pose(
                robot_cfg.get("pickup_top_pose"),
                "robot.pickup_top_pose",
            ),
            pickup_top_poses=_parse_stone_poses(
                robot_cfg.get("pickup_top_poses"),
                "robot.pickup_top_poses",
            ),
            waiting_pose=_parse_optional_robot_pose(
                robot_cfg.get("waiting_pose", "waiting"),
                "robot.waiting_pose",
            ),
            play_area=parse_play_area_config(
                game_cfg.get("play_area", game_cfg.get("active_area"))
            ),
            human_turn_controller=human_turn_controller,
            interaction_controller=interaction_controller,
            trash_talk_enabled=bool(game_cfg.get("trash_talk_enabled", False)),
            my_stone=_parse_robot_stone_from_game_config(game_cfg),
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

        self.move_to_waiting_pose()
        self.extractor.reset()
        ai_reset()
        self.board = self.get_board_state()
        self.move_count = sum(_stone_counts(self.board.state))
        self._next_turn_override = None
        return self.board

    @property
    def robot_moves_first(self) -> bool:
        """Return whether the robot is black and therefore plays first."""

        return self.robot_stone == BLACK

    @property
    def human_moves_first(self) -> bool:
        """Return whether the human is black and therefore plays first."""

        return self.human_stone == BLACK

    def next_turn_stone(self) -> int:
        """Return the stone color that should move next under normal Gomoku order."""

        if self._next_turn_override is not None:
            return self._next_turn_override

        black_count, white_count = _stone_counts(self.board.state)
        if black_count == white_count:
            return BLACK
        if black_count == white_count + 1:
            return WHITE
        raise RuntimeError(
            "Illegal board turn balance: "
            f"black={black_count}, white={white_count}. Check perception before continuing."
        )

    def is_robot_turn(self) -> bool:
        """Return whether the next legal move belongs to the robot."""

        return self.next_turn_stone() == self.robot_stone

    def is_human_turn(self) -> bool:
        """Return whether the next legal move belongs to the human."""

        return self.next_turn_stone() == self.human_stone

    def play_robot_turn(self) -> tuple[int, int, RobotPose]:
        """Ask AI for the next move and execute the robot placement sequence."""

        if not self.is_robot_turn():
            raise RuntimeError(
                f"Robot plays {stone_name(self.robot_stone)}, but next turn is "
                f"{stone_name(self.next_turn_stone())}"
            )
        decision = ai_decide_verbose(self.play_area.crop(self.board.state), self.robot_stone)
        if not decision.should_play:
            raise AIRefusedMoveError("AI chose not to play")
        self._handle_ai_decision(decision)
        local_row, local_col = decision.row, decision.col
        row, col = self.play_area.to_global(local_row, local_col)
        target = self.execute_my_move(row, col)
        self._speak_after_robot_move(decision)
        return row, col, target

    def get_board_state(self) -> Board:
        """拍照并获取当前棋盘状态。"""
        board_matrix, _ = self.extractor.extract()
        board_matrix = self.play_area.filter_board_state(board_matrix)
        return Board(board_matrix)

    def sync_board_state(self) -> Board:
        """Capture the current physical board and adopt it as game state."""

        self.move_to_waiting_pose()
        camera_board = self.get_board_state()
        current_count = sum(_stone_counts(self.board.state))
        camera_count = sum(_stone_counts(camera_board.state))
        if camera_count < current_count:
            logger.warning(
                "Skipping camera sync because detected stone count regressed "
                "from %d to %d",
                current_count,
                camera_count,
            )
            return self.board
        self.board = camera_board
        self.move_count = camera_count
        return self.board

    def is_play_area_full(self) -> bool:
        """Return whether the configured playable window has no empty positions."""

        return self.play_area.is_full(self.board.state)

    def wait_for_opponent(
        self,
        *,
        confirm_human: bool = True,
        max_attempts: int = 30,
        poll_interval_seconds: float = 1.0,
    ) -> Board | None:
        """等待对方落子，返回更新后的 Board；超时或错误返回 None。"""
        import time

        self.move_to_waiting_pose()
        if self.human_turn_controller is not None and confirm_human:
            logger.info("Waiting for human move confirmation before perception")
            result = self.human_turn_controller.wait_for_move_done(
                expected_stone=self.human_stone,
                board_state=self.board.state.copy(),
            )
            if result.command == HumanTurnCommand.QUIT:
                return None
            logger.info("Human move confirmed; starting perception polling")

        for _ in range(max_attempts):
            time.sleep(poll_interval_seconds)
            try:
                new_matrix, _extractor_delta = self.extractor.extract()
                filtered_matrix = self.play_area.filter_board_state(new_matrix)
                delta = _find_added_stone(self.board.state, filtered_matrix)
                if delta is not None:
                    r, c, stone = delta
                    if r is None or c is None:
                        continue
                    if not self.play_area.contains(r, c):
                        logger.warning(
                            "Detected human move outside play area at (%d, %d); ignoring",
                            r,
                            c,
                        )
                        continue
                    if stone == self.human_stone:
                        self.board.place(r, c, stone)
                        self.move_count = sum(_stone_counts(self.board.state))
                        if self._next_turn_override is not None:
                            self._next_turn_override = self.robot_stone
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
            if self.waiting_pose is None:
                raise ValueError("waiting_pose is required before moving to a board target")
            stone_picked = False
            try:
                pickup_pose = self._pickup_pose_for_robot_stone()
                pickup_top_pose = self._pickup_top_pose_for_robot_stone()
                if pickup_pose is not None:
                    if pickup_top_pose is None:
                        raise ValueError("pickup_top_pose is required when pickup_pose is used")
                    self.robot_mover.move_to(pickup_top_pose)
                    self.robot_mover.move_to(pickup_pose)
                if self.suction_controller is not None:
                    self.suction_controller.pick_stone()
                    stone_picked = True
                if pickup_pose is not None and pickup_top_pose is not None:
                    self.robot_mover.move_to(pickup_top_pose)
                self.move_to_waiting_pose()
                self.robot_mover.move_to(target)
                if self.suction_controller is not None:
                    self.suction_controller.drop_stone()
                    stone_picked = False
                self.move_to_waiting_pose()
            except Exception:
                if self.suction_controller is not None and stone_picked:
                    self.suction_controller.off()
                raise
        self.board.place(row, col, self.my_stone)
        self.move_count += 1
        if self._next_turn_override is not None:
            self._next_turn_override = self.human_stone
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

        if not self.is_robot_turn():
            logger.info(
                "Skip AI move: robot plays %s, next turn is %s",
                stone_name(self.robot_stone),
                stone_name(self.next_turn_stone()),
            )
            return EMPTY

        decision = ai_decide_verbose(self.play_area.crop(self.board.state), self.robot_stone)
        if not decision.should_play:
            raise AIRefusedMoveError("AI chose not to play")
        self._handle_ai_decision(decision)
        local_row, local_col = decision.row, decision.col
        row, col = self.play_area.to_global(local_row, local_col)
        target = self._target_for_cell(row, col)
        logger.info(f"AI decides: ({row}, {col}), robot target: {target}")

        self.board.place(row, col, self.robot_stone)
        self.move_count += 1
        if self._next_turn_override is not None:
            self._next_turn_override = self.human_stone
        self._speak_after_robot_move(decision)
        winner = check_win(self.board.state, (row, col))
        return winner

    def robot_say(self, text: str) -> None:
        """Reserved HRI hook for robot speech."""

        self.interaction_controller.speak(text)

    def robot_dance(self, name: str = "default") -> None:
        """Reserved HRI hook for robot dance or motion routines."""

        self.interaction_controller.dance(name)

    def robot_use_skill_gomoku(
        self,
        context: Mapping[str, Any] | None = None,
        *,
        speech: str = "不装了，我要开挂了。",
    ) -> None:
        """Reserved HRI hook for invoking an external skill-gomoku action."""

        _play_audio_file(SKILL_AUDIO_PATH, wait=True)
        self.robot_say(speech)
        self.interaction_controller.use_skill_gomoku(context)

    def move_to_waiting_pose(self) -> RobotPose | None:
        """Move the robot to the configured camera-clear waiting pose."""

        if self.robot_mover is None or self.waiting_pose is None:
            return None
        return self.robot_mover.move_to(self.waiting_pose)

    def _handle_ai_decision(self, decision: AIDecision) -> None:
        if decision.use_skill:
            if not _has_immediate_winning_move(
                self.play_area.crop(self.board.state),
                self.human_stone,
            ):
                logger.warning("Ignoring AI skill request: opponent has no immediate win threat")
                return
            row, col = self._skill_target_for_decision(decision)
            self.robot_use_skill_gomoku(
                {
                    "source": decision.source,
                    "row": decision.row,
                    "col": decision.col,
                    "skill_row": row,
                    "skill_col": col,
                    "rationale": decision.rationale,
                }
            )
            self._execute_remove_opponent_stone_skill(row, col)

    def _speak_after_robot_move(self, decision: AIDecision) -> None:
        if not self.trash_talk_enabled:
            return
        self.robot_say(decision.trash_talk or "这手下完，我都替你紧张。")

    def _pickup_pose_for_robot_stone(self) -> dict[str, float] | None:
        """Return the pickup pose for the robot's configured stone colour."""

        return self.pickup_poses.get(self.robot_stone, self.pickup_pose)

    def _pickup_top_pose_for_robot_stone(self) -> dict[str, float] | None:
        """Return the safe pickup exit/entry pose for the configured stone colour."""

        return self.pickup_top_poses.get(self.robot_stone, self.pickup_top_pose)

    def _pickup_pose_for_stone(self, stone: int) -> dict[str, float] | None:
        return self.pickup_poses.get(stone, self.pickup_pose)

    def _pickup_top_pose_for_stone(self, stone: int) -> dict[str, float] | None:
        return self.pickup_top_poses.get(stone, self.pickup_top_pose)

    def _skill_target_for_decision(self, decision: AIDecision) -> tuple[int, int]:
        if decision.skill_row is not None and decision.skill_col is not None:
            row, col = self.play_area.to_global(decision.skill_row, decision.skill_col)
            if self.board.get(row, col) != self.human_stone:
                raise AIMoveError(f"Skill target ({row}, {col}) is not an opponent stone")
            return row, col

        target = _best_stone_to_remove(self.play_area.crop(self.board.state), self.human_stone)
        if target is None:
            raise AIMoveError("AI requested skill, but there is no opponent stone to remove")
        return self.play_area.to_global(*target)

    def _execute_remove_opponent_stone_skill(self, row: int, col: int) -> None:
        if self.board.get(row, col) != self.human_stone:
            raise AIMoveError(f"Skill target ({row}, {col}) is not an opponent stone")

        if self.robot_mover is not None:
            self._move_board_stone_to_pickup_box(row, col, self.human_stone)

        self.board.remove(row, col, self.human_stone)
        self.move_count = sum(_stone_counts(self.board.state))
        self._next_turn_override = self.robot_stone

    def _move_board_stone_to_pickup_box(self, row: int, col: int, stone: int) -> None:
        if self.suction_controller is None:
            raise ValueError("suction_controller is required for the remove-stone skill")
        if self.waiting_pose is None:
            raise ValueError("waiting_pose is required before moving to a board target")

        board_target = self._target_for_cell(row, col)
        if not isinstance(board_target, Mapping):
            raise TypeError("robot_mover integration requires a mapping robot target")

        drop_pose = self._pickup_pose_for_stone(stone)
        drop_top_pose = self._pickup_top_pose_for_stone(stone)
        if drop_pose is None or drop_top_pose is None:
            raise ValueError("pickup pose and pickup top pose are required for the skill stone")

        stone_picked = False
        try:
            self.move_to_waiting_pose()
            self.robot_mover.move_to(board_target)
            self.suction_controller.pick_stone()
            stone_picked = True
            self.move_to_waiting_pose()
            self.robot_mover.move_to(drop_top_pose)
            self.robot_mover.move_to(drop_pose)
            self.suction_controller.drop_stone()
            stone_picked = False
            self.robot_mover.move_to(drop_top_pose)
            self.move_to_waiting_pose()
        except Exception:
            if stone_picked:
                self.suction_controller.off()
            raise

    def _target_for_cell(self, row: int, col: int) -> RobotPose:
        if not self.play_area.contains(row, col):
            raise ValueError(
                f"Robot target ({row}, {col}) is outside play area "
                f"{self.play_area.describe(include_board=False)}"
            )
        if self.pose_mapper is not None:
            if (
                self.pose_mapper.board_rows == self.board.rows
                and self.pose_mapper.board_cols == self.board.cols
            ):
                return self.pose_mapper.target_for_cell(row, col)
            local_row, local_col = self.play_area.to_local(row, col)
            return self.pose_mapper.target_for_cell(local_row, local_col)

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


def _parse_robot_stone_from_game_config(game_cfg: Mapping[str, Any]) -> int:
    robot_value = game_cfg.get("robot_stone", game_cfg.get("my_stone", "black"))
    robot_stone = _parse_stone(robot_value)
    if "robot_stone" in game_cfg and "my_stone" in game_cfg:
        legacy_stone = _parse_stone(game_cfg["my_stone"])
        if legacy_stone != robot_stone:
            raise ValueError("game.robot_stone and legacy game.my_stone disagree")
    return robot_stone


def _parse_optional_robot_pose(value: Any, field_name: str) -> dict[str, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _pose_from_preset(value, field_name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a preset name or mapping of LeRobot action keys")
    if "preset" in value:
        return _pose_from_preset(str(value["preset"]), field_name)
    if not value:
        raise ValueError(f"{field_name} cannot be empty when configured")
    return {str(key): float(item) for key, item in value.items()}


def _parse_pickup_poses(value: Any) -> dict[int, dict[str, float]] | None:
    return _parse_stone_poses(value, "robot.pickup_poses")


def _parse_stone_poses(value: Any, field_name: str) -> dict[int, dict[str, float]] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping with black/white keys")

    parsed: dict[int, dict[str, float]] = {}
    for stone_key, pose_value in value.items():
        if pose_value is None:
            continue
        stone = _parse_stone(stone_key)
        pose = _parse_optional_robot_pose(
            pose_value,
            f"{field_name}.{stone_key}",
        )
        if pose is None:
            continue
        parsed[stone] = pose
    return parsed or None


def _pose_from_preset(name: str, field_name: str) -> dict[str, float]:
    try:
        pose = PRESET_ACTIONS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown {field_name} preset {name!r}; available presets: {sorted(PRESET_ACTIONS)}"
        ) from exc
    return dict(pose)


def _stone_counts(board_matrix: np.ndarray) -> tuple[int, int]:
    return (
        int(np.count_nonzero(board_matrix == BLACK)),
        int(np.count_nonzero(board_matrix == WHITE)),
    )


def _best_stone_to_remove(board_matrix: np.ndarray, stone: int) -> tuple[int, int] | None:
    rows, cols = np.where(board_matrix == stone)
    if rows.size == 0:
        return None
    center_r = (board_matrix.shape[0] - 1) / 2
    center_c = (board_matrix.shape[1] - 1) / 2

    def score(row: int, col: int) -> tuple[int, float]:
        best_line = 1
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            line = 1
            for sign in (1, -1):
                r, c = row, col
                while True:
                    r += sign * dr
                    c += sign * dc
                    if not (
                        0 <= r < board_matrix.shape[0]
                        and 0 <= c < board_matrix.shape[1]
                        and board_matrix[r, c] == stone
                    ):
                        break
                    line += 1
            best_line = max(best_line, line)
        center_distance = abs(row - center_r) + abs(col - center_c)
        return best_line, -center_distance

    candidates = [(int(row), int(col)) for row, col in zip(rows, cols, strict=True)]
    return max(candidates, key=lambda item: score(*item))


def _has_immediate_winning_move(board_matrix: np.ndarray, stone: int) -> bool:
    for row, col in zip(*np.where(board_matrix == EMPTY), strict=True):
        candidate = board_matrix.copy()
        candidate[row, col] = stone
        if check_win(candidate, (int(row), int(col))) == stone:
            return True
    return False


def _play_audio_file(path: Path, *, wait: bool = False) -> None:
    command = _audio_command(path)
    if command is None:
        logger.warning("No audio player found for %s", path)
        return
    try:
        if wait:
            subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as exc:
        logger.warning("Failed to play audio %s: %s", path, exc)


def _audio_command(path: Path) -> list[str] | None:
    if not path.is_file():
        return None
    players = (
        ("afplay", [str(path)]),
        ("mpg123", ["-q", str(path)]),
        ("mpg321", ["-q", str(path)]),
        ("mpv", ["--no-video", "--really-quiet", str(path)]),
        ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]),
        ("cvlc", ["--play-and-exit", "--quiet", str(path)]),
    )
    for executable, args in players:
        resolved = shutil.which(executable)
        if resolved is not None:
            return [resolved, *args]
    return None


def _find_added_stone(
    previous: np.ndarray,
    current: np.ndarray,
) -> tuple[int, int, int] | None:
    added_rows, added_cols = np.where((previous == EMPTY) & (current != EMPTY))
    if added_rows.size != 1:
        return None
    row = int(added_rows[0])
    col = int(added_cols[0])
    return row, col, int(current[row, col])
