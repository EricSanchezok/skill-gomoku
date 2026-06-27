#!/usr/bin/env python3
"""Run a live Gomoku game with camera, Rapfi, SO101, and suction control."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.game.ai import ai_reset, resolve_rapfi_engine_path  # noqa: E402
from src.game.board import check_win  # noqa: E402
from src.interaction import (  # noqa: E402
    ConsoleRobotInteraction,
    KeyboardHumanTurnController,
    stone_name,
)
from src.orchestrator import GameOrchestrator  # noqa: E402
from src.perception.camera import create_camera  # noqa: E402
from src.perception.state_extractor import StateExtractor  # noqa: E402
from src.robot.so101_mover import (  # noqa: E402
    DEFAULT_MAX_RELATIVE_TARGET,
    DEFAULT_ROBOT_ID,
    MotionProfile,
    SO101SmoothMover,
    suppress_lerobot_clamp_warnings,
)
from src.utils.config_loader import load_config  # noqa: E402
from src.utils.constants import BOARD_COLS, BOARD_ROWS, EMPTY  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("run_live_game")


class ConfirmingRobotMover:
    """Prompt before every real SO101 move in live-game bring-up runs."""

    def __init__(
        self,
        delegate: SO101SmoothMover,
        *,
        verbose_target: bool = False,
        input_fn=input,
        print_fn=print,
    ) -> None:
        self._delegate = delegate
        self._verbose_target = verbose_target
        self._input = input_fn
        self._print = print_fn

    def move_to(self, target_pose: Mapping[str, float]) -> Any:
        target = {str(key): float(value) for key, value in target_pose.items()}
        self._print("\nNext SO101 move:")
        if self._verbose_target:
            self._print(_format_action(target))
        else:
            self._print(f"  target joints: {len(target)}")
        try:
            current = self._delegate.read_action(target)
            self._print(_format_action_delta(current, target))
        except Exception as exc:
            self._print(f"  current/delta: unavailable ({exc})")

        answer = self._input("Press Enter to execute this move, or type q to abort > ")
        if answer.strip().lower() == "q":
            raise KeyboardInterrupt
        return self._delegate.move_to(target)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    suppress_lerobot_clamp_warnings(enabled=not args.show_clamp_warnings)

    config_path = _project_path(args.config)
    config = load_config(config_path)
    _apply_cli_overrides(config, args)
    _validate_engine_path(config)
    _validate_live_config_safety(config, args)

    interaction = ConsoleRobotInteraction()
    human_controller = KeyboardHumanTurnController(robot_interaction=interaction)
    mover = None
    robot_mover = None
    orchestrator = None

    try:
        camera = create_camera(config, mock=args.mock_camera)
        extractor = StateExtractor(camera=camera, config=config)

        if not args.dry_run_robot:
            mover = _create_mover(config, args)
            mover.connect()
            print("Holding current SO101 pose before starting...")
            mover.hold_current()
            robot_mover = (
                ConfirmingRobotMover(mover, verbose_target=args.verbose)
                if args.confirm_robot_moves
                else mover
            )

        orchestrator = GameOrchestrator.from_config(
            state_extractor=extractor,
            config=config,
            config_base_dir=PROJECT_ROOT,
            robot_mover=robot_mover,
            human_turn_controller=human_controller,
            interaction_controller=interaction,
        )
        _validate_robot_mapping(orchestrator, dry_run_robot=args.dry_run_robot)
        _validate_live_robot_safety(orchestrator, args)
        max_turns = _resolve_max_turns(args, orchestrator)

        _confirm_start(args, config, orchestrator, max_turns=max_turns)

        board = orchestrator.start_new_game()
        print("Initial board:")
        print(board)
        winner = check_win(orchestrator.board.state)

        turn_idx = 0
        while (
            winner == EMPTY
            and turn_idx < max_turns
            and not orchestrator.is_play_area_full()
        ):
            turn_idx += 1
            print()
            print(f"Turn {turn_idx}: next = {stone_name(orchestrator.next_turn_stone())}")

            if orchestrator.is_robot_turn():
                interaction.speak("轮到我了。")
                row, col, target = orchestrator.play_robot_turn()
                print(f"Robot played {stone_name(orchestrator.robot_stone)} at ({row}, {col})")
                print(f"Robot target: {target}")
                winner = check_win(orchestrator.board.state, (row, col))

                if winner == EMPTY and not args.dry_run_robot and args.sync_after_robot:
                    time.sleep(args.robot_settle_seconds)
                    print("Syncing camera state after robot move...")
                    orchestrator.sync_board_state()
                    winner = check_win(orchestrator.board.state)
            else:
                interaction.speak("请你下棋，下完后按确认键。")
                board = orchestrator.wait_for_opponent(
                    max_attempts=args.human_attempts,
                    poll_interval_seconds=args.poll_interval,
                )
                if board is None:
                    print("Human turn ended without a detected move.")
                    break
                print("Human move detected.")
                winner = check_win(orchestrator.board.state)

            print(orchestrator.board)

        if winner != EMPTY:
            print(f"Game over: {stone_name(winner)} wins.")
            if winner == orchestrator.robot_stone:
                interaction.dance("win")
            return 0

        if orchestrator.is_play_area_full():
            print("Game over: playable area is full.")
            return 0

        print("Game stopped before a winner was detected.")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Returning hardware to a safe state...")
        return 130
    finally:
        ai_reset()
        if orchestrator is not None and orchestrator.suction_controller is not None:
            try:
                orchestrator.suction_controller.off()
                close = getattr(orchestrator.suction_controller, "close", None)
                if callable(close):
                    close()
            except Exception as exc:
                logger.warning("Failed to close suction controller: %s", exc)
        if mover is not None:
            try:
                if args.release_on_exit:
                    mover.release()
                mover.disconnect()
            except Exception as exc:
                logger.warning("Failed to disconnect SO101 mover: %s", exc)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML path")
    parser.add_argument("--mock-camera", action="store_true", help="Use MockCamera")
    parser.add_argument("--dry-run-robot", action="store_true", help="Do not connect or move SO101")
    parser.add_argument("--dry-run-air-pump", action="store_true", help="Use fake GPIO servos")
    parser.add_argument("--enable-air-pump", action="store_true", help="Force-enable suction")
    parser.add_argument("--disable-air-pump", action="store_true", help="Force-disable suction")
    parser.add_argument("--port", default=None, help="Override robot.port")
    parser.add_argument("--robot-id", default=None, help="Override robot.id")
    parser.add_argument("--robot-stone", choices=("black", "white"), default=None)
    parser.add_argument("--engine-path", default=None, help="Override game.ai.engine_path")
    parser.add_argument("--time-per-move-ms", type=int, default=None)
    parser.add_argument("--duration", type=float, default=5.0, help="SO101 move duration")
    parser.add_argument("--dt", type=float, default=0.01, help="SO101 move command interval")
    parser.add_argument("--max-relative-target", type=float, default=DEFAULT_MAX_RELATIVE_TARGET)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum live loop turns. Default is 1 for safer hardware bring-up.",
    )
    parser.add_argument(
        "--full-game",
        action="store_true",
        help="Run until the configured play area is full instead of the one-turn safety default.",
    )
    parser.add_argument("--human-attempts", type=int, default=30)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--robot-settle-seconds", type=float, default=0.5)
    parser.add_argument("--no-sync-after-robot", dest="sync_after_robot", action="store_false")
    parser.add_argument(
        "--no-confirm-robot-moves",
        dest="confirm_robot_moves",
        action="store_false",
        help="Do not prompt before each real SO101 movement.",
    )
    parser.add_argument(
        "--allow-missing-pickup-pose",
        action="store_true",
        help="Allow live robot motion when the configured robot stone has no pickup pose.",
    )
    parser.add_argument(
        "--show-clamp-warnings",
        action="store_true",
        help="Show LeRobot max-relative-target clamp warnings.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip final start confirmation")
    parser.add_argument("--release-on-exit", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(sync_after_robot=True, confirm_robot_moves=True)
    return parser


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    robot_cfg = _ensure_mapping(config, "robot")
    game_cfg = _ensure_mapping(config, "game")
    ai_cfg = _ensure_mapping(game_cfg, "ai")

    if args.port is not None:
        robot_cfg["port"] = args.port
    if args.robot_id is not None:
        robot_cfg["id"] = args.robot_id
    if args.robot_stone is not None:
        game_cfg["robot_stone"] = args.robot_stone
        game_cfg["my_stone"] = args.robot_stone
    if args.engine_path is not None:
        ai_cfg["engine_path"] = args.engine_path
    if args.time_per_move_ms is not None:
        ai_cfg["time_per_move_ms"] = int(args.time_per_move_ms)

    pump_cfg = _ensure_mapping(robot_cfg, "air_pump")
    if args.enable_air_pump:
        pump_cfg["enabled"] = True
    if args.disable_air_pump:
        pump_cfg["enabled"] = False
    if args.dry_run_air_pump:
        pump_cfg["dry_run"] = True


def _create_mover(config: Mapping[str, Any], args: argparse.Namespace) -> SO101SmoothMover:
    robot_cfg = config.get("robot", {})
    if not isinstance(robot_cfg, Mapping):
        robot_cfg = {}

    port = str(robot_cfg.get("port", ""))
    if not port:
        raise ValueError("robot.port is required unless --dry-run-robot is set")
    robot_id = str(robot_cfg.get("id", DEFAULT_ROBOT_ID))
    profile = MotionProfile(
        duration_seconds=args.duration,
        dt_seconds=args.dt,
        max_relative_target=args.max_relative_target,
    )
    return SO101SmoothMover(port=port, robot_id=robot_id, profile=profile)


def _validate_engine_path(config: Mapping[str, Any]) -> None:
    game_cfg = config.get("game", {})
    if not isinstance(game_cfg, Mapping):
        game_cfg = {}
    ai_cfg = game_cfg.get("ai", {})
    if not isinstance(ai_cfg, Mapping):
        ai_cfg = {}

    engine_value = ai_cfg.get("engine_path")
    if engine_value in (None, ""):
        engine_path = resolve_rapfi_engine_path()
    else:
        engine_path = Path(str(engine_value)).expanduser()
        if not engine_path.is_absolute():
            engine_path = PROJECT_ROOT / engine_path

    if not engine_path.exists():
        raise FileNotFoundError(
            f"Rapfi executable not found: {engine_path}. "
            "Put the Pi build at bin/rapfi/linux-aarch64/rapfi or set game.ai.engine_path."
        )
    if not engine_path.is_file():
        raise FileNotFoundError(f"Rapfi path is not a file: {engine_path}")


def _validate_robot_mapping(orchestrator: GameOrchestrator, *, dry_run_robot: bool) -> None:
    if dry_run_robot or orchestrator.pose_mapper is None:
        return
    play_area = orchestrator.play_area
    if (
        orchestrator.pose_mapper.board_rows == BOARD_ROWS
        and orchestrator.pose_mapper.board_cols == BOARD_COLS
    ):
        return
    if (
        orchestrator.pose_mapper.board_rows == play_area.rows
        and orchestrator.pose_mapper.board_cols == play_area.cols
    ):
        return
    raise ValueError(
        "Measured robot pose map does not match the configured play area: "
        f"pose_map={orchestrator.pose_mapper.board_rows}x{orchestrator.pose_mapper.board_cols}, "
        f"play_area={play_area.rows}x{play_area.cols}, board={BOARD_ROWS}x{BOARD_COLS}. "
        "Use a pose map for the active robot play window, or run with --dry-run-robot "
        "for perception/AI testing."
    )


def _validate_live_robot_safety(
    orchestrator: GameOrchestrator,
    args: argparse.Namespace,
) -> None:
    if args.dry_run_robot:
        return

    if orchestrator.waiting_pose is None:
        raise ValueError(
            "robot.waiting_pose is required for live robot runs. "
            "It keeps the arm out of the camera and routes motion through a known pose."
        )

    pickup_pose = orchestrator.pickup_poses.get(orchestrator.robot_stone, orchestrator.pickup_pose)
    if pickup_pose is None and not args.allow_missing_pickup_pose:
        raise ValueError(
            "Missing pickup pose for the robot stone. Record it first with "
            "`python scripts/record_pickup_poses.py`, or pass "
            "--allow-missing-pickup-pose only for deliberate dry movement tests."
        )


def _validate_live_config_safety(
    config: Mapping[str, Any],
    args: argparse.Namespace,
) -> None:
    if args.dry_run_robot:
        return

    robot_cfg = config.get("robot", {})
    if not isinstance(robot_cfg, Mapping):
        robot_cfg = {}
    game_cfg = config.get("game", {})
    if not isinstance(game_cfg, Mapping):
        game_cfg = {}

    if robot_cfg.get("waiting_pose", "waiting") is None:
        raise ValueError(
            "robot.waiting_pose is required for live robot runs before hardware startup."
        )

    robot_stone = str(game_cfg.get("robot_stone", game_cfg.get("my_stone", "black"))).lower()
    pickup_pose = None
    pickup_poses = robot_cfg.get("pickup_poses")
    if isinstance(pickup_poses, Mapping):
        pickup_pose = pickup_poses.get(robot_stone)
    pickup_pose = pickup_pose or robot_cfg.get("pickup_pose")
    if pickup_pose is None and not args.allow_missing_pickup_pose:
        raise ValueError(
            "Missing pickup pose for the configured robot stone before hardware startup. "
            "Run `python scripts/record_pickup_poses.py` first."
        )


def _resolve_max_turns(args: argparse.Namespace, orchestrator: GameOrchestrator) -> int:
    if args.max_turns is not None:
        if args.max_turns <= 0:
            raise ValueError("--max-turns must be positive")
        return args.max_turns
    if args.full_game:
        return orchestrator.play_area.rows * orchestrator.play_area.cols
    return 1


def _confirm_start(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    orchestrator: GameOrchestrator,
    *,
    max_turns: int,
) -> None:
    if args.yes:
        return

    game_cfg = config.get("game", {})
    robot_cfg = config.get("robot", {})
    ai_cfg = game_cfg.get("ai", {}) if isinstance(game_cfg, Mapping) else {}
    robot_stone = game_cfg.get("robot_stone", game_cfg.get("my_stone", "black"))
    pump_enabled = False
    if isinstance(robot_cfg, Mapping):
        pump_cfg = robot_cfg.get("air_pump", {})
        pump_enabled = isinstance(pump_cfg, Mapping) and bool(pump_cfg.get("enabled", False))

    print()
    print("About to start live Gomoku:")
    print(f"  robot_stone: {robot_stone}")
    print(f"  robot: {'dry-run' if args.dry_run_robot else 'enabled'}")
    print(f"  air_pump: {'enabled' if pump_enabled else 'disabled'}")
    print(f"  play_area: {orchestrator.play_area.describe()}")
    print(f"  max_turns: {max_turns}")
    print(
        "  per_move_confirm: "
        f"{'enabled' if args.confirm_robot_moves and not args.dry_run_robot else 'disabled'}"
    )
    print(f"  rapfi: {ai_cfg.get('engine_path') or resolve_rapfi_engine_path()}")
    confirm = input("Press Enter to start, or type q to cancel > ").strip().lower()
    if confirm == "q":
        raise KeyboardInterrupt


def _ensure_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _format_action(action: Mapping[str, float]) -> str:
    return "\n".join(f"  {key}: {float(action[key]):.3f}" for key in sorted(action))


def _format_action_delta(
    current: Mapping[str, float],
    target: Mapping[str, float],
    *,
    limit: int = 3,
) -> str:
    deltas = {
        key: float(target[key]) - float(current[key])
        for key in target
        if key in current
    }
    if not deltas:
        return "  current/delta: unavailable"
    biggest = sorted(deltas.items(), key=lambda item: abs(item[1]), reverse=True)[:limit]
    summary = ", ".join(f"{key} {delta:+.2f}" for key, delta in biggest)
    return f"  largest joint deltas: {summary}"


if __name__ == "__main__":
    raise SystemExit(main())
