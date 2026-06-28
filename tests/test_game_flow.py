from __future__ import annotations

import numpy as np
import pytest

import src.game.ai as ai_module
import src.game.llm_ai as llm_ai_module
import src.orchestrator as orchestrator_module
from src.game.decision import AIDecision, AIMoveError
from src.game.play_area import PlayArea, parse_play_area_config
from src.interaction import (
    HumanTurnCommand,
    HumanTurnResult,
    KeyboardHumanTurnController,
)
from src.robot.pose_mapper import MeasuredBoardPoseMapper
from src.utils.config_loader import load_config
from src.utils.constants import BLACK, BOARD_COLS, BOARD_ROWS, EMPTY, EMPTY_BOARD, WHITE


def _measured_pose_data(size: int = 3) -> dict:
    positions = {}
    for row in range(size):
        for col in range(size):
            positions[f"r{row + 1}c{col + 1}"] = {
                "row": row + 1,
                "col": col + 1,
                "row_index": row,
                "col_index": col,
                "action": {
                    "joint.pos": row * 100.0 + col * 10.0,
                    "gripper.pos": row + col,
                },
            }
    return {
        "board": {"kind": "gomoku", "size": size},
        "coordinate_space": "lerobot_action",
        "positions": positions,
    }


def test_orchestrator_tracks_robot_and_human_colors() -> None:
    orchestrator = orchestrator_module.GameOrchestrator(
        state_extractor=object(),
        my_stone=WHITE,
    )

    assert orchestrator.robot_stone == WHITE
    assert orchestrator.my_stone == WHITE
    assert orchestrator.human_stone == BLACK
    assert orchestrator.human_moves_first is True
    assert orchestrator.robot_moves_first is False
    assert orchestrator.is_human_turn() is True

    orchestrator.board.place(7, 7, BLACK)

    assert orchestrator.is_robot_turn() is True


def test_rapfi_default_paths_are_platform_specific() -> None:
    macos_path = ai_module.resolve_rapfi_engine_path(system="Darwin", machine="arm64")
    pi_path = ai_module.resolve_rapfi_engine_path(system="Linux", machine="aarch64")

    assert macos_path.as_posix().endswith("bin/rapfi/macos-arm64/rapfi")
    assert pi_path.as_posix().endswith("bin/rapfi/linux-aarch64/rapfi")


def test_ai_config_derives_board_size_from_play_area(tmp_path) -> None:
    engine_path = tmp_path / "rapfi"

    settings = ai_module.configure_ai_from_config(
        {
            "game": {
                "play_area": {"rows": 9, "cols": 9, "row_offset": 3, "col_offset": 3},
                "ai": {"engine_path": str(engine_path)},
            }
        }
    )

    assert settings.board_size == 9
    assert settings.engine_path == engine_path
    assert settings.provider == "rapfi"


def test_ai_config_can_select_openrouter(tmp_path) -> None:
    settings = ai_module.configure_ai_from_config(
        {
            "game": {
                "ai_level": "强力",
                "play_area": {"rows": 9, "cols": 9, "row_offset": 3, "col_offset": 3},
                "ai": {
                    "provider": "openrouter",
                    "openrouter": {
                        "key_path": "config/key.yaml",
                        "model": "openrouter/auto",
                    },
                },
            }
        },
        base_dir=tmp_path,
    )

    assert settings.provider == "openrouter"
    assert settings.board_size == 9
    assert settings.openrouter is not None
    assert settings.openrouter.key_path == tmp_path / "config" / "key.yaml"
    assert settings.openrouter.strength == "strong"


def test_default_live_config_uses_openrouter() -> None:
    config_path = ai_module.PROJECT_ROOT / "config" / "default.yaml"
    config = load_config(config_path)

    assert config["game"]["ai"]["provider"] == "openrouter"
    assert config["game"]["ai"]["openrouter"]["model"] == "deepseek/deepseek-v4-flash"


def test_from_config_prefers_robot_stone_and_keeps_my_stone_compatibility() -> None:
    orchestrator = orchestrator_module.GameOrchestrator.from_config(
        state_extractor=object(),
        config={
            "robot": {"calibrate_before_game": True},
            "game": {"robot_stone": "white", "my_stone": "white"},
        },
    )

    assert orchestrator.robot_stone == WHITE
    assert orchestrator.human_stone == BLACK


def test_from_config_rejects_conflicting_robot_stone_aliases() -> None:
    with pytest.raises(ValueError, match="robot_stone.*my_stone"):
        orchestrator_module.GameOrchestrator.from_config(
            state_extractor=object(),
            config={
                "robot": {"calibrate_before_game": True},
                "game": {"robot_stone": "white", "my_stone": "black"},
            },
        )


def test_play_robot_turn_uses_ai_and_places_robot_stone(monkeypatch: pytest.MonkeyPatch) -> None:
    mapper = MeasuredBoardPoseMapper.from_json_data(_measured_pose_data())
    monkeypatch.setattr(
        orchestrator_module,
        "ai_decide_verbose",
        lambda _board, _stone: AIDecision(row=1, col=1),
    )
    orchestrator = orchestrator_module.GameOrchestrator(
        state_extractor=object(),
        pose_mapper=mapper,
        my_stone=BLACK,
    )

    row, col, target = orchestrator.play_robot_turn()

    assert (row, col) == (1, 1)
    assert target == {"joint.pos": 110.0, "gripper.pos": 2.0}
    assert orchestrator.board.get(1, 1) == BLACK


def test_play_robot_turn_crops_center_9x9_for_ai_and_maps_to_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapper = MeasuredBoardPoseMapper.from_json_data(_measured_pose_data(size=9))
    play_area = PlayArea(row_offset=3, col_offset=3, rows=9, cols=9)
    calls = []

    def fake_ai_decide(board, stone):
        calls.append((board.shape, stone))
        assert board.shape == (9, 9)
        return AIDecision(row=4, col=4)

    monkeypatch.setattr(orchestrator_module, "ai_decide_verbose", fake_ai_decide)
    orchestrator = orchestrator_module.GameOrchestrator(
        state_extractor=object(),
        pose_mapper=mapper,
        play_area=play_area,
        my_stone=BLACK,
    )

    row, col, target = orchestrator.play_robot_turn()

    assert calls == [((9, 9), BLACK)]
    assert (row, col) == (7, 7)
    assert target == {"joint.pos": 440.0, "gripper.pos": 8.0}
    assert orchestrator.board.get(7, 7) == BLACK


def test_play_area_config_center_9x9_filters_full_board_state() -> None:
    play_area = parse_play_area_config("center_9x9")
    board = EMPTY_BOARD.copy()
    board[0, 0] = BLACK
    board[7, 7] = WHITE

    filtered = play_area.filter_board_state(board)

    assert play_area.to_local(7, 7) == (4, 4)
    assert play_area.to_global(4, 4) == (7, 7)
    assert filtered[0, 0] == EMPTY
    assert filtered[7, 7] == WHITE


def test_play_robot_turn_rejects_wrong_turn() -> None:
    orchestrator = orchestrator_module.GameOrchestrator(
        state_extractor=object(),
        my_stone=WHITE,
    )

    with pytest.raises(RuntimeError, match="next turn"):
        orchestrator.play_robot_turn()


def test_sync_board_state_adopts_latest_camera_board() -> None:
    board_from_camera = EMPTY_BOARD.copy()
    board_from_camera[7, 7] = BLACK

    class FakeExtractor:
        def extract(self):
            return board_from_camera.copy(), None

    orchestrator = orchestrator_module.GameOrchestrator(
        state_extractor=FakeExtractor(),
        my_stone=BLACK,
    )

    synced = orchestrator.sync_board_state()

    assert synced is orchestrator.board
    assert orchestrator.board.get(7, 7) == BLACK
    assert orchestrator.move_count == 1


def test_sync_board_state_keeps_current_board_when_camera_loses_stones() -> None:
    class FakeExtractor:
        def extract(self):
            return EMPTY_BOARD.copy(), None

    orchestrator = orchestrator_module.GameOrchestrator(
        state_extractor=FakeExtractor(),
        my_stone=BLACK,
    )
    orchestrator.board.place(7, 7, BLACK)
    orchestrator.move_count = 1

    synced = orchestrator.sync_board_state()

    assert synced is orchestrator.board
    assert orchestrator.board.get(7, 7) == BLACK
    assert orchestrator.move_count == 1


def test_run_once_skips_ai_when_robot_is_not_next_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_ai_decide(_board, _stone):
        pytest.fail("AI should not move before the human first move")

    class FakeExtractor:
        def extract(self):
            return EMPTY_BOARD.copy(), None

    monkeypatch.setattr(orchestrator_module, "ai_decide_verbose", fail_ai_decide)
    orchestrator = orchestrator_module.GameOrchestrator(
        state_extractor=FakeExtractor(),
        my_stone=WHITE,
    )

    assert orchestrator.run_once() == EMPTY
    assert orchestrator.move_count == 0


def test_wait_for_opponent_confirms_before_perception() -> None:
    board_after_human = EMPTY_BOARD.copy()
    board_after_human[0, 0] = BLACK
    events: list[tuple[str, int] | tuple[str]] = []

    class FakeHumanTurnController:
        def wait_for_move_done(self, *, expected_stone, board_state=None):
            events.append(("confirm", expected_stone))
            assert board_state is not None
            return HumanTurnResult()

    class FakeExtractor:
        def extract(self):
            events.append(("extract",))
            return board_after_human.copy(), (0, 0, BLACK)

    orchestrator = orchestrator_module.GameOrchestrator(
        state_extractor=FakeExtractor(),
        human_turn_controller=FakeHumanTurnController(),
        my_stone=WHITE,
    )

    board = orchestrator.wait_for_opponent(max_attempts=1, poll_interval_seconds=0)

    assert board is orchestrator.board
    assert orchestrator.board.get(0, 0) == BLACK
    assert events == [("confirm", BLACK), ("extract",)]


def test_wait_for_opponent_detects_human_move_against_internal_board() -> None:
    board_after_human = EMPTY_BOARD.copy()
    board_after_human[7, 7] = BLACK
    board_after_human[7, 8] = WHITE

    class FakeExtractor:
        def extract(self):
            return board_after_human.copy(), None

    orchestrator = orchestrator_module.GameOrchestrator(
        state_extractor=FakeExtractor(),
        my_stone=BLACK,
    )
    orchestrator.board.place(7, 7, BLACK)
    orchestrator.move_count = 1

    board = orchestrator.wait_for_opponent(
        confirm_human=False,
        max_attempts=1,
        poll_interval_seconds=0,
    )

    assert board is orchestrator.board
    assert orchestrator.board.get(7, 7) == BLACK
    assert orchestrator.board.get(7, 8) == WHITE


def test_wait_for_opponent_quit_returns_without_perception() -> None:
    class FakeHumanTurnController:
        def wait_for_move_done(self, *, expected_stone, board_state=None):
            return HumanTurnResult(HumanTurnCommand.QUIT)

    class FakeExtractor:
        calls = 0

        def extract(self):
            self.calls += 1
            return EMPTY_BOARD.copy(), None

    extractor = FakeExtractor()
    orchestrator = orchestrator_module.GameOrchestrator(
        state_extractor=extractor,
        human_turn_controller=FakeHumanTurnController(),
    )

    assert orchestrator.wait_for_opponent(max_attempts=1, poll_interval_seconds=0) is None
    assert extractor.calls == 0


def test_keyboard_human_turn_controller_dispatches_reserved_actions() -> None:
    inputs = iter(["s", "d", "g", " "])
    events: list[tuple[str, object]] = []

    class FakeInteraction:
        def speak(self, text: str) -> None:
            events.append(("speak", text))

        def dance(self, name: str = "default") -> None:
            events.append(("dance", name))

        def use_skill_gomoku(self, context=None) -> None:
            events.append(("skill", context))

    controller = KeyboardHumanTurnController(
        input_fn=lambda _prompt: next(inputs),
        print_fn=lambda _message: None,
        robot_interaction=FakeInteraction(),
    )

    result = controller.wait_for_move_done(expected_stone=BLACK, board_state="board")

    assert result.command == HumanTurnCommand.MOVE_DONE
    assert events == [
        ("speak", "我在看棋盘，准备继续。"),
        ("dance", "gomoku_waiting"),
        ("skill", {"expected_stone": "黑棋", "board_state": "board"}),
    ]


def test_orchestrator_hri_hooks_forward_to_interaction_controller() -> None:
    events: list[tuple[str, object]] = []

    class FakeInteraction:
        def speak(self, text: str) -> None:
            events.append(("speak", text))

        def dance(self, name: str = "default") -> None:
            events.append(("dance", name))

        def use_skill_gomoku(self, context=None) -> None:
            events.append(("skill", context))

    orchestrator = orchestrator_module.GameOrchestrator(
        state_extractor=object(),
        interaction_controller=FakeInteraction(),
    )

    orchestrator.robot_say("轮到我了")
    orchestrator.robot_dance("win")
    orchestrator.robot_use_skill_gomoku({"phase": "opening"})

    assert events == [
        ("speak", "轮到我了"),
        ("dance", "win"),
        ("skill", {"phase": "opening"}),
    ]


def test_play_robot_turn_forwards_llm_talk_and_skill(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, object]] = []
    mapper = MeasuredBoardPoseMapper.from_json_data(_measured_pose_data())

    class FakeInteraction:
        def speak(self, text: str) -> None:
            events.append(("speak", text))

        def dance(self, name: str = "default") -> None:
            events.append(("dance", name))

        def use_skill_gomoku(self, context=None) -> None:
            events.append(("skill", context))

    monkeypatch.setattr(
        orchestrator_module,
        "ai_decide_verbose",
        lambda _board, _stone: AIDecision(
            row=1,
            col=1,
            use_skill=True,
            trash_talk="这手棋很有节目效果。",
            rationale="center pressure",
            source="openrouter",
        ),
    )
    orchestrator = orchestrator_module.GameOrchestrator(
        state_extractor=object(),
        pose_mapper=mapper,
        interaction_controller=FakeInteraction(),
        my_stone=BLACK,
    )

    orchestrator.play_robot_turn()

    assert events == [
        ("speak", "这手棋很有节目效果。"),
        (
            "skill",
            {
                "source": "openrouter",
                "row": 1,
                "col": 1,
                "rationale": "center pressure",
            },
        ),
    ]


def test_openrouter_decision_parses_one_based_legal_move() -> None:
    board = np.zeros((9, 9), dtype=np.int8)
    board[4, 4] = BLACK

    decision = llm_ai_module._parse_decision(
        '{"should_play": true, "row": 5, "col": 6, '
        '"use_skill": true, "trash_talk": "你这棋盘归我了"}',
        board,
    )

    assert decision.row == 4
    assert decision.col == 5
    assert decision.use_skill is True
    assert decision.trash_talk == "你这棋盘归我了"
    assert decision.source == "openrouter"


def test_openrouter_decision_rejects_occupied_move() -> None:
    board = np.zeros((9, 9), dtype=np.int8)
    board[4, 4] = BLACK

    with pytest.raises(AIMoveError):
        llm_ai_module._parse_decision('{"should_play": true, "row": 5, "col": 5}', board)


def test_ai_decide_uses_begin_move_for_empty_black_opening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    class FakeEngine:
        def set_board(self, board: np.ndarray, my_stone: int):
            calls.append(("set_board", int(np.count_nonzero(board)), my_stone))
            raise RuntimeError("not started")

        def start_game(self, is_black: bool = True):
            calls.append(("start_game", is_black))
            return (7, 7) if is_black else None

    monkeypatch.setattr(ai_module, "_engine", None)
    monkeypatch.setattr(ai_module, "_get_engine", lambda time_per_move_ms=3000: FakeEngine())

    board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)

    assert ai_module.ai_decide(board, BLACK) == (7, 7)
    assert calls == [("set_board", 0, BLACK), ("start_game", True)]


def test_ai_decide_restarts_without_begin_for_nonempty_black_board(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    class FakeEngine:
        def __init__(self) -> None:
            self.started = False

        def set_board(self, board: np.ndarray, my_stone: int):
            calls.append(("set_board", int(np.count_nonzero(board)), my_stone))
            if not self.started:
                raise RuntimeError("not started")
            return (8, 8)

        def start_game(self, is_black: bool = True):
            calls.append(("start_game", is_black))
            self.started = True
            return (7, 7) if is_black else None

    fake_engine = FakeEngine()
    monkeypatch.setattr(ai_module, "_engine", None)
    monkeypatch.setattr(ai_module, "_get_engine", lambda time_per_move_ms=3000: fake_engine)

    board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    board[7, 7] = BLACK

    assert ai_module.ai_decide(board, BLACK) == (8, 8)
    assert calls == [
        ("set_board", 1, BLACK),
        ("start_game", False),
        ("set_board", 1, BLACK),
    ]
