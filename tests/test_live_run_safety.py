from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

from src.game.play_area import PlayArea
from src.orchestrator import GameOrchestrator
from src.robot.pose_mapper import MeasuredBoardPoseMapper
from src.utils.constants import BLACK


def _load_live_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_live_game.py"
    spec = importlib.util.spec_from_file_location("run_live_game", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _measured_pose_data(size: int = 3) -> dict:
    positions = {}
    for row in range(size):
        for col in range(size):
            positions[f"r{row + 1}c{col + 1}"] = {
                "row": row + 1,
                "col": col + 1,
                "row_index": row,
                "col_index": col,
                "action": {"joint.pos": row * 10.0 + col},
            }
    return {
        "board": {"kind": "gomoku", "size": size},
        "coordinate_space": "lerobot_action",
        "positions": positions,
    }


def _args(**overrides):
    values = {
        "dry_run_robot": False,
        "allow_missing_pickup_pose": False,
        "max_turns": None,
        "full_game": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_live_robot_requires_pickup_pose_before_motion() -> None:
    live = _load_live_module()
    mapper = MeasuredBoardPoseMapper.from_json_data(_measured_pose_data())
    orchestrator = GameOrchestrator(
        state_extractor=object(),
        pose_mapper=mapper,
        robot_mover=object(),
        waiting_pose={"joint.pos": 0.0},
        my_stone=BLACK,
    )

    with pytest.raises(ValueError, match="pickup pose"):
        live._validate_live_robot_safety(orchestrator, _args())


def test_live_config_rejects_missing_pickup_pose_before_hardware_startup() -> None:
    live = _load_live_module()

    with pytest.raises(ValueError, match="before hardware startup"):
        live._validate_live_config_safety(
            {
                "robot": {
                    "waiting_pose": "waiting",
                    "pickup_poses": {"black": None, "white": None},
                },
                "game": {"robot_stone": "black"},
            },
            _args(),
        )


def test_live_config_allows_matching_pickup_pose_before_hardware_startup() -> None:
    live = _load_live_module()

    live._validate_live_config_safety(
        {
            "robot": {
                "waiting_pose": "waiting",
                "pickup_poses": {"black": {"joint.pos": 1.0}, "white": None},
            },
            "game": {"robot_stone": "black"},
        },
        _args(),
    )


def test_live_robot_allows_recorded_pickup_pose() -> None:
    live = _load_live_module()
    mapper = MeasuredBoardPoseMapper.from_json_data(_measured_pose_data())
    orchestrator = GameOrchestrator(
        state_extractor=object(),
        pose_mapper=mapper,
        robot_mover=object(),
        pickup_pose={"joint.pos": -1.0},
        waiting_pose={"joint.pos": 0.0},
        my_stone=BLACK,
    )

    live._validate_live_robot_safety(orchestrator, _args())


def test_live_run_defaults_to_one_turn_unless_full_game_is_requested() -> None:
    live = _load_live_module()
    orchestrator = GameOrchestrator(
        state_extractor=object(),
        play_area=PlayArea(row_offset=3, col_offset=3, rows=9, cols=9),
    )

    assert live._resolve_max_turns(_args(), orchestrator) == 1
    assert live._resolve_max_turns(_args(full_game=True), orchestrator) == 81
    assert live._resolve_max_turns(_args(max_turns=4), orchestrator) == 4


def test_confirming_robot_mover_prompts_before_delegating() -> None:
    live = _load_live_module()
    events = []

    class FakeMover:
        def read_action(self, target):
            events.append(("read", dict(target)))
            return {"joint.pos": 1.0}

        def move_to(self, target):
            events.append(("move", dict(target)))
            return dict(target)

    wrapper = live.ConfirmingRobotMover(
        FakeMover(),
        input_fn=lambda _prompt: "",
        print_fn=lambda _message: None,
    )

    assert wrapper.move_to({"joint.pos": 3.0}) == {"joint.pos": 3.0}
    assert events == [
        ("read", {"joint.pos": 3.0}),
        ("move", {"joint.pos": 3.0}),
    ]
