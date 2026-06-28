from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

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
                "pickup_top_poses": {"black": {"joint.pos": 2.0}, "white": None},
            },
            "game": {"robot_stone": "black"},
        },
        _args(),
    )


def test_live_config_rejects_missing_pickup_top_pose_before_hardware_startup() -> None:
    live = _load_live_module()

    with pytest.raises(ValueError, match="pickup top pose"):
        live._validate_live_config_safety(
            {
                "robot": {
                    "waiting_pose": "waiting",
                    "pickup_poses": {"black": {"joint.pos": 1.0}, "white": None},
                    "pickup_top_poses": {"black": None, "white": None},
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
        pickup_top_pose={"joint.pos": -0.5},
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


def test_startup_menu_choice_four_starts_without_side_effects(monkeypatch, tmp_path) -> None:
    live = _load_live_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("robot: {}\ngame: {}\n", encoding="utf-8")

    monkeypatch.setattr("builtins.input", lambda _prompt: "4")

    config = {"robot": {}, "game": {}}
    assert live._run_startup_menu(config_path, config, _args(), mover=None) == config


def test_lowlevel_pickup_recording_writes_black_and_white(monkeypatch, tmp_path) -> None:
    live = _load_live_module()
    config_path = tmp_path / "config.yaml"
    config = {"robot": {"pickup_poses": {}, "pickup_top_poses": {}}, "game": {}}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    class FakeMover:
        def __init__(self):
            self.events = []
            self.poses = iter(
                [
                    {"shoulder_pan.pos": 1.0, "gripper.pos": 2.0},
                    {"shoulder_pan.pos": 3.0, "gripper.pos": 4.0},
                    {"shoulder_pan.pos": 5.0, "gripper.pos": 6.0},
                    {"shoulder_pan.pos": 7.0, "gripper.pos": 8.0},
                ]
            )

        def release(self):
            self.events.append("release")

        def read_action(self):
            self.events.append("read")
            return next(self.poses)

        def hold_current(self):
            self.events.append("hold")

    mover = FakeMover()
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    live._record_pickup_poses_lowlevel(config_path, mover)

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["robot"]["pickup_poses"]["black"]["shoulder_pan.pos"] == 1.0
    assert saved["robot"]["pickup_top_poses"]["black"]["gripper.pos"] == 4.0
    assert saved["robot"]["pickup_poses"]["white"]["shoulder_pan.pos"] == 5.0
    assert saved["robot"]["pickup_top_poses"]["white"]["gripper.pos"] == 8.0
    assert mover.events == ["release", "read", "read", "read", "read", "hold"]


def test_corner_replay_routes_every_board_corner_through_waiting(
    monkeypatch,
    tmp_path,
) -> None:
    live = _load_live_module()
    pose_map_path = tmp_path / "poses.json"
    pose_map_path.write_text(json.dumps(_measured_pose_data()), encoding="utf-8")
    waiting = {"joint.pos": -1.0}
    config = {
        "robot": {
            "pose_map": {"method": "measured", "path": str(pose_map_path)},
            "waiting_pose": waiting,
        }
    }
    events = []

    class FakeMover:
        def hold_current(self):
            events.append(("hold",))

        def move_to(self, pose):
            events.append(("move", dict(pose)))

    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    live._move_robot_to_board_corners(config, FakeMover())

    assert events == [
        ("hold",),
        ("move", waiting),
        ("move", {"joint.pos": 0.0}),
        ("move", waiting),
        ("move", {"joint.pos": 2.0}),
        ("move", waiting),
        ("move", {"joint.pos": 22.0}),
        ("move", waiting),
        ("move", {"joint.pos": 20.0}),
        ("move", waiting),
    ]


def test_move_to_waiting_from_config_uses_waiting_preset() -> None:
    live = _load_live_module()
    moves = []

    class FakeMover:
        def move_to(self, pose):
            moves.append(dict(pose))

    live._move_to_waiting_from_config({"robot": {"waiting_pose": "waiting"}}, FakeMover())

    assert moves == [live.PRESET_ACTIONS["waiting"]]
