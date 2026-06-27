from __future__ import annotations

import json

import pytest
import yaml

from src.orchestrator import GameOrchestrator
from src.perception.board_calibration import load_board_frame_calibration
from src.perception.board_detector import BoardDetector
from src.robot.calibration import (
    load_robot_calibration,
    run_manual_robot_calibration,
    save_robot_calibration,
)
from src.robot.controller import (
    CalibrationPoints,
    board_to_robot_coords,
    board_to_robot_pose,
)
from src.robot.lerobot_calibration import (
    DEFAULT_SO101_ROBOT_ID,
    bundled_lerobot_calibration_path,
    install_bundled_lerobot_calibration,
    lerobot_calibration_cache_path,
)
from src.robot.pose_mapper import MeasuredBoardPoseMapper
from src.robot.so101_lowlevel_mover import (
    lerobot_action_to_raw,
    load_lerobot_motor_calibration,
    make_lowlevel_profile,
    raw_to_lerobot_action,
)
from src.robot.so101_mover import (
    MotionProfile,
    SO101SmoothMover,
    ensure_action,
    interpolate_action,
    smoothstep,
)
from src.utils.constants import WHITE


def _measured_pose_data(size: int = 3) -> dict:
    positions = {}
    for row in range(size):
        for col in range(size):
            nonlinear_center_offset = 50.0 if row == 1 and col == 1 else 0.0
            positions[f"r{row + 1}c{col + 1}"] = {
                "row": row + 1,
                "col": col + 1,
                "row_index": row,
                "col_index": col,
                "action": {
                    "joint.pos": row * 100.0 + col * 10.0 + nonlinear_center_offset,
                    "gripper.pos": row + col,
                },
            }
    return {
        "board": {"kind": "gomoku", "size": size},
        "coordinate_space": "lerobot_action",
        "positions": positions,
    }


def test_cartesian_calibration_interpolates_board_center() -> None:
    calib = CalibrationPoints.from_list(
        [
            (0.0, 0.0, 10.0),
            (14.0, 0.0, 10.0),
            (14.0, 14.0, 10.0),
            (0.0, 14.0, 10.0),
        ]
    )

    assert board_to_robot_coords(7, 7, 15, 15, calib, z_height=42.0) == (7.0, 7.0, 42.0)


def test_mapping_calibration_interpolates_each_action_key() -> None:
    calib = CalibrationPoints.from_corners(
        {
            "top_left": {"shoulder_pan.pos": 0.0, "elbow_flex.pos": 10.0},
            "top_right": {"shoulder_pan.pos": 14.0, "elbow_flex.pos": 10.0},
            "bottom_right": {"shoulder_pan.pos": 14.0, "elbow_flex.pos": 24.0},
            "bottom_left": {"shoulder_pan.pos": 0.0, "elbow_flex.pos": 24.0},
        }
    )

    assert board_to_robot_pose(7, 7, 15, 15, calib) == {
        "shoulder_pan.pos": 7.0,
        "elbow_flex.pos": 17.0,
    }


def test_manual_calibration_uses_sampler_and_finishes_with_hold() -> None:
    class FakeSampler:
        coordinate_space = "fake"

        def __init__(self) -> None:
            self.started = False
            self.finished_hold = None
            self.poses = iter(
                [
                    (0, 0, 0),
                    (10, 0, 0),
                    (10, 10, 0),
                    (0, 10, 0),
                ]
            )

        def prepare_manual_guidance(self) -> None:
            self.started = True

        def read_current_pose(self):
            return next(self.poses)

        def finish_manual_guidance(self, hold: bool = False) -> None:
            self.finished_hold = hold

    sampler = FakeSampler()
    prompts: list[str] = []

    calib = run_manual_robot_calibration(
        sampler,
        input_fn=lambda prompt: prompts.append(prompt) or "",
        print_fn=lambda _message: None,
        hold_after=True,
    )

    assert sampler.started
    assert sampler.finished_hold is True
    assert len(prompts) == 4
    assert board_to_robot_coords(14, 14, 15, 15, calib, z_height=5) == (10.0, 10.0, 5.0)


def test_save_and_load_robot_calibration_round_trip(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("robot:\n  port: test\n", encoding="utf-8")
    calib = CalibrationPoints.from_list(
        [
            {"a.pos": 0.0, "b.pos": 0.0},
            {"a.pos": 1.0, "b.pos": 0.0},
            {"a.pos": 1.0, "b.pos": 1.0},
            {"a.pos": 0.0, "b.pos": 1.0},
        ]
    )

    save_robot_calibration(config_path, calib, coordinate_space="lerobot_action", z_height=30.0)

    loaded_yaml = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    loaded = load_robot_calibration(loaded_yaml)

    assert loaded.to_corners_dict() == calib.to_corners_dict()
    assert loaded_yaml["robot"]["calibration"]["coordinate_space"] == "lerobot_action"
    assert loaded_yaml["robot"]["z_height"] == 30.0


def test_orchestrator_from_config_allows_required_startup_calibration() -> None:
    orchestrator = GameOrchestrator.from_config(
        state_extractor=object(),
        config={
            "robot": {"calibrate_before_game": True, "z_height": 12.0},
            "game": {"my_stone": "white"},
        },
    )

    assert orchestrator.calib is None
    assert orchestrator.z_height == 12.0
    assert orchestrator.my_stone == WHITE


def test_board_frame_calibration_loads_manual_corners() -> None:
    calibration = load_board_frame_calibration(
        {
            "board": {
                "calibration": {
                    "method": "manual",
                    "corners": {
                        "top_left": [10, 20],
                        "top_right": [110, 20],
                        "bottom_right": [120, 120],
                        "bottom_left": [0, 120],
                    },
                }
            }
        },
        required=True,
    )

    assert calibration is not None
    assert calibration.corners["top_left"] == (10.0, 20.0)
    assert calibration.as_array().shape == (4, 2)


def test_manual_board_detection_requires_recorded_board_frame() -> None:
    with pytest.raises(ValueError, match="calibrate_board.py"):
        BoardDetector(
            {
                "board_detection": {"method": "manual"},
                "board": {"calibration": {"method": "manual", "corners": {}}},
            }
        )


def test_measured_pose_mapper_loads_direct_positions_without_interpolation() -> None:
    mapper = MeasuredBoardPoseMapper.from_json_data(_measured_pose_data())

    assert mapper.board_rows == 3
    assert mapper.board_cols == 3
    assert mapper.coordinate_space == "lerobot_action"
    assert mapper.target_for_cell(1, 1) == {
        "joint.pos": 160.0,
        "gripper.pos": 2.0,
    }

    corners = mapper.corner_calibration_points()
    interpolated_center = board_to_robot_pose(1, 1, 3, 3, corners)

    assert interpolated_center == {"joint.pos": 110.0, "gripper.pos": 2.0}
    assert mapper.target_for_cell(1, 1) != interpolated_center


def test_measured_pose_mapper_replays_four_corners_in_order() -> None:
    mapper = MeasuredBoardPoseMapper.from_json_data(_measured_pose_data())
    seen: list[dict[str, float]] = []
    labels: list[str] = []

    class FakeMover:
        def move_to(self, target_pose):
            seen.append(dict(target_pose))
            return dict(target_pose)

    mapper.replay_corners(
        FakeMover(),
        before_move=lambda _name, target: labels.append(target.label),
    )

    assert labels == ["r1c1", "r1c3", "r3c3", "r3c1"]
    assert seen == [
        mapper.target_for_label("r1c1"),
        mapper.target_for_label("r1c3"),
        mapper.target_for_label("r3c3"),
        mapper.target_for_label("r3c1"),
    ]


def test_orchestrator_prefers_measured_pose_mapper_over_legacy_calibration() -> None:
    mapper = MeasuredBoardPoseMapper.from_json_data(_measured_pose_data())
    legacy_calib = CalibrationPoints.from_list(
        [
            {"joint.pos": 0.0, "gripper.pos": 0.0},
            {"joint.pos": 20.0, "gripper.pos": 2.0},
            {"joint.pos": 220.0, "gripper.pos": 4.0},
            {"joint.pos": 200.0, "gripper.pos": 2.0},
        ]
    )
    orchestrator = GameOrchestrator(
        state_extractor=object(),
        calib=legacy_calib,
        pose_mapper=mapper,
    )

    assert orchestrator.execute_my_move(1, 1) == {
        "joint.pos": 160.0,
        "gripper.pos": 2.0,
    }


def test_orchestrator_from_config_loads_measured_pose_map(tmp_path) -> None:
    pose_map_path = tmp_path / "poses.json"
    pose_map_path.write_text(json.dumps(_measured_pose_data()), encoding="utf-8")

    orchestrator = GameOrchestrator.from_config(
        state_extractor=object(),
        config={
            "robot": {
                "pose_map": {
                    "method": "measured",
                    "path": str(pose_map_path),
                }
            }
        },
    )

    assert orchestrator.pose_mapper is not None
    assert orchestrator.execute_my_move(1, 1) == {
        "joint.pos": 160.0,
        "gripper.pos": 2.0,
    }


def test_orchestrator_executes_pick_move_drop_sequence_with_suction() -> None:
    mapper = MeasuredBoardPoseMapper.from_json_data(_measured_pose_data())
    events = []
    pickup_pose = {"joint.pos": -10.0, "gripper.pos": 0.0}

    class FakeMover:
        def move_to(self, target_pose):
            events.append(("move", dict(target_pose)))
            return dict(target_pose)

    class FakeSuction:
        def pick_stone(self) -> None:
            events.append(("pick",))

        def hold_stone(self) -> None:
            events.append(("hold",))

        def drop_stone(self) -> None:
            events.append(("drop",))

        def off(self) -> None:
            events.append(("off",))

    orchestrator = GameOrchestrator(
        state_extractor=object(),
        pose_mapper=mapper,
        robot_mover=FakeMover(),
        suction_controller=FakeSuction(),
        pickup_pose=pickup_pose,
    )

    assert orchestrator.execute_my_move(1, 1) == {
        "joint.pos": 160.0,
        "gripper.pos": 2.0,
    }
    assert events == [
        ("move", pickup_pose),
        ("pick",),
        ("move", {"joint.pos": 160.0, "gripper.pos": 2.0}),
        ("drop",),
    ]


def test_orchestrator_uses_pickup_pose_for_robot_stone_colour() -> None:
    mapper = MeasuredBoardPoseMapper.from_json_data(_measured_pose_data())
    events = []
    black_pickup = {"joint.pos": -10.0, "gripper.pos": 1.0}
    white_pickup = {"joint.pos": -20.0, "gripper.pos": 2.0}

    class FakeMover:
        def move_to(self, target_pose):
            events.append(("move", dict(target_pose)))
            return dict(target_pose)

    class FakeSuction:
        def pick_stone(self) -> None:
            events.append(("pick",))

        def drop_stone(self) -> None:
            events.append(("drop",))

        def off(self) -> None:
            events.append(("off",))

    orchestrator = GameOrchestrator(
        state_extractor=object(),
        pose_mapper=mapper,
        robot_mover=FakeMover(),
        suction_controller=FakeSuction(),
        pickup_poses={1: black_pickup, 2: white_pickup},
        my_stone=WHITE,
    )

    orchestrator.execute_my_move(1, 1)

    assert events[0] == ("move", white_pickup)
    assert ("move", black_pickup) not in events


def test_orchestrator_uses_waiting_pose_between_pick_and_place() -> None:
    mapper = MeasuredBoardPoseMapper.from_json_data(_measured_pose_data())
    events = []
    pickup_pose = {"joint.pos": -10.0, "gripper.pos": 0.0}
    waiting_pose = {"joint.pos": -5.0, "gripper.pos": 1.0}

    class FakeMover:
        def move_to(self, target_pose):
            events.append(("move", dict(target_pose)))
            return dict(target_pose)

    class FakeSuction:
        def pick_stone(self) -> None:
            events.append(("pick",))

        def drop_stone(self) -> None:
            events.append(("drop",))

        def off(self) -> None:
            events.append(("off",))

    orchestrator = GameOrchestrator(
        state_extractor=object(),
        pose_mapper=mapper,
        robot_mover=FakeMover(),
        suction_controller=FakeSuction(),
        pickup_pose=pickup_pose,
        waiting_pose=waiting_pose,
    )

    orchestrator.execute_my_move(1, 1)

    assert events == [
        ("move", pickup_pose),
        ("pick",),
        ("move", waiting_pose),
        ("move", {"joint.pos": 160.0, "gripper.pos": 2.0}),
        ("drop",),
        ("move", waiting_pose),
    ]


def test_orchestrator_turns_suction_off_if_move_fails_after_pick() -> None:
    mapper = MeasuredBoardPoseMapper.from_json_data(_measured_pose_data())
    events = []

    class FailingMover:
        def move_to(self, target_pose):
            events.append(("move", dict(target_pose)))
            raise RuntimeError("blocked")

    class FakeSuction:
        def pick_stone(self) -> None:
            events.append(("pick",))

        def hold_stone(self) -> None:
            events.append(("hold",))

        def drop_stone(self) -> None:
            events.append(("drop",))

        def off(self) -> None:
            events.append(("off",))

    orchestrator = GameOrchestrator(
        state_extractor=object(),
        pose_mapper=mapper,
        robot_mover=FailingMover(),
        suction_controller=FakeSuction(),
    )

    with pytest.raises(RuntimeError, match="blocked"):
        orchestrator.execute_my_move(1, 1)

    assert events == [
        ("pick",),
        ("move", {"joint.pos": 160.0, "gripper.pos": 2.0}),
        ("off",),
    ]


def test_so101_action_helpers_validate_and_interpolate() -> None:
    target = ensure_action({"shoulder_pan.pos": 10, "elbow_flex.pos": 20})
    start = {"shoulder_pan.pos": 0.0, "elbow_flex.pos": 10.0}

    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0
    assert interpolate_action(start, target, 0.5) == {
        "shoulder_pan.pos": 5.0,
        "elbow_flex.pos": 15.0,
    }


def test_so101_lerobot_action_converts_to_lowlevel_raw_ticks() -> None:
    calibration = load_lerobot_motor_calibration()
    action = {
        "elbow_flex.pos": 44.08791208791209,
        "gripper.pos": 32.26925746009716,
        "shoulder_lift.pos": -10.901098901098901,
        "shoulder_pan.pos": 2.10989010989011,
        "wrist_flex.pos": 25.0989010989011,
        "wrist_roll.pos": 1.1868131868131868,
    }

    raw = lerobot_action_to_raw(action, calibration)

    assert raw == {
        "elbow_flex": 2549,
        "gripper": 2440,
        "shoulder_lift": 1909,
        "shoulder_pan": 2028,
        "wrist_flex": 2349,
        "wrist_roll": 2061,
    }
    roundtrip = raw_to_lerobot_action(raw, calibration)
    for key, value in action.items():
        assert roundtrip[key] == pytest.approx(value, abs=0.1)


def test_lowlevel_profile_keeps_verified_v3_lookahead_defaults() -> None:
    profile = make_lowlevel_profile()

    assert profile.duration_seconds == 12.0
    assert profile.dt_seconds == 0.02
    assert profile.lookahead_for("shoulder_pan") == 24
    assert profile.lookahead_for("shoulder_lift") == 80
    assert profile.lookahead_for("elbow_flex") == 60
    assert profile.lookahead_for("wrist_flex") == 24
    assert profile.lookahead_for("wrist_roll") == 8
    assert profile.lookahead_for("gripper") == 8


def test_so101_move_to_streams_intermediate_targets_then_final_target() -> None:
    class FakeRobot:
        def __init__(self) -> None:
            self.sent = []
            self.observations = iter(
                [
                    {"joint.pos": 0.0},
                    {"joint.pos": 10.0},
                ]
            )

        def send_action(self, action):
            self.sent.append(dict(action))
            return dict(action)

        def get_observation(self):
            return next(self.observations)

    fake_robot = FakeRobot()
    mover = SO101SmoothMover.__new__(SO101SmoothMover)
    mover.robot = fake_robot
    mover.profile = MotionProfile(
        duration_seconds=0.003,
        dt_seconds=0.001,
    )

    final = mover.move_to({"joint.pos": 10.0})

    assert len(fake_robot.sent) == 4
    assert fake_robot.sent[0]["joint.pos"] == pytest.approx(smoothstep(1 / 3) * 10.0)
    assert fake_robot.sent[1]["joint.pos"] == pytest.approx(smoothstep(2 / 3) * 10.0)
    assert fake_robot.sent[2] == {"joint.pos": 10.0}
    assert fake_robot.sent[3] == {"joint.pos": 10.0}
    assert final == {"joint.pos": 10.0}


def test_bundled_lerobot_calibration_installs_without_overwriting(tmp_path) -> None:
    source = bundled_lerobot_calibration_path(DEFAULT_SO101_ROBOT_ID)
    assert source.is_file()

    installed = install_bundled_lerobot_calibration(calibration_root=tmp_path)
    expected = lerobot_calibration_cache_path(calibration_root=tmp_path)

    assert installed == expected
    assert expected.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")

    expected.write_text('{"custom": true}\n', encoding="utf-8")
    install_bundled_lerobot_calibration(calibration_root=tmp_path)
    assert expected.read_text(encoding="utf-8") == '{"custom": true}\n'

    install_bundled_lerobot_calibration(calibration_root=tmp_path, overwrite=True)
    assert expected.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
