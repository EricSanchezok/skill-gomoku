from __future__ import annotations

import yaml

from src.orchestrator import GameOrchestrator
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
from src.robot.so101_mover import (
    ensure_action,
    interpolate_action,
    smoothstep,
)
from src.utils.constants import WHITE


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


def test_so101_action_helpers_validate_and_interpolate() -> None:
    target = ensure_action({"shoulder_pan.pos": 10, "elbow_flex.pos": 20})
    start = {"shoulder_pan.pos": 0.0, "elbow_flex.pos": 10.0}

    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0
    assert interpolate_action(start, target, 0.5) == {
        "shoulder_pan.pos": 5.0,
        "elbow_flex.pos": 15.0,
    }
