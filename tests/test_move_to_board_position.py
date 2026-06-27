from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

from src.game.play_area import PlayArea
from src.robot.pose_mapper import MeasuredBoardPoseMapper
from src.robot.so101_mover import (
    LEROBOT_CLAMP_WARNING_PREFIX,
    suppress_lerobot_clamp_warnings,
)


def _load_move_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "move_to_board_position.py"
    spec = importlib.util.spec_from_file_location("move_to_board_position", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _measured_pose_data(size: int = 9) -> dict:
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


def test_position_tokens_are_one_based() -> None:
    move_module = _load_move_module()

    assert move_module._parse_position_token("r5c6") == (4, 5)
    assert move_module._parse_position_token("5,6") == (4, 5)


def test_move_target_resolves_local_pose_map_position_to_global_play_area() -> None:
    move_module = _load_move_module()
    mapper = MeasuredBoardPoseMapper.from_json_data(_measured_pose_data())
    play_area = PlayArea(row_offset=3, col_offset=3, rows=9, cols=9)

    target = move_module._resolve_position_target("r5c5", mapper, play_area, "local")

    assert target.map_label == "r5c5"
    assert target.global_label == "r8c8"
    assert target.action == {"joint.pos": 44.0}


def test_move_target_resolves_global_position_through_play_area() -> None:
    move_module = _load_move_module()
    mapper = MeasuredBoardPoseMapper.from_json_data(_measured_pose_data())
    play_area = PlayArea(row_offset=3, col_offset=3, rows=9, cols=9)

    target = move_module._resolve_position_target("8,8", mapper, play_area, "global")

    assert target.map_label == "r5c5"
    assert target.global_label == "r8c8"
    assert target.action == {"joint.pos": 44.0}


def test_waiting_action_accepts_preset_name() -> None:
    move_module = _load_move_module()

    waiting = move_module._resolve_waiting_action({"robot": {"waiting_pose": "waiting"}})

    assert "shoulder_pan.pos" in waiting
    assert "gripper.pos" in waiting


def test_clamp_warning_filter_can_be_installed_and_removed() -> None:
    root_logger = logging.getLogger()
    original_filters = list(root_logger.filters)
    original_handler_filters = [
        (handler, list(handler.filters))
        for handler in root_logger.handlers
    ]

    try:
        suppress_lerobot_clamp_warnings(True)
        filters = [
            item
            for item in root_logger.filters
            if getattr(item, "_so101_clamp_warning_filter", False)
        ]
        assert filters

        clamp_record = logging.LogRecord(
            name="root",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg=f"{LEROBOT_CLAMP_WARNING_PREFIX}: details",
            args=(),
            exc_info=None,
        )
        normal_record = logging.LogRecord(
            name="root",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="ordinary warning",
            args=(),
            exc_info=None,
        )

        assert filters[0].filter(clamp_record) is False
        assert filters[0].filter(normal_record) is True

        suppress_lerobot_clamp_warnings(False)
        assert not any(
            getattr(item, "_so101_clamp_warning_filter", False)
            for item in root_logger.filters
        )
    finally:
        root_logger.filters = original_filters
        for handler, filters in original_handler_filters:
            handler.filters = filters
