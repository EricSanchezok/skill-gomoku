from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

from src.robot.so101_mover import (
    LEROBOT_CLAMP_WARNING_PREFIX,
    PRESET_ACTIONS,
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


def test_position_tokens_are_one_based() -> None:
    move_module = _load_move_module()

    assert move_module.parse_position("r5c6") == (4, 5)
    assert move_module.parse_position("5,6") == (4, 5)


def test_waiting_preset_is_available() -> None:
    waiting = PRESET_ACTIONS["waiting"]

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
