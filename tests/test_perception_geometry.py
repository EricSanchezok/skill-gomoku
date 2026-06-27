from __future__ import annotations

import pytest

from src.perception.board_detector import BoardDetector
from src.perception.grid_mapper import GridMapper


def _center(rect: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = rect
    return x + w / 2.0, y + h / 2.0


def test_board_detector_fallback_positions_are_gomoku_intersections() -> None:
    detector = BoardDetector({"board_detection": {"method": "auto"}})
    cells = detector._compute_grid_cells([], [], 600)

    assert len(cells) == 15
    assert len(cells[0]) == 15
    assert _center(cells[0][0]) == pytest.approx((0.0, 0.0), abs=0.6)
    assert _center(cells[14][14]) == pytest.approx((599.0, 599.0), abs=0.6)
    assert _center(cells[7][7]) == pytest.approx((299.5, 299.5), abs=0.6)


def test_board_detector_detected_lines_use_15_intersections_not_16_boundaries() -> None:
    detector = BoardDetector({"board_detection": {"method": "auto"}})
    lines = [idx * 10 for idx in range(15)]
    cells = detector._compute_grid_cells(lines, lines, 600)

    assert _center(cells[0][0]) == pytest.approx((0.0, 0.0), abs=0.01)
    assert _center(cells[14][14]) == pytest.approx((140.0, 140.0), abs=0.01)
    assert _center(cells[7][7]) == pytest.approx((70.0, 70.0), abs=0.01)


def test_grid_mapper_maps_pixels_to_intersection_positions() -> None:
    detector = BoardDetector({"board_detection": {"method": "auto"}})
    mapper = GridMapper()
    mapper.load_cells(detector._compute_grid_cells([], [], 600))

    assert mapper.pixel_to_grid(0, 0) == (0, 0)
    assert mapper.pixel_to_grid(599, 599) == (14, 14)
    assert mapper.grid_to_pixel(7, 7) == pytest.approx((299.5, 299.5), abs=0.6)
