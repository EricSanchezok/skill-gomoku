#!/usr/bin/env python3
"""Perception pipeline 端到端测试脚本.

用法:
    # 使用 MockCamera 调试
    python scripts/test_perception.py --mock

    # 使用真实 webcam / RealSense RGB UVC 流
    python scripts/test_perception.py

    # 保存检测结果到文件
    python scripts/test_perception.py --mock --save output/

    # 交互模式：逐帧检测，按 q 退出
    python scripts/test_perception.py --mock --interactive
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import cv2
import numpy as np

from src.perception.camera import create_camera
from src.perception.state_extractor import StateExtractor
from src.utils.config_loader import load_config
from src.utils.constants import BLACK, EMPTY, WHITE
from src.utils.visualizer import draw_board_graphics

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("test_perception")


def print_board(state: np.ndarray) -> None:
    """打印棋盘到终端."""
    char_map = {EMPTY: ".", BLACK: "●", WHITE: "○"}
    cols_header = "   " + "".join(f"{c:2}" for c in range(state.shape[1]))
    print(cols_header)
    for r in range(state.shape[0]):
        row_str = " ".join(char_map.get(state[r, c], "?") for c in range(state.shape[1]))
        print(f"{r:2d} {row_str}")


def run_single(extractor: StateExtractor, save_dir: str | None = None) -> np.ndarray:
    """单次拍照分析."""
    logger.info("拍照中...")
    board, delta = extractor.extract()

    black_count = int(np.count_nonzero(board == BLACK))
    white_count = int(np.count_nonzero(board == WHITE))
    logger.info("棋盘状态 (%d●=%d, %d○=%d)", BLACK, black_count, WHITE, white_count)
    print_board(board)

    if delta is not None:
        r, c, stone = delta
        stone_name = "● 黑子" if stone == BLACK else "○ 白子"
        logger.info(f"检测到新落子: ({r}, {c}) {stone_name}")

    return board


def run_interactive(extractor: StateExtractor) -> None:
    """交互模式：逐帧检测并可视化."""
    logger.info("交互模式 — 按 q 退出，按 空格 拍照分析")

    board_detector = extractor.board_detector
    cam = extractor.camera
    window = "Gomoku Perception"

    while True:
        color = cam.get_color()
        display = color.copy()

        # 检测棋盘
        result = board_detector.detect(color)
        if result.success:
            # 画调试标记
            display = board_detector.draw_debug(color)

            # 叠加棋子状态
            board, delta = extractor.extract_from_image(color)
            if board is not None:
                warp_resized = cv2.resize(result.warped, (300, 300))
                h, w = display.shape[:2]
                display[10 : 10 + 300, w - 310 : w - 10] = warp_resized
                cv2.rectangle(display, (w - 310, 10), (w - 10, 310), (0, 255, 0), 2)

                # 终端输出
                if delta is not None:
                    r, c, stone = delta
                    stone_name = "●" if stone == BLACK else "○"
                    total = int(np.count_nonzero(board != EMPTY))
                    print(f"  新落子: ({r:2d}, {c:2d}) {stone_name}  (共 {total} 子)")

        cv2.imshow(window, display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            run_single(extractor)

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="测试感知管线")
    parser.add_argument("--mock", action="store_true", help="使用 MockCamera")
    parser.add_argument("--save", type=str, help="保存结果到指定目录")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    args = parser.parse_args()

    config = load_config()
    camera = create_camera(config, mock=args.mock)
    extractor = StateExtractor(camera=camera, config=config)

    if args.mock:
        logger.info("使用 MockCamera（模拟相机）")

    if args.interactive:
        run_interactive(extractor)
        return

    # 单次检测
    board = run_single(extractor)

    if args.save:
        save_path = Path(args.save)
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存终端棋盘
        char_map = {EMPTY: ".", BLACK: "B", WHITE: "W"}
        with open(save_path / "board.txt", "w") as f:
            for r in range(board.shape[0]):
                row = " ".join(char_map[board[r, c]] for c in range(board.shape[1]))
                f.write(row + "\n")

        # 保存图形化棋盘
        board_img = draw_board_graphics(board)
        cv2.imwrite(str(save_path / "board.png"), board_img)

        logger.info(f"结果已保存到 {save_path}/")


if __name__ == "__main__":
    main()
