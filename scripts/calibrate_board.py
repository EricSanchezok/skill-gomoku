#!/usr/bin/env python3
"""棋盘四角标定工具。

打开相机，请用户依次点击棋盘最外侧四个落子交叉点（左上→右上→右下→左下），
预览透视矫正效果，确认后把角点坐标写入配置文件。

用法:
    source .venv/bin/activate && python scripts/calibrate_board.py

操作:
    点击 4 次选择最外侧四个落子交叉点 → 按 'y' 确认 → 自动保存到 config/default.yaml
    按 'r' 重选，按 'q' 退出。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import yaml

from src.perception.camera import WebCamera
from src.utils.config_loader import load_config

WINDOW = "Board Calibration"
CORNER_LABELS = ["top_left", "top_right", "bottom_right", "bottom_left"]
DST_SIZE = 600
WARP_OFFSET = 680  # warp 预览窗口在原图右侧的 x 偏移


def draw_cross(
    image: np.ndarray, pt: tuple[int, int], color: tuple[int, int, int], size: int = 20
) -> None:
    """画十字标记."""
    x, y = pt
    cv2.line(image, (x - size, y), (x + size, y), color, 2)
    cv2.line(image, (x, y - size), (x, y + size), color, 2)
    cv2.circle(image, pt, 5, color, -1)


def run_calibration(config_path: str) -> None:
    """交互式标定主函数."""
    config = load_config(config_path)

    cam_cfg = config.get("camera", {})
    device = int(cam_cfg.get("device_index", 0))
    width = int(cam_cfg.get("width", 1280))
    height = int(cam_cfg.get("height", 720))

    cam = WebCamera(device_index=device, width=width, height=height)

    print("=" * 60)
    print("棋盘四角标定工具")
    print("=" * 60)
    print()
    print("请按顺序点击棋盘最外侧四个落子交叉点：")
    print("  1. 左上 (top_left)")
    print("  2. 右上 (top_right)")
    print("  3. 右下 (bottom_right)")
    print("  4. 左下 (bottom_left)")
    print()
    print("操作：点击 4 次 → 按 'y' 确认保存，'r' 重新选择，'q' 退出")
    print()

    corners: list[tuple[int, int]] = []
    confirmed = False

    def mouse_callback(event: int, x: int, y: int, _flags, _param) -> None:
        nonlocal corners
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append((x, y))
            label = CORNER_LABELS[len(corners) - 1]
            print(f"  [{len(corners)}/4] {label} = ({x}, {y})")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, mouse_callback)

    try:
        while True:
            color = cam.get_color()
            h, w = color.shape[:2]

            # 创建展示画布：原图 + 右侧 warp 预览区
            canvas = np.full((h, w + DST_SIZE, 3), 30, dtype=np.uint8)
            canvas[:, :w] = color

            # 画已选角点
            for i, pt in enumerate(corners):
                color_idx = (0, 255, 0) if len(corners) < 4 else (0, 255, 255)
                draw_cross(canvas, pt, color_idx)
                cv2.putText(
                    canvas,
                    CORNER_LABELS[i],
                    (pt[0] + 15, pt[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )

            # 连线（先选中的点之间）
            if len(corners) > 1:
                pts = np.array(corners, np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts], False, (0, 255, 0), 2)
                if len(corners) == 4:
                    cv2.polylines(canvas, [pts], True, (0, 255, 0), 2)

            # 4 个点选完后，显示 warp 预览
            if len(corners) == 4:
                try:
                    src = np.array(corners, dtype=np.float32)
                    dst = np.array(
                        [
                            [0, 0],
                            [DST_SIZE - 1, 0],
                            [DST_SIZE - 1, DST_SIZE - 1],
                            [0, DST_SIZE - 1],
                        ],
                        dtype=np.float32,
                    )
                    transform = cv2.getPerspectiveTransform(src, dst)
                    warped = cv2.warpPerspective(color, transform, (DST_SIZE, DST_SIZE))

                    # 纠正 90° 旋转（如果用户选点的顺序使 warp 旋转了）
                    # 不做自动修正，信任用户按正确顺序点击

                    canvas[0:DST_SIZE, w : w + DST_SIZE] = warped
                    cv2.rectangle(
                        canvas,
                        (w, 0),
                        (w + DST_SIZE, DST_SIZE),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        canvas,
                        "Warped Preview",
                        (w + 10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        1,
                    )

                    # 画 15×15 落子交叉点对应的 15 条线；物理方格是 14×14。
                    grid_step = (DST_SIZE - 1) / 14
                    for i in range(15):
                        pos = int(i * grid_step)
                        cv2.line(canvas, (w + pos, 0), (w + pos, DST_SIZE), (0, 255, 255), 1)
                        cv2.line(canvas, (w, pos), (w + DST_SIZE, pos), (0, 255, 255), 1)
                except cv2.error:
                    pass

            # 提示文字
            if len(corners) < 4:
                hint = f"Click corner {len(corners) + 1}/4: {CORNER_LABELS[len(corners)]}"
            elif not confirmed:
                hint = "Press Y=confirm  R=retry  Q=quit"
            else:
                hint = "Calibration saved! Press any key to exit."
            cv2.putText(
                canvas, hint, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )

            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(30) & 0xFF

            if confirmed:
                cv2.waitKey(0)
                break

            if key == ord("q"):
                print("取消标定，不保存。")
                break
            elif key == ord("r"):
                corners = []
                confirmed = False
                print("已清除所有角点，请重新选择。")
            elif key == ord("y") and len(corners) == 4:
                confirmed = True
                # 保存到配置文件
                config_path_obj = Path(config_path)
                with open(config_path_obj) as f:
                    yaml_data = yaml.safe_load(f)

                if "board" not in yaml_data:
                    yaml_data["board"] = {}

                yaml_data["board"]["calibration"] = {
                    "method": "manual",
                    "corners": {
                        "top_left": list(corners[0]),
                        "top_right": list(corners[1]),
                        "bottom_right": list(corners[2]),
                        "bottom_left": list(corners[3]),
                    },
                    "dst_size": DST_SIZE,
                }

                # 保留注释的简单写入方式
                with open(config_path_obj, "w") as f:
                    yaml.safe_dump(
                        yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False
                    )

                print(f"✓ 标定已保存到 {config_path_obj.resolve()}")
                print(f"  角点坐标: {corners}")

    finally:
        cam.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="棋盘四角标定工具")
    parser.add_argument("--config", default="config/default.yaml", help="配置文件路径")
    args = parser.parse_args()
    run_calibration(args.config)


if __name__ == "__main__":
    main()
