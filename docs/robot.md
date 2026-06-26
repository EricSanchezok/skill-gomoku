# 机器人相关说明

这份文档记录五子棋项目里 SO101 机械臂相关的代码、标定流程和安全习惯。核心原则是：主对弈流程保持硬件无关，但把 `gomoku_so101` 里已经验证过的机械臂控制经验沉淀进主仓库，方便队友复现。

## 文件分工

- `src/robot/pose_mapper.py`：加载实测姿态表，把抽象棋位直接映射成机械臂目标姿态。
- `src/robot/controller.py`：旧四角插值映射，保留作兼容/测试路径。
- `src/robot/calibration.py`：手带机械臂记录棋盘四角的通用标定流程。
- `src/robot/lerobot_calibration.py`：把仓库内携带的 LeRobot 电机标定安装到本机缓存。
- `src/robot/so101_adapter.py`：标定时读取 SO101 当前 LeRobot action 姿态。
- `src/robot/so101_mover.py`：从本地测试工具提炼出的 SO101 平滑移动封装。
- `src/robot/tools/so101_move_demo.py`：移动到 `center` / `waiting` 预设的小工具。
- `scripts/install_lerobot_calibration.py`：显式恢复 `so101_follower_0610` 的 LeRobot 标定。
- `scripts/replay_robot_corners.py`：按实测姿态表依次恢复棋盘四角。
- `scripts/calibrate_robot_board.py`：把棋盘四角标定结果写入 YAML 配置。

`gomoku_so101` 里的旧 raw tick 实验、大体量 Web UI 和 `old_scripts` 没有原样搬进主包。它们适合硬件 bring-up 和诊断，但不适合成为主对弈流程的运行依赖。

## 坐标空间

视觉模块使用 15 x 15 棋盘上的 `(row, col)` 坐标。SO101 这边使用 LeRobot action 字典，例如：

```python
{
    "shoulder_pan.pos": 3.69,
    "shoulder_lift.pos": -13.80,
    "elbow_flex.pos": 51.47,
    "wrist_flex.pos": 14.11,
    "wrist_roll.pos": 1.45,
    "gripper.pos": 36.02,
}
```

当前主流程使用实测姿态表，不再用四角双线性插值推算落点。`robot.pose_map.path`
指向的 JSON 里，每个抽象棋位都有一组录好的 action。主流程调用
`MeasuredBoardPoseMapper.target_for_cell(row, col)` 直接查表返回目标 action。

## LeRobot 电机标定

`gomoku_so101` 已验证可用的 SO101 LeRobot 电机标定随仓库存放在：

```text
calibration/lerobot/robots/so_follower/so101_follower_0610.json
```

LeRobot 默认会从下面的位置读取同名文件：

```text
~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower_0610.json
```

新机器 clone 后可以手动恢复：

```bash
python scripts/install_lerobot_calibration.py
```

`SO101SmoothMover` 和 `SO101PoseSampler` 初始化时会自动执行同样的安装检查。默认保留本机已有同名标定；确认要恢复仓库版本时加 `--overwrite`。

## 开局前恢复棋盘四角

棋盘位置可能变化时，开局前先让 SO101 依次恢复实测表中的四个角：

```bash
conda run -n lerobot python scripts/replay_robot_corners.py \
  --config config/default.yaml \
  --port /dev/tty.usbmodem5A4B0487101 \
  --robot-id so101_follower_0610
```

恢复顺序固定为：

1. 左上
2. 右上
3. 右下
4. 左下

用这四个角定位好棋盘后，再运行视觉棋盘标定 `scripts/calibrate_board.py`。
`scripts/calibrate_robot_board.py` 仍保留为旧兼容路径；没有实测姿态表时才需要它：

```bash
python scripts/calibrate_robot_board.py --backend input
```

## 平滑移动工具

`SO101SmoothMover` 是从已经实测可用的 `so101_smooth_mover.py` 提炼出来的。它保留了最重要的安全路径：

1. 只连接 motor bus。
2. 读取当前 `.pos` 姿态。
3. 先把当前姿态写回去作为 hold target。
4. 再开启力矩。
5. 用很多小步的 cubic easing action 慢慢移动到目标。
6. 只在明确需要时调用 `release()` 释放力矩。

测试移动到中心姿态：

```bash
conda run -n lerobot python -m src.robot.tools.so101_move_demo center \
  --port /dev/tty.usbmodem5A4B0487101 \
  --robot-id so101_follower_0610
```

移动到等待姿态：

```bash
conda run -n lerobot python -m src.robot.tools.so101_move_demo waiting
```

代码里可以这样用：

```python
from src.robot.so101_mover import SO101SmoothMover

mover = SO101SmoothMover(
    port="/dev/tty.usbmodem5A4B0487101",
    robot_id="so101_follower_0610",
)

try:
    mover.connect()
    mover.hold_current(target_action)
    mover.move_to(target_action)
finally:
    mover.disconnect()
```

## 接入对弈流程

`GameOrchestrator.from_config()` 会优先加载 `robot.pose_map.path` 指向的实测姿态表。
我方落子时，`GameOrchestrator.execute_my_move(row, col)` 会直接查表得到目标
action。若构造 orchestrator 时传入 `robot_mover=SO101SmoothMover(...)`，它会把这个
target 交给 `SO101SmoothMover.move_to()`，再等待后续接入末端落子机构。

## 安全检查

- 开启力矩前确认串口是当前 SO101。
- 测试时保持 `max_relative_target` 保守。
- 每次上力矩前先 hold 当前姿态，避免冲向旧目标。
- 每个新 session 的第一步移动要慢，旁边有人看着电源和机械臂。
- 手带标定后一般释放力矩，方便重新摆位。
- raw register 级脚本只适合诊断具体电机问题，不建议在真实对局时运行。

## 常见问题

- `ImportError: lerobot`：需要在控制机械臂的 LeRobot 环境里运行。
- `No '.pos' keys found`：当前 observation 不是预期的 SO101 LeRobot action 格式。
- 上力矩后机械臂突然动：检查是否先把当前姿态写成 hold target。
- 实测表能加载但落点不准：优先检查棋盘是否按四角恢复结果摆正；必要时重新记录对应棋位，而不是只改四角。
- 使用旧四角标定时落点不准：这是插值误差的常见表现，优先切换到实测姿态表。
- 本地提交后推送失败：先 `gh auth login`，再 push feature 分支。
