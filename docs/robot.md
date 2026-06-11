# 机器人相关说明

这份文档记录五子棋项目里 SO101 机械臂相关的代码、标定流程和安全习惯。核心原则是：主对弈流程保持硬件无关，但把 `gomoku_so101` 里已经验证过的机械臂控制经验沉淀进主仓库，方便队友复现。

## 文件分工

- `src/robot/controller.py`：把棋盘 `(row, col)` 映射成机械臂目标姿态。
- `src/robot/calibration.py`：手带机械臂记录棋盘四角的通用标定流程。
- `src/robot/so101_adapter.py`：标定时读取 SO101 当前 LeRobot action 姿态。
- `src/robot/so101_mover.py`：从本地测试工具提炼出的 SO101 平滑移动封装。
- `src/robot/tools/so101_move_demo.py`：移动到 `center` / `waiting` 预设的小工具。
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

每局开始前，可以手带机械臂末端依次对准棋盘左上、右上、右下、左下四个角。系统记录这四个角的 action 姿态，然后通过双线性插值，把任意 `(row, col)` 转成同一组关节 key 的目标 action。

## 开局前标定棋盘四角

棋盘位置可能变化时，开局前运行：

```bash
conda run -n lerobot python scripts/calibrate_robot_board.py \
  --backend so101 \
  --config config/default.yaml \
  --port /dev/tty.usbmodem5A4B0487101 \
  --robot-id so101_follower_0610
```

脚本会先释放力矩，方便手带机械臂。记录顺序固定为：

1. 左上
2. 右上
3. 右下
4. 左下

结果会写入 `config/default.yaml` 的 `robot.calibration.corners`。没有硬件时可以用终端输入模式测试配置写入：

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

开局时可以把手动标定接到 orchestrator：

```python
orchestrator.start_new_game(
    sampler=sampler,
    calibrate_robot=True,
    config_path="config/default.yaml",
)
```

我方落子时，`GameOrchestrator.execute_my_move(row, col)` 会先算出插值后的目标 action。下一步集成时，把这个 target 交给 `SO101SmoothMover.move_to()`，再触发末端落子机构即可。

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
- 标定能加载但落点不准：重新按左上、右上、右下、左下顺序记录四角。
- 本地提交后推送失败：先 `gh auth login`，再 push feature 分支。
