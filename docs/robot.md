# 机器人相关说明

这份文档记录五子棋项目里 SO101 机械臂相关的代码、标定流程和安全习惯。核心原则是：主对弈流程保持硬件无关，但把 `gomoku_so101` 里已经验证过的机械臂控制经验沉淀进主仓库，方便队友复现。

## 文件分工

- `src/robot/pose_mapper.py`：加载实测姿态表，把抽象棋位直接映射成机械臂目标姿态。
- `src/robot/controller.py`：旧四角插值映射，保留作兼容/测试路径。
- `src/robot/calibration.py`：手带机械臂记录棋盘四角的通用标定流程。
- `src/robot/lerobot_calibration.py`：把仓库内携带的 LeRobot 电机标定安装到本机缓存。
- `src/robot/so101_adapter.py`：标定时读取 SO101 当前 LeRobot action 姿态。
- `src/robot/so101_mover.py`：从本地测试工具提炼出的 SO101 平滑移动封装。
- `src/robot/air_pump.py`：气泵/电磁阀吸棋控制，供主流程调用。
- `src/interaction.py`：人类确认落子，以及机器人说话、跳舞、调用 skill 的接口。
- `src/robot/tools/so101_move_demo.py`：移动到 `center` / `waiting` 预设的小工具。
- `scripts/install_lerobot_calibration.py`：显式恢复 `so101_follower_0610` 的 LeRobot 标定。
- `scripts/replay_robot_corners.py`：按实测姿态表依次恢复棋盘四角。
- `scripts/calibrate_robot_board.py`：把棋盘四角标定结果写入 YAML 配置。
- `scripts/run_live_game.py`：相机、Rapfi、键盘确认、SO101、气泵的完整对局入口。
- `bin/rapfi/`：按平台存放 Rapfi，可自动选择 macOS 或 Raspberry Pi 版本。

`gomoku_so101` 里的旧 raw tick 实验、大体量 Web UI 和 `old_scripts` 没有原样搬进主包。它们适合硬件 bring-up 和诊断，但不适合成为主对弈流程的运行依赖。

## 坐标空间

视觉模块使用 15 x 15 落子交叉点上的 `(row, col)` 坐标。这里的 15 x 15
不是物理方格数；五子棋的物理方格间隔是 14 x 14。SO101 这边使用
LeRobot action 字典，例如：

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

## Rapfi 二进制

Rapfi 按平台放在独立目录：

```text
bin/rapfi/macos-arm64/rapfi
bin/rapfi/linux-aarch64/rapfi
```

代码默认按当前平台自动选择。树莓派队友重新编译好的 Linux ARM64 版放到
`bin/rapfi/linux-aarch64/rapfi`，并确保可执行：

```bash
chmod +x bin/rapfi/linux-aarch64/rapfi
```

如果要临时使用其他路径，可以在配置里写：

```yaml
game:
  ai:
    engine_path: /path/to/rapfi
```

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
  --port /dev/ttyACM0 \
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
5. 发送最终目标 action，然后轮询 observation 等待到位。
6. 只在明确需要时调用 `release()` 释放力矩。

测试移动到中心姿态：

```bash
conda run -n lerobot python -m src.robot.tools.so101_move_demo center \
  --port /dev/ttyACM0 \
  --robot-id so101_follower_0610
```

移动到等待姿态：

```bash
conda run -n lerobot python -m src.robot.tools.so101_move_demo waiting
```

测试某一个实测棋位时，用 `scripts/move_to_board_position.py`。输入的是中心
9 x 9 pose map 的 1-based 坐标，运动顺序固定为
`lock current -> waiting_pose -> target`：

```bash
# 到 9x9 局部棋位 r5c5
python scripts/move_to_board_position.py r5c5

# 只解析目标，不连接机械臂
python scripts/move_to_board_position.py r5c5 --dry-run
```

这个测试脚本默认不设置 `max_relative_target`，会直接发送最终目标，然后按当前关节误差
轮询等待到位。

代码里可以这样用：

```python
from src.robot.so101_mover import SO101SmoothMover

mover = SO101SmoothMover(
    port="/dev/ttyACM0",
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
默认配置下，相机和棋局状态仍是 15 x 15 落子交叉点，但机器人只在中心 9 x 9
活动窗口里下棋。AI 看到的是这个 9 x 9 局部棋盘；返回的局部坐标会映射回
15 x 15 棋盘上的真实交叉点，再查 81 点实测姿态表。

完整入口脚本是：

```bash
python scripts/run_live_game.py
```

真机默认是保守 bring-up 模式：最多跑 1 个 turn，并且每一次 SO101
移动前都会打印最大关节变化摘要，等待人工按 Enter。确认完整路径、
棋盒位置和棋盘位置都安全后，才考虑加 `--full-game` 或 `--max-turns N`。如果要
关闭逐步确认，必须显式加 `--no-confirm-robot-moves`。

真机运行会强制检查当前机器人棋色对应的取子位。也就是说
`robot.pickup_poses.black` 或 `robot.pickup_poses.white` 至少要录好当前棋色；
缺失时脚本会拒绝开局，不会再从当前位置直接吸棋、乱走。

建议按这个顺序上真机：

```bash
# 1. 只跑主循环，不动机械臂/气泵
python scripts/run_live_game.py --mock-camera --dry-run-robot --disable-air-pump

# 2. 真实相机 + AI + 键盘确认，不动机械臂/气泵
python scripts/run_live_game.py --dry-run-robot --disable-air-pump

# 3. 树莓派真机，默认只跑一手且每段移动都要人工确认
python scripts/run_live_game.py --enable-air-pump --port /dev/ttyACM0

# 4. 确认安全后再跑完整对局
python scripts/run_live_game.py --enable-air-pump --full-game --port /dev/ttyACM0
```

当前仓库里的 `so101_board_81_positions.json` 是中心 9 x 9 的 81 个落点，正好对应
默认 `game.play_area`。`run_live_game.py` 会接受 9 x 9 活动窗口姿态表；如果以后要让
机器人覆盖完整 15 x 15 落子交叉点，才需要重新录入 225 个落点并调整配置。

棋色由 `game.robot_stone` 明确配置：

```yaml
game:
  robot_stone: black  # black=机器人先手；white=人类先手
  play_area:
    rows: 9
    cols: 9
    row_offset: 3
    col_offset: 3
```

黑棋永远先手。主流程用 `next_turn_stone()`、`is_robot_turn()` 和
`is_human_turn()` 判断当前该谁行动；旧的 `game.my_stone` 仍然兼容，但建议只改
`robot_stone`。

人类回合可以接 `KeyboardHumanTurnController`：人下完棋后按 `Enter/Space`，
主流程才拍照并检测新增棋子。默认还预留这些键位：

```text
Enter/Space  人类已下完
s            机器人说话
d            机器人跳舞
g            调用 skill-gomoku 钩子
q            退出等待
```

语音、动作和外部 skill 接口都挂在 `RobotInteractionController` 上。现在仓库里有
`NullRobotInteraction` 和 `ConsoleRobotInteraction` 占位实现；真机动作库接好后，
只需要替换这个 controller：

```python
orchestrator.robot_say("我来想一下")
orchestrator.robot_dance("win")
orchestrator.robot_use_skill_gomoku({"phase": "midgame"})
```

气泵吸棋也已经接入 `execute_my_move()`。启用方式：

```yaml
robot:
  # Legacy fallback. Prefer pickup_poses.black / pickup_poses.white.
  pickup_pose:
    shoulder_pan.pos: 0.0
    shoulder_lift.pos: 0.0
    elbow_flex.pos: 0.0
    wrist_flex.pos: 0.0
    wrist_roll.pos: 0.0
    gripper.pos: 0.0
  pickup_poses:
    black: null
    white: null
  waiting_pose: waiting
  air_pump:
    enabled: true
    valve_pin: 20
    pump_pin: 21
    pick_seconds: 1.0
    drop_delay_seconds: 0.05
```

真机验证后的气路逻辑是：

1. 吸棋子 / 保持吸住：打开电磁阀，开启气泵。
2. 放棋子：关闭电磁阀，棋子立刻落下。
3. 降噪：落子后关闭气泵。

所以完整落子顺序是：

1. 按机器人棋色移动到 `robot.pickup_poses.black` 或 `robot.pickup_poses.white`；如果没有配置对应棋色，退回旧的 `robot.pickup_pose`。
2. 调用气泵 `pick_stone()`，建立吸力。
3. 移动到 `robot.waiting_pose`，确保转场稳定。
4. 移动到目标棋位 action。
5. 调用气泵 `drop_stone()`，关阀落子并关泵。
6. 回到 `robot.waiting_pose`，避免挡住相机。

完整对局节奏是：抓棋子 → `waiting_pose` → 下棋的位置 → `waiting_pose` →
人类下棋 → 人类确认 → 视觉识别 → AI 分析 → 再抓棋子。

录黑/白两个取子位：

```bash
conda run -n lerobot python scripts/record_pickup_poses.py --port /dev/ttyACM0
```

## 安全检查

- 开启力矩前确认串口是当前 SO101。
- 真机完整对局如果启用 `max_relative_target`，保持取值保守。
- 每次上力矩前先 hold 当前姿态，避免冲向旧目标。
- 取子位没有录好时不要跑真机对局；`run_live_game.py` 会默认拒绝这种状态。
- 新场地第一次跑保持逐步确认，不要加 `--no-confirm-robot-moves`。
- 每个新 session 的第一步移动要慢，旁边有人看着电源和机械臂。
- 手带标定后一般释放力矩，方便重新摆位。
- raw register 级脚本只适合诊断具体电机问题，不建议在真实对局时运行。

## 常见问题

- `ImportError: lerobot`：需要在控制机械臂的 LeRobot 环境里运行。
- `No '.pos' keys found`：当前 observation 不是预期的 SO101 LeRobot action 格式。
- 上力矩后机械臂突然动：检查是否先把当前姿态写成 hold target。
- 实测表能加载但落点不准：优先检查棋盘是否按四角恢复结果摆正；必要时重新记录对应棋位，而不是只改四角。
- 使用旧四角标定时落点不准：这是插值误差的常见表现，优先切换到实测姿态表。
- 主流程提示缺少 `board.calibration.corners`：先运行 `python scripts/calibrate_board.py`
  录入相机视角下的棋框四角。
- 气泵不动作：确认 `robot.air_pump.enabled: true`，并且在树莓派环境安装了 `gpiozero`。
- 本地提交后推送失败：先 `gh auth login`，再 push feature 分支。
