# Skill-Gomoku — 机械臂下五子棋

基于 **Webcam 视觉 + Rapfi AI 引擎 + SO-ARM101 机械臂** 全自动五子棋对弈系统。

```
┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────┐
│  RealSense │ → │ BoardDetector │ → │ StoneDetector│ → │  Rapfi   │
│  (UVC RGB) │    │ 标定四角+透视  │    │ HoughCircles│    │  AI 决策  │
└──────────┘    └──────────────┘    └───────────┘    └──────────┘
                                                           │
                                                    (row, col)
                                                           │
                                                    ┌──────▼──────┐
                                                    │ 机械臂执行    │
                                                    │ (队友实现)   │
                                                    └─────────────┘
```

## 项目结构

```
skill-gomoku/
├── bin/
│   └── rapfi/
│       ├── macos-arm64/rapfi      # Apple Silicon macOS Rapfi
│       └── linux-aarch64/         # Raspberry Pi Rapfi 放这里
├── config/
│   └── default.yaml         # 标定角点、Hough 参数、时间控制
├── calibration/
│   └── lerobot/             # 随仓库携带的 SO101 LeRobot 电机标定
├── scripts/
│   ├── install_lerobot_calibration.py # 恢复 LeRobot 本地标定缓存
│   ├── calibrate_board.py   # 交互式棋盘四角标定
│   ├── calibrate_robot_board.py # 手带机械臂记录棋盘四角
│   ├── run_live_game.py     # 真机/半真机完整对局入口
│   └── test_perception.py   # 感知管线测试（mock / 真实相机）
├── src/
│   ├── perception/
│   │   ├── camera.py        # WebCamera (OpenCV VideoCapture)
│   │   ├── board_detector.py # 棋盘检测（manual 标定 / auto 检测）
│   │   ├── stone_detector.py # 棋子检测（HoughCircles + 对比度分类）
│   │   ├── grid_mapper.py   # 像素 → 15×15 落子交叉点映射
│   │   └── state_extractor.py # 完整管线入口 + 差异检测
│   ├── robot/
│   │   ├── controller.py    # 棋盘落子点 → 机械臂姿态映射
│   │   ├── calibration.py   # 手带机械臂四角标定
│   │   ├── so101_adapter.py # SO101 当前姿态读取
│   │   └── so101_mover.py   # 已验证的 SO101 平滑移动工具
│   ├── game/
│   │   ├── board.py         # 15×15 棋盘状态 & 胜负判定
│   │   └── ai.py            # Rapfi 引擎子进程封装 (Gomocup 协议)
│   ├── interaction.py       # 人类确认落子 + 机器人说话/跳舞/技能钩子
│   ├── orchestrator.py      # 主流程编排
│   └── utils/               # 常量、配置加载、可视化
├── tests/
│   └── fixtures/            # 测试用图片
└── docs/
```

## 核心算法

### 棋盘检测

**手动标定模式（默认）**：相机与棋盘固定，标定一次，永久复用。

1. 运行 `calibrate_board.py`，依次点击棋盘最外侧四个落子交叉点
2. 预览 15 条横线/15 条竖线对齐效果，确认后存入 `config/default.yaml`
3. 后续每帧直接使用预存角点做透视矫正 → 600×600 俯视图

也支持 `auto` 模式：自适应阈值 + Hough 线聚类自动检测，用于相机或棋盘可能移动的场景。

### 棋子检测

**HoughCircles + 局部对比度分类**：不受棋盘颜色和光照影响。

1. 在俯视图上检测所有近圆形轮廓（棋子）
2. 对每个候选圆取内圈 vs 背景环的灰度对比度
3. 内圈亮度显著高于背景 → 白子；否则 → 黑子
4. 将圆形分配到最近的 15×15 落子交叉点

注意这里的“15×15”指 15 行 × 15 列落子点，不是 15×15 个物理方格。
五子棋棋盘的方格间隔是 14×14。

### AI 决策

**Rapfi 引擎**（Gomocup 2024 冠军），通过 Gomocup 文本协议子进程通信。

- ARM64 + NEON 原生编译，Apple Silicon 优化
- 每步思考时间可配置（默认 3000ms）
- 通过 `BOARD` 命令编码 15×15 状态，接收 `x,y` 落子回复
- 默认按平台选择 `bin/rapfi/<platform>/rapfi`；也可以用
  `game.ai.engine_path` 显式指定

### 对局节奏与人类确认

`game.robot_stone` 明确机器人执黑还是执白。黑棋按五子棋规则先手：

```yaml
game:
  robot_stone: black  # black=机器人先手；white=人类先手
```

旧字段 `game.my_stone` 仍兼容，但新代码优先使用 `robot_stone`。
主流程里可以通过 `orchestrator.is_robot_turn()` / `is_human_turn()`
判断当前该谁下。人类回合不会盲等相机变化，而是先等确认事件：
默认键盘约定是 `Enter/Space` = “我下完了”，然后拍照识别新增棋子。
同时预留了 `s` 说话、`d` 跳舞、`g` 调用 `skill-gomoku`、`q` 退出。

机器人侧的人机交互接口集中在 `src/interaction.py`：

```python
orchestrator.robot_say("轮到我了")
orchestrator.robot_dance("win")
orchestrator.robot_use_skill_gomoku({"phase": "opening"})
```

默认实现只是 console/no-op 占位；后续接语音、动作库或外部 skill 时替换
`RobotInteractionController` 即可，不需要改棋局核心。

## 快速开始

### 环境要求

- Python ≥ 3.10
- macOS 或 Linux
- OpenCV（系统级或 pip 安装）

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/EricSanchezok/skill-gomoku.git
cd skill-gomoku

# 2. 创建虚拟环境（uv 管理）
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 3. Rapfi 分平台存放在 bin/rapfi/
#    macOS Apple Silicon: bin/rapfi/macos-arm64/rapfi
#    Raspberry Pi:       bin/rapfi/linux-aarch64/rapfi
#    如需重新编译：
#    brew install ninja cmake
#    git clone https://github.com/dhbloo/rapfi.git
#    cd Rapfi && cmake --preset arm64-clang-NEON -DCMAKE_BUILD_TYPE=Release
#    cmake --build build/arm64-clang-NEON
```

树莓派队友重新编译好的 Rapfi 放到：

```text
bin/rapfi/linux-aarch64/rapfi
```

并确保它可执行：

```bash
chmod +x bin/rapfi/linux-aarch64/rapfi
```

### 安装 SO101 的 LeRobot 标定

仓库内已携带 `gomoku_so101` 实测使用的 LeRobot 电机标定：

```text
calibration/lerobot/robots/so_follower/so101_follower_0610.json
```

在新机器 clone 后可以显式安装一次：

```bash
python scripts/install_lerobot_calibration.py
```

它会复制到 LeRobot 默认读取的位置：

```text
~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower_0610.json
```

`SO101SmoothMover` 和 `SO101PoseSampler` 初始化时也会自动补齐这份缓存。默认不会覆盖本机已有同名标定；如果确认要恢复仓库版本，运行：

```bash
python scripts/install_lerobot_calibration.py --overwrite
```

### 标定棋盘（首次使用必须）

```bash
source .venv/bin/activate
python scripts/calibrate_board.py
```

按提示依次点击最外侧四个落子交叉点：**左上 → 右上 → 右下 → 左下**，
预览 warp 对齐效果后按 `Y` 保存。预览线是 15 条横线/15 条竖线，
中间间隔是 14×14 个物理方格。
主流程在 `board_detection.method: manual` 时会强制检查
`board.calibration.corners`，没有录入棋框位置会直接报错并提示先运行这个脚本。

### 使用 81 个实测姿态映射机械臂落点

当前机械臂落点不再依赖四角双线性插值。SO101 直接加载
`so101_board_81_positions.json` 里的实测姿态表，把抽象棋位 `(row, col)`
映射到对应的 LeRobot action。

注意：当前随仓库的 `so101_board_81_positions.json` 覆盖的是中心 9×9 的 81 个点。
这是默认真机对局范围：相机仍识别完整 15×15 落子交叉点，主流程只把中心 9×9
活动窗口交给 AI 分析和机械臂执行。Rapfi 会按 9×9 局部棋局运行，返回的局部坐标
再映射回 15×15 棋盘中的真实交叉点。

开局前可以先让机械臂依次恢复四个角的实测位置，用这四个角来摆正/定位棋盘：

```bash
conda run -n lerobot python scripts/replay_robot_corners.py \
  --config config/default.yaml \
  --port /dev/ttyACM0 \
  --robot-id so101_follower_0610
```

确认棋盘位置后，再运行相机棋盘标定：

```bash
source .venv/bin/activate
python scripts/calibrate_board.py
```

旧的四角手动标定脚本仍保留为兼容路径，但它只适合作为没有实测姿态表时的备用方案：

```bash
python scripts/calibrate_robot_board.py --backend input
```

在代码里，主流程会优先读取 `robot.pose_map.path` 指向的实测姿态表：

```python
target_action = orchestrator.execute_my_move(row, col)
```

### 接入吸棋气泵

气泵控制已接入主流程，但默认关闭，避免在没有 Raspberry Pi GPIO 的开发机上误初始化硬件。
真机启用时在 `config/default.yaml` 里设置：

```yaml
robot:
  # Legacy fallback. Prefer pickup_poses.black / pickup_poses.white below.
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

落子动作顺序是：按机器人棋色移动到 `pickup_poses.black` 或 `pickup_poses.white`
（没有配置时退回旧的 `pickup_pose`）→ 开电磁阀并开气泵吸棋子 →
移动到 `waiting_pose` → 移动到目标棋位 → 关闭电磁阀落子 → 关闭气泵降噪 →
回到 `waiting_pose`。之后才进入人类下棋、确认、视觉识别和 AI 分析，避免机械臂挡住相机。

录黑/白两个取子位：

```bash
conda run -n lerobot python scripts/record_pickup_poses.py --port /dev/ttyACM0
```

机器人控制、SO101 平滑移动工具和安全检查见 `docs/robot.md`。

### 跑完整对局入口

先用 dry-run 跑通相机、AI、键盘确认，不移动机械臂：

```bash
python scripts/run_live_game.py \
  --dry-run-robot \
  --disable-air-pump
```

在树莓派上真机运行时，确认 `robot.port`、`robot.pickup_poses.black/white`、
`robot.waiting_pose`、气泵 GPIO、相机角点和 Rapfi Linux 二进制都准备好，再运行：

```bash
python scripts/run_live_game.py \
  --enable-air-pump \
  --port /dev/ttyACM0
```

真机默认是安全 bring-up：最多跑 1 个 turn，每次 SO101 移动前都需要人工确认。
确认棋盒、棋盘和路径都没问题后，再显式跑完整对局：

```bash
python scripts/run_live_game.py \
  --enable-air-pump \
  --full-game \
  --port /dev/ttyACM0
```

常用调试参数：

```bash
# 相机 mock + 不动机械臂，只验证主循环能启动
python scripts/run_live_game.py --mock-camera --dry-run-robot --disable-air-pump

# 使用假 GPIO 测气泵流程
python scripts/run_live_game.py --dry-run-robot --enable-air-pump --dry-run-air-pump

# 显式指定树莓派上的 Rapfi
python scripts/run_live_game.py --engine-path bin/rapfi/linux-aarch64/rapfi

# 已确认路径安全后，关闭每段机械臂移动前的确认
python scripts/run_live_game.py --enable-air-pump --full-game --no-confirm-robot-moves
```

### 测试感知管线

```bash
# 用 MockCamera 测试（不需要硬件）
python scripts/test_perception.py --mock

# 用真实相机测试
python scripts/test_perception.py

# 交互模式
python scripts/test_perception.py --interactive
```

### 配置

`config/default.yaml`：

```yaml
game:
  robot_stone: black   # 机器人棋色；black 先手，white 后手
  play_area:
    rows: 9
    cols: 9
    row_offset: 3
    col_offset: 3
  ai:
    engine_path: null   # null=按平台自动选择 bin/rapfi/<platform>/rapfi
    board_size: null    # null=从 play_area 推导；默认即 9x9
    time_per_move_ms: 3000

camera:
  device_index: 4        # 相机编号（0=棋盘视角）

board_detection:
  method: manual         # manual=标定模式  auto=自动检测

stone_detection:
  hough_min_radius: 10   # 棋子最小半径（像素）
  hough_max_radius: 22   # 棋子最大半径（像素）
  white_inner_mean: 75   # 白子内圈亮度阈值
  white_contrast: 10     # 白子背景对比度阈值

interaction:
  human_done:
    mode: keyboard
    confirm_keys: [enter, space]
    speak_key: s
    dance_key: d
    skill_gomoku_key: g
    quit_key: q
```

AI 思考时间在 `src/game/ai.py` 中修改 `time_per_move_ms` 参数。

## 完成状态

| 模块       | 状态 | 说明                         |
| ---------- | ---- | ---------------------------- |
| 相机驱动   | ✅   | OpenCV UVC webcam            |
| 棋盘标定   | ✅   | 交互式四点标定工具           |
| 棋盘检测   | ✅   | 手动标定 + auto 备选         |
| 棋子检测   | ✅   | HoughCircles + 对比度分类    |
| AI 决策    | ✅   | Rapfi 引擎（Gomocup 冠军级） |
| 机械臂控制 | ✅   | SO101 平滑移动 + 姿态表路径  |
| 气泵吸棋   | ✅   | 电磁阀/气泵流程已接主流程    |
| 主流程编排 | ✅   | `scripts/run_live_game.py`   |
