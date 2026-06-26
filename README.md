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
│   └── rapfi                # Rapfi 引擎 (ARM64 NEON, ~1.8MB)
├── config/
│   └── default.yaml         # 标定角点、Hough 参数、时间控制
├── calibration/
│   └── lerobot/             # 随仓库携带的 SO101 LeRobot 电机标定
├── scripts/
│   ├── install_lerobot_calibration.py # 恢复 LeRobot 本地标定缓存
│   ├── calibrate_board.py   # 交互式棋盘四角标定
│   ├── calibrate_robot_board.py # 手带机械臂记录棋盘四角
│   └── test_perception.py   # 感知管线测试（mock / 真实相机）
├── src/
│   ├── perception/
│   │   ├── camera.py        # WebCamera (OpenCV VideoCapture)
│   │   ├── board_detector.py # 棋盘检测（manual 标定 / auto 检测）
│   │   ├── stone_detector.py # 棋子检测（HoughCircles + 对比度分类）
│   │   ├── grid_mapper.py   # 像素 → 棋盘行列映射
│   │   └── state_extractor.py # 完整管线入口 + 差异检测
│   ├── robot/
│   │   ├── controller.py    # 棋盘格 → 机械臂姿态映射
│   │   ├── calibration.py   # 手带机械臂四角标定
│   │   ├── so101_adapter.py # SO101 当前姿态读取
│   │   └── so101_mover.py   # 已验证的 SO101 平滑移动工具
│   ├── game/
│   │   ├── board.py         # 15×15 棋盘状态 & 胜负判定
│   │   └── ai.py            # Rapfi 引擎子进程封装 (Gomocup 协议)
│   ├── orchestrator.py      # 主流程编排
│   └── utils/               # 常量、配置加载、可视化
├── tests/
│   └── fixtures/            # 测试用图片
└── docs/
```

## 核心算法

### 棋盘检测

**手动标定模式（默认）**：相机与棋盘固定，标定一次，永久复用。

1. 运行 `calibrate_board.py`，依次点击棋盘四角
2. 预览格子对齐效果，确认后存入 `config/default.yaml`
3. 后续每帧直接使用预存角点做透视矫正 → 600×600 俯视图

也支持 `auto` 模式：自适应阈值 + Hough 线聚类自动检测，用于相机或棋盘可能移动的场景。

### 棋子检测

**HoughCircles + 局部对比度分类**：不受棋盘颜色和光照影响。

1. 在俯视图上检测所有近圆形轮廓（棋子）
2. 对每个候选圆取内圈 vs 背景环的灰度对比度
3. 内圈亮度显著高于背景 → 白子；否则 → 黑子
4. 将圆形分配到最近的 15×15 网格单元格

### AI 决策

**Rapfi 引擎**（Gomocup 2024 冠军），通过 Gomocup 文本协议子进程通信。

- ARM64 + NEON 原生编译，Apple Silicon 优化
- 每步思考时间可配置（默认 3000ms）
- 通过 `BOARD` 命令编码 15×15 状态，接收 `x,y` 落子回复

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

# 3. Rapfi 已预编译在 bin/rapfi，无需额外安装
#    如需重新编译：
#    brew install ninja cmake
#    git clone https://github.com/dhbloo/rapfi.git
#    cd Rapfi && cmake --preset arm64-clang-NEON -DCMAKE_BUILD_TYPE=Release
#    cmake --build build/arm64-clang-NEON
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

按提示依次点击棋盘四个角：**左上 → 右上 → 右下 → 左下**，预览 warp 对齐效果后按 `Y` 保存。

### 使用 81 个实测姿态映射机械臂落点

当前机械臂落点不再依赖四角双线性插值。SO101 直接加载
`so101_board_81_positions.json` 里的实测姿态表，把抽象棋位 `(row, col)`
映射到对应的 LeRobot action。

开局前可以先让机械臂依次恢复四个角的实测位置，用这四个角来摆正/定位棋盘：

```bash
conda run -n lerobot python scripts/replay_robot_corners.py \
  --config config/default.yaml \
  --port /dev/tty.usbmodem5A4B0487101 \
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

机器人控制、SO101 平滑移动工具和安全检查见 `docs/robot.md`。

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
camera:
  device_index: 4        # 相机编号（0=棋盘视角）

board_detection:
  method: manual         # manual=标定模式  auto=自动检测

stone_detection:
  hough_min_radius: 10   # 棋子最小半径（像素）
  hough_max_radius: 22   # 棋子最大半径（像素）
  white_inner_mean: 75   # 白子内圈亮度阈值
  white_contrast: 10     # 白子背景对比度阈值
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
| 机械臂控制 | 🔧   | 队友开发中                   |
| 主流程编排 | 🏗️ | 骨架完成，等待机械臂集成     |
