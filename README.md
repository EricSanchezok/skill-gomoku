# Skill-Gomoku — 机械臂下五子棋

基于 webcam RGB 视觉 + SO-ARM101 机械臂 + AI 决策的五子棋系统。RealSense 在当前方案中作为普通 UVC 摄像头使用，不依赖深度 SDK。

## 项目结构

```
skill-gomoku/
├── config/            # YAML 配置文件
├── src/
│   ├── perception/    # 相机驱动 & 棋盘检测 & 棋子识别 ← 当前开发
│   ├── robot/         # 机械臂控制 & 标定 (队友实现)
│   ├── game/          # 棋盘状态 & 规则判定 & AI 策略
│   ├── orchestrator.py  # 主流程编排
│   └── utils/         # 常量、配置加载、可视化
├── scripts/           # 可执行脚本
├── tests/             # 测试
└── docs/              # 文档
```

## 数据流

```
Webcam RGB 图像 → board_detector（棋盘定位 + 透视矫正）
               → stone_detector（棋子颜色识别）
               → StateExtractor（输出 15×15 矩阵 + 差异检测）
               → AI 决策 → 机械臂执行
```

## 依赖

```bash
pip install -r requirements.txt
```


## 使用

```bash
# 测试相机和感知管线
python scripts/test_perception.py

# 运行完整对局
python scripts/run_game.py
```
