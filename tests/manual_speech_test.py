#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.interaction import ConsoleRobotInteraction, _speech_command  # noqa: E402

text = "测试一下语音功能。如果你听到这句话，朗读正常。"
print("speech command:", (_speech_command(text) or ["not found"])[0])
ConsoleRobotInteraction().speak(text)
