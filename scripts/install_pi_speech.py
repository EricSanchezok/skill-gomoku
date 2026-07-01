#!/usr/bin/env python3
"""Install and test local Chinese speech support on Raspberry Pi OS/Debian."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.interaction import ConsoleRobotInteraction, _speech_command  # noqa: E402

APT_PACKAGES = ("espeak-ng", "speech-dispatcher", "alsa-utils", "mpg123")
TEST_TEXT = "树莓派中文语音已经准备好了。现在开始放狠话。"


def main() -> int:
    if not _has_apt():
        print("This installer expects Raspberry Pi OS/Debian with apt-get.")
        return 1

    runner = _sudo_prefix()
    if runner is None:
        print("Need root or sudo to install speech packages.")
        return 1

    subprocess.run([*runner, "apt-get", "update"], check=True)
    subprocess.run([*runner, "apt-get", "install", "-y", *APT_PACKAGES], check=True)

    command = _speech_command(TEST_TEXT)
    print("speech command:", command[0] if command else "not found")
    if command is None:
        return 1
    ConsoleRobotInteraction().speak(TEST_TEXT)
    return 0


def _has_apt() -> bool:
    return shutil.which("apt-get") is not None


def _sudo_prefix() -> list[str] | None:
    if os.geteuid() == 0:
        return []
    sudo = shutil.which("sudo")
    return [sudo] if sudo is not None else None


if __name__ == "__main__":
    raise SystemExit(main())
