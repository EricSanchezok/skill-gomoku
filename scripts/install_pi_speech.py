#!/usr/bin/env python3
"""Install and test local Chinese speech support for the active Conda/Pi env."""

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
CONDA_PACKAGES = ("espeak-ng", "mpg123")
TEST_TEXT = "树莓派中文语音已经准备好了。现在开始放狠话。"


def main() -> int:
    installed = False
    runner = _sudo_prefix()
    if runner is not None and _has_apt():
        subprocess.run([*runner, "apt-get", "update"], check=True)
        subprocess.run([*runner, "apt-get", "install", "-y", *APT_PACKAGES], check=True)
        installed = True
    else:
        conda = _conda_command()
        if conda is None:
            print("No sudo/apt and no conda/mamba executable found.")
            print("Manual install inside the active env:")
            print("  conda install -y -c conda-forge espeak-ng mpg123")
            return 1
        subprocess.run([*conda, "install", "-y", "-c", "conda-forge", *CONDA_PACKAGES], check=True)
        installed = True

    command = _speech_command(TEST_TEXT)
    print("speech command:", command[0] if command else "not found")
    if command is None:
        if installed:
            print("Install finished, but no speech command was found in this Python env.")
            print("Check that live game uses the same conda env as this script.")
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


def _conda_command() -> list[str] | None:
    for env_name in ("MAMBA_EXE", "CONDA_EXE"):
        value = os.environ.get(env_name)
        if value:
            return [value]
    for name in ("mamba", "micromamba", "conda"):
        executable = shutil.which(name)
        if executable is not None:
            return [executable]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
