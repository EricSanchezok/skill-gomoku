"""Configuration loader — reads YAML config and merges defaults."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config/default.yaml") -> dict[str, Any]:
    """Load YAML config file.

    Args:
        path: Path to config yaml file.

    Returns:
        Parsed config dictionary.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f) or {}
