"""OpenRouter-backed Gomoku move selection."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.game.decision import AIDecision, AIMoveError, AIRefusedMoveError
from src.utils.constants import BLACK, EMPTY, WHITE

DEFAULT_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4-flash"


@dataclass(frozen=True)
class OpenRouterSettings:
    """Runtime settings for OpenRouter move selection."""

    key_path: Path
    model: str = DEFAULT_OPENROUTER_MODEL
    strength: str = "medium"
    endpoint: str = DEFAULT_OPENROUTER_ENDPOINT
    timeout_seconds: float = 20.0
    temperature: float = 0.7
    max_tokens: int = 300
    api_key: str | None = None


def decide_with_openrouter(
    board: np.ndarray,
    my_stone: int,
    settings: OpenRouterSettings,
) -> AIDecision:
    """Ask OpenRouter for one legal move on the supplied play-area board."""

    if board.ndim != 2 or board.shape[0] != board.shape[1]:
        raise ValueError(f"LLM board must be square, got shape {board.shape}")

    api_key = settings.api_key or load_openrouter_api_key(settings.key_path)
    payload = {
        "model": settings.model,
        "messages": _messages(board, my_stone, settings.strength),
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        settings.endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise AIMoveError(f"OpenRouter request failed: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise AIMoveError(f"OpenRouter request failed: {exc}") from exc

    content = _message_content(data)
    return _parse_decision(content, board)


def openrouter_settings_from_config(
    game_cfg: Mapping[str, Any],
    ai_cfg: Mapping[str, Any],
    *,
    base_dir: str | Path,
) -> OpenRouterSettings:
    """Parse ``game.ai.openrouter`` plus a few convenient aliases."""

    openrouter_cfg = ai_cfg.get("openrouter", {})
    if not isinstance(openrouter_cfg, Mapping):
        openrouter_cfg = {}

    key_path = _resolve_config_path(
        openrouter_cfg.get("key_path", ai_cfg.get("key_path", "config/key.yaml")),
        base_dir=base_dir,
    )
    strength = str(
        openrouter_cfg.get(
            "strength",
            ai_cfg.get("strength", ai_cfg.get("llm_strength", game_cfg.get("ai_level", "medium"))),
        )
    )
    return OpenRouterSettings(
        key_path=key_path,
        model=str(openrouter_cfg.get("model", ai_cfg.get("model", DEFAULT_OPENROUTER_MODEL))),
        strength=_normalize_strength(strength),
        endpoint=str(
            openrouter_cfg.get("endpoint", ai_cfg.get("endpoint", DEFAULT_OPENROUTER_ENDPOINT))
        ),
        timeout_seconds=float(
            openrouter_cfg.get("timeout_seconds", ai_cfg.get("timeout_seconds", 20.0))
        ),
        temperature=float(openrouter_cfg.get("temperature", ai_cfg.get("temperature", 0.7))),
        max_tokens=int(openrouter_cfg.get("max_tokens", ai_cfg.get("max_tokens", 300))),
    )


def load_openrouter_api_key(path: str | Path) -> str:
    """Load an OpenRouter API key from YAML or ``OPENROUTER_API_KEY``."""

    env_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_key:
        return env_key

    key_path = Path(path).expanduser()
    if not key_path.exists():
        raise FileNotFoundError(
            f"OpenRouter API key file not found: {key_path}. "
            "Create config/key.yaml or set OPENROUTER_API_KEY."
        )
    with key_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"OpenRouter API key file must be a YAML mapping: {key_path}")

    candidates = [
        data.get("openrouter_api_key"),
        data.get("api_key"),
        data.get("OPENROUTER_API_KEY"),
    ]
    nested = data.get("openrouter")
    if isinstance(nested, Mapping):
        candidates.extend(
            [
                nested.get("api_key"),
                nested.get("key"),
                nested.get("OPENROUTER_API_KEY"),
            ]
        )
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(
        f"No OpenRouter API key found in {key_path}. "
        "Use openrouter.api_key, openrouter_api_key, or api_key."
    )


def _messages(board: np.ndarray, my_stone: int, strength: str) -> list[dict[str, str]]:
    size = board.shape[0]
    my_name = _stone_name(my_stone)
    opponent_name = _stone_name(WHITE if my_stone == BLACK else BLACK)
    system = (
        "你是一个五子棋机器人。你必须只在给出的 9x9 表格里选择落子，"
        "row 和 col 都使用这张表的 1-based 坐标。不要使用外层 15x15 或 0-14 坐标。"
        "你不能选择已有棋子的位置。"
        "返回严格 JSON，不要 Markdown，不要额外解释。"
    )
    user = "\n".join(
        [
            f"你执{my_name}，对手执{opponent_name}。",
            f"你的实力设定：{_strength_prompt(strength)}",
            f"下面这张 {size}x{size} 表就是唯一可下棋区域。",
            "row/col 只使用表格左侧和上方的 1-based 编号。",
            "符号：.=空，B=黑棋，W=白棋。",
            _board_text(board),
            "已落子坐标，同样使用这张表的 1-based row,col：",
            _stone_coordinate_line(board, BLACK),
            _stone_coordinate_line(board, WHITE),
            "禁止选择 black/white 列表里已经出现过的坐标。",
            "只能返回这个 JSON：",
            (
                '{"should_play": true, "row": 1, "col": 1, '
                '"use_skill": false, "trash_talk": "短句，可为空", "rationale": "短句"}'
            ),
            "如果还有合法位置，should_play 必须为 true；row/col 必须在 1 到 "
            f"{size} 之间。",
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_decision(content: str, board: np.ndarray) -> AIDecision:
    obj = _load_json_object(content)
    should_play = _as_bool(obj.get("should_play", True))
    use_skill = _as_bool(obj.get("use_skill", obj.get("cheat", False)))
    trash_talk = _short_text(obj.get("trash_talk", obj.get("taunt", "")), limit=120)
    rationale = _short_text(obj.get("rationale", obj.get("reason", "")), limit=240)

    if not should_play:
        raise AIRefusedMoveError("OpenRouter LLM chose not to play.")

    row_value, col_value = _row_col_values(obj)
    row = int(row_value) - 1
    col = int(col_value) - 1
    rows, cols = board.shape
    if not (0 <= row < rows and 0 <= col < cols):
        raise AIMoveError(
            f"OpenRouter LLM chose play-area local 1-based "
            f"({row + 1}, {col + 1}) outside {rows}x{cols}"
        )
    if board[row, col] != EMPTY:
        raise AIMoveError(
            "OpenRouter LLM chose occupied play-area local 1-based cell "
            f"({row + 1}, {col + 1}); LLM board was:\n{_board_text(board)}"
        )
    return AIDecision(
        row=row,
        col=col,
        should_play=should_play,
        use_skill=use_skill,
        trash_talk=trash_talk,
        rationale=rationale,
        source="openrouter",
    )


def _load_json_object(content: str) -> Mapping[str, Any]:
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match is None:
            raise AIMoveError(f"OpenRouter LLM did not return JSON: {content!r}") from None
        obj = json.loads(match.group(0))
    if not isinstance(obj, Mapping):
        raise AIMoveError(f"OpenRouter LLM returned non-object JSON: {obj!r}")
    return obj


def _row_col_values(obj: Mapping[str, Any]) -> tuple[Any, Any]:
    if "move" in obj and isinstance(obj["move"], list) and len(obj["move"]) == 2:
        return obj["move"][0], obj["move"][1]
    if "row" in obj and "col" in obj:
        return obj["row"], obj["col"]
    if "r" in obj and "c" in obj:
        return obj["r"], obj["c"]
    raise AIMoveError(f"OpenRouter LLM response is missing row/col: {obj!r}")


def _message_content(data: Mapping[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AIMoveError(f"OpenRouter response missing choices: {data!r}")
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    if not isinstance(message, Mapping):
        raise AIMoveError(f"OpenRouter response missing message: {data!r}")
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text", item.get("content", ""))
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    raise AIMoveError(f"OpenRouter response content is not text: {content!r}")


def _board_text(board: np.ndarray) -> str:
    rows = ["    " + " ".join(str(i) for i in range(1, board.shape[1] + 1))]
    for r in range(board.shape[0]):
        cells = []
        for c in range(board.shape[1]):
            value = int(board[r, c])
            cells.append("." if value == EMPTY else "B" if value == BLACK else "W")
        rows.append(f"{r + 1:2d}: " + " ".join(cells))
    return "\n".join(rows)


def _stone_coordinate_line(board: np.ndarray, stone: int) -> str:
    coords = [
        f"({int(row) + 1}, {int(col) + 1})"
        for row, col in zip(*np.where(board == stone), strict=True)
    ]
    name = "black" if stone == BLACK else "white"
    return f"{name}({len(coords)}): {', '.join(coords) if coords else 'none'}"


def _stone_name(stone: int) -> str:
    if stone == BLACK:
        return "黑棋"
    if stone == WHITE:
        return "白棋"
    raise ValueError(f"Unknown stone: {stone}")


def _strength_prompt(value: str) -> str:
    if value == "garbage":
        return "垃圾棋手。可以嘴硬，可以下得离谱，但必须选择合法空位。"
    if value == "strong":
        return "强力棋手。优先成五，其次阻止对手成五，再考虑活四、冲四、活三。"
    return "中等棋手。认真寻找攻防平衡，别太离谱。"


def _normalize_strength(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"garbage", "bad", "weak", "trash", "垃圾", "菜", "菜鸡"}:
        return "garbage"
    if normalized in {"strong", "hard", "expert", "powerful", "强", "强力", "高手"}:
        return "strong"
    return "medium"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "是", "要"}
    return bool(value)


def _short_text(value: Any, *, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().replace("\n", " ")[:limit]


def _resolve_config_path(value: Any, *, base_dir: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return Path(base_dir) / path
