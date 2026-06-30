"""OpenRouter-backed Gomoku move selection."""

from __future__ import annotations

import json
import logging
import os
import re
import time
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

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenRouterSettings:
    """Runtime settings for OpenRouter move selection."""

    key_path: Path
    model: str = DEFAULT_OPENROUTER_MODEL
    strength: str = "medium"
    endpoint: str = DEFAULT_OPENROUTER_ENDPOINT
    timeout_seconds: float = 45.0
    temperature: float = 0.7
    max_tokens: int = 800
    max_retries: int = 2
    retry_delay_seconds: float = 0.5
    trash_talk_enabled: bool = False
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
    attempts = max(1, int(settings.max_retries) + 1)
    payload = _chat_payload(board, my_stone, settings)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        logger.info(
            "OpenRouter move request %d/%d: model=%s board=%s stones black=%d white=%d",
            attempt,
            attempts,
            settings.model,
            f"{board.shape[0]}x{board.shape[1]}",
            int(np.count_nonzero(board == BLACK)),
            int(np.count_nonzero(board == WHITE)),
        )
        try:
            data = _request_chat_completion(payload, api_key=api_key, settings=settings)
            decision = _parse_decision(_message_content(data), board, my_stone)
            logger.info(
                "OpenRouter move response %d/%d: local_1based=(%d,%d) skill=%s "
                "skill_target=%s talk=%s",
                attempt,
                attempts,
                decision.row + 1,
                decision.col + 1,
                decision.use_skill,
                (
                    None
                    if decision.skill_row is None or decision.skill_col is None
                    else (decision.skill_row + 1, decision.skill_col + 1)
                ),
                bool(decision.trash_talk),
            )
            return decision
        except (AIMoveError, ValueError, TimeoutError) as exc:
            last_error = exc
            logger.warning(
                "OpenRouter move attempt %d/%d failed: %s: %s",
                attempt,
                attempts,
                type(exc).__name__,
                exc,
            )
            if attempt < attempts:
                time.sleep(max(0.0, float(settings.retry_delay_seconds)))

    raise AIMoveError(f"OpenRouter failed after {attempts} attempts: {last_error}") from last_error


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
            openrouter_cfg.get("timeout_seconds", ai_cfg.get("timeout_seconds", 45.0))
        ),
        temperature=float(openrouter_cfg.get("temperature", ai_cfg.get("temperature", 0.7))),
        max_tokens=int(openrouter_cfg.get("max_tokens", ai_cfg.get("max_tokens", 800))),
        max_retries=int(openrouter_cfg.get("max_retries", ai_cfg.get("max_retries", 2))),
        retry_delay_seconds=float(
            openrouter_cfg.get("retry_delay_seconds", ai_cfg.get("retry_delay_seconds", 0.5))
        ),
        trash_talk_enabled=bool(
            openrouter_cfg.get(
                "trash_talk_enabled",
                ai_cfg.get("trash_talk_enabled", game_cfg.get("trash_talk_enabled", False)),
            )
        ),
    )


def _chat_payload(
    board: np.ndarray,
    my_stone: int,
    settings: OpenRouterSettings,
) -> dict[str, Any]:
    return {
        "model": settings.model,
        "messages": _messages(
            board,
            my_stone,
            settings.strength,
            trash_talk_enabled=settings.trash_talk_enabled,
        ),
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }


def _request_chat_completion(
    payload: Mapping[str, Any],
    *,
    api_key: str,
    settings: OpenRouterSettings,
) -> Mapping[str, Any]:
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

    if not isinstance(data, Mapping):
        raise AIMoveError(f"OpenRouter response is not a JSON object: {data!r}")
    return data


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


def _messages(
    board: np.ndarray,
    my_stone: int,
    strength: str,
    *,
    trash_talk_enabled: bool = False,
) -> list[dict[str, str]]:
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
            (
                "你有一个不讲武德的技能，但开启条件极端严格：只有当对手下一手"
                "已经存在直接成五的落点、你马上要输时，才可以 use_skill=true，"
                f"吸走一个{opponent_name}并放回棋盒。"
                "如果使用技能，skill_row/skill_col 必须指向一个对手已有棋子。"
                "其他任何局面都必须 use_skill=false。"
            ),
            (
                "垃圾话已开启：trash_talk 必须是一句短促、有节目效果的嘲讽。"
                if trash_talk_enabled
                else "垃圾话未开启：trash_talk 必须为空字符串。"
            ),
            "只能返回这个 JSON：",
            (
                '{"should_play": true, "row": 1, "col": 1, '
                '"use_skill": false, "skill_row": null, "skill_col": null, '
                '"trash_talk": "短句，可为空", "rationale": "短句"}'
            ),
            "如果还有合法位置，should_play 必须为 true；row/col 必须在 1 到 "
            f"{size} 之间。",
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_decision(content: str, board: np.ndarray, my_stone: int = BLACK) -> AIDecision:
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
    opponent_stone = WHITE if my_stone == BLACK else BLACK
    skill_row, skill_col = _parse_skill_target(obj, board, opponent_stone) if use_skill else (
        None,
        None,
    )
    return AIDecision(
        row=row,
        col=col,
        should_play=should_play,
        use_skill=use_skill,
        skill_row=skill_row,
        skill_col=skill_col,
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


def _parse_skill_target(
    obj: Mapping[str, Any],
    board: np.ndarray,
    opponent_stone: int,
) -> tuple[int | None, int | None]:
    if not np.any(board == opponent_stone):
        raise AIMoveError("OpenRouter LLM requested skill, but there is no opponent stone")

    values: tuple[Any, Any] | None = None
    if "skill_target" in obj and isinstance(obj["skill_target"], list):
        target = obj["skill_target"]
        if len(target) == 2:
            values = (target[0], target[1])
    elif "skill_row" in obj and "skill_col" in obj:
        if obj["skill_row"] is not None and obj["skill_col"] is not None:
            values = (obj["skill_row"], obj["skill_col"])
    elif "remove_row" in obj and "remove_col" in obj:
        values = (obj["remove_row"], obj["remove_col"])

    if values is None:
        return None, None

    row = int(values[0]) - 1
    col = int(values[1]) - 1
    rows, cols = board.shape
    if not (0 <= row < rows and 0 <= col < cols):
        raise AIMoveError(
            f"OpenRouter LLM chose skill target local 1-based "
            f"({row + 1}, {col + 1}) outside {rows}x{cols}"
        )
    if board[row, col] != opponent_stone:
        raise AIMoveError(
            "OpenRouter LLM chose skill target that is not an opponent stone: "
            f"({row + 1}, {col + 1}); LLM board was:\n{_board_text(board)}"
        )
    return row, col


def _message_content(data: Mapping[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AIMoveError(f"OpenRouter response missing choices: {data!r}")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, Mapping) else None
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
    finish_reason = choice.get("finish_reason") if isinstance(choice, Mapping) else None
    raise AIMoveError(
        "OpenRouter response content is not text: "
        f"{content!r}; finish_reason={finish_reason!r}; message_keys={sorted(message)}"
    )


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
