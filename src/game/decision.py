"""Shared AI decision objects."""

from __future__ import annotations

from dataclasses import dataclass


class AIMoveError(RuntimeError):
    """Raised when the configured AI cannot produce a legal Gomoku move."""


class AIRefusedMoveError(AIMoveError):
    """Raised when an AI explicitly chooses not to play."""


@dataclass(frozen=True)
class AIDecision:
    """One AI turn decision in play-area local coordinates."""

    row: int
    col: int
    should_play: bool = True
    use_skill: bool = False
    skill_row: int | None = None
    skill_col: int | None = None
    trash_talk: str = ""
    rationale: str = ""
    source: str = "rapfi"
