"""Selector base types and shared helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SelectionDecision:
    selected: bool
    score: float | None = None
    reason: str = ""


class CandidateSelector(Protocol):
    name: str

    def decide(self, candidate: dict) -> SelectionDecision:
        ...

