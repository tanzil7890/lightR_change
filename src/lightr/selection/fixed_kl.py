"""Fixed-KL selector matching the original LightReasoner decision rule."""

from __future__ import annotations

from dataclasses import dataclass

from .base import SelectionDecision


@dataclass(frozen=True)
class FixedKLSelector:
    beta: float = 0.4
    name: str = "fixed_kl"

    def decide(self, candidate: dict) -> SelectionDecision:
        kl_value = float(candidate["distribution_features"]["kl_expert_amateur"])
        selected = kl_value >= self.beta
        reason = "kl_above_beta" if selected else "kl_below_beta"
        return SelectionDecision(selected=selected, score=kl_value, reason=reason)

