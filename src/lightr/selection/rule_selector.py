"""Interpretable adaptive selector for step-aware LightReasoner."""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import SelectionDecision


DEFAULT_STEP_TYPE_THRESHOLDS = {
    "setup": 0.30,
    "calculation": 0.20,
    "transformation": 0.35,
    "verification": 0.40,
    "final_answer": 0.25,
    "other": 0.50,
}

DEFAULT_ALLOWED_TOKEN_CATEGORIES = {
    "number",
    "operator",
    "equals",
    "variable",
    "currency_or_unit",
    "final_answer_marker",
}


@dataclass(frozen=True)
class RuleSelector:
    name: str = "rule_v1"
    min_expert_confidence: float = 0.30
    max_expert_entropy: float = 1.5
    min_entropy_gap: float = 0.5
    step_type_thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_STEP_TYPE_THRESHOLDS))
    allowed_token_categories: set[str] = field(default_factory=lambda: set(DEFAULT_ALLOWED_TOKEN_CATEGORIES))
    denied_token_categories: set[str] = field(default_factory=lambda: {"punctuation", "space"})

    @classmethod
    def from_config(cls, config: dict) -> "RuleSelector":
        allowed = config.get("allowed_token_categories")
        denied = config.get("denied_token_categories")
        return cls(
            name=config.get("name", "rule_v1"),
            min_expert_confidence=float(config.get("min_expert_confidence", 0.30)),
            max_expert_entropy=float(config.get("max_expert_entropy", 1.5)),
            min_entropy_gap=float(config.get("min_entropy_gap", 0.5)),
            step_type_thresholds={
                **DEFAULT_STEP_TYPE_THRESHOLDS,
                **config.get("step_type_thresholds", {}),
            },
            allowed_token_categories=set(allowed) if allowed is not None else set(DEFAULT_ALLOWED_TOKEN_CATEGORIES),
            denied_token_categories=set(denied) if denied is not None else {"punctuation", "space"},
        )

    def decide(self, candidate: dict) -> SelectionDecision:
        distribution = candidate["distribution_features"]
        token_category = candidate["token_features"]["token_category"]
        step_type = candidate["step_features"]["step_type"]
        kl_value = float(distribution["kl_expert_amateur"])
        threshold = float(self.step_type_thresholds.get(step_type, self.step_type_thresholds["other"]))

        if token_category in self.denied_token_categories:
            return SelectionDecision(False, kl_value, f"denied_token_category:{token_category}")
        if self.allowed_token_categories and token_category not in self.allowed_token_categories:
            return SelectionDecision(False, kl_value, f"token_category_not_allowed:{token_category}")
        if float(distribution["expert_top1_prob"]) < self.min_expert_confidence:
            return SelectionDecision(False, kl_value, "expert_confidence_too_low")
        if float(distribution["expert_entropy"]) > self.max_expert_entropy:
            return SelectionDecision(False, kl_value, "expert_entropy_too_high")
        if float(distribution["entropy_gap"]) < self.min_entropy_gap:
            return SelectionDecision(False, kl_value, "entropy_gap_too_low")
        if kl_value < threshold:
            return SelectionDecision(False, kl_value, f"kl_below_step_threshold:{step_type}")
        return SelectionDecision(True, kl_value, f"selected:{step_type}:{token_category}")

