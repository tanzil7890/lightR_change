"""Lightweight learned selector for adaptive LightReasoner candidates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import SelectionDecision


NUMERIC_FEATURES = [
    "kl_expert_amateur",
    "expert_entropy",
    "amateur_entropy",
    "entropy_gap",
    "expert_top1_prob",
    "amateur_top1_prob",
    "expert_top2_margin",
    "expert_amateur_top1_prob_gap",
    "target_expert_prob",
    "target_amateur_prob",
    "target_prob_gap",
    "normalized_position",
]

TOKEN_CATEGORIES = [
    "number",
    "operator",
    "equals",
    "variable",
    "punctuation",
    "newline",
    "word",
    "space",
    "currency_or_unit",
    "final_answer_marker",
    "other",
]

STEP_TYPES = [
    "setup",
    "calculation",
    "transformation",
    "verification",
    "final_answer",
    "other",
]

POSITION_BUCKETS = ["early", "middle", "late"]


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _position_bucket(normalized_position: float) -> str:
    if normalized_position < 0.33:
        return "early"
    if normalized_position < 0.66:
        return "middle"
    return "late"


def candidate_feature_dict(candidate: dict[str, Any]) -> dict[str, float]:
    distribution = candidate["distribution_features"]
    token_category = str(candidate.get("token_features", {}).get("token_category", "other"))
    step_type = str(candidate.get("step_features", {}).get("step_type", "other"))
    normalized_position = float(candidate.get("normalized_position", 0.0))
    values: dict[str, float] = {}

    for name in NUMERIC_FEATURES:
        if name == "normalized_position":
            raw = normalized_position
        else:
            raw = distribution.get(name)
        values[name] = 0.0 if raw is None else float(raw)

    values["top1_match"] = 1.0 if bool(distribution.get("top1_match", False)) else 0.0

    for category in TOKEN_CATEGORIES:
        values[f"token_category={category}"] = 1.0 if token_category == category else 0.0

    for current_step_type in STEP_TYPES:
        values[f"step_type={current_step_type}"] = 1.0 if step_type == current_step_type else 0.0

    position_bucket = _position_bucket(normalized_position)
    for bucket in POSITION_BUCKETS:
        values[f"position_bucket={bucket}"] = 1.0 if position_bucket == bucket else 0.0

    return values


def default_feature_names() -> list[str]:
    names = list(NUMERIC_FEATURES)
    names.append("top1_match")
    names.extend(f"token_category={category}" for category in TOKEN_CATEGORIES)
    names.extend(f"step_type={step_type}" for step_type in STEP_TYPES)
    names.extend(f"position_bucket={bucket}" for bucket in POSITION_BUCKETS)
    return names


def vectorize(candidate: dict[str, Any], feature_names: list[str]) -> list[float]:
    values = candidate_feature_dict(candidate)
    return [float(values.get(name, 0.0)) for name in feature_names]


@dataclass(frozen=True)
class LinearSelector:
    weights: list[float]
    bias: float
    feature_names: list[str]
    threshold: float = 0.5
    name: str = "learned_linear_v1"
    means: list[float] | None = None
    scales: list[float] | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> "LinearSelector":
        with Path(path).open("r", encoding="utf-8") as handle:
            artifact = json.load(handle)
        return cls(
            weights=[float(value) for value in artifact["weights"]],
            bias=float(artifact["bias"]),
            feature_names=list(artifact["feature_names"]),
            threshold=float(artifact.get("threshold", 0.5)),
            name=str(artifact.get("name", "learned_linear_v1")),
            means=artifact.get("standardization", {}).get("means"),
            scales=artifact.get("standardization", {}).get("scales"),
        )

    def score(self, candidate: dict[str, Any]) -> float:
        features = vectorize(candidate, self.feature_names)
        if self.means is not None and self.scales is not None:
            features = [
                (value - float(mean)) / float(scale)
                for value, mean, scale in zip(features, self.means, self.scales)
            ]
        logit = self.bias + sum(weight * value for weight, value in zip(self.weights, features))
        return sigmoid(logit)

    def decide(self, candidate: dict[str, Any]) -> SelectionDecision:
        score = self.score(candidate)
        selected = score >= self.threshold
        return SelectionDecision(
            selected=selected,
            score=score,
            reason="learned_score_above_threshold" if selected else "learned_score_below_threshold",
        )
