"""Bucket filtering helpers for adaptive LightReasoner ablations."""

from __future__ import annotations

from typing import Any


def position_bucket(candidate: dict[str, Any]) -> str:
    normalized_position = float(candidate.get("normalized_position", 0.0))
    if normalized_position < 0.33:
        return "early"
    if normalized_position < 0.66:
        return "middle"
    return "late"


def confidence_buckets(candidate: dict[str, Any]) -> set[str]:
    distribution = candidate["distribution_features"]
    buckets = set()
    if float(distribution.get("expert_entropy", 0.0)) <= 1.0:
        buckets.add("low_expert_entropy")
    else:
        buckets.add("high_expert_entropy")
    if float(distribution.get("entropy_gap", 0.0)) >= 1.0:
        buckets.add("high_entropy_gap")
    if not bool(distribution.get("top1_match", False)):
        buckets.add("top1_mismatch")
    return buckets


def should_exclude_candidate(
    candidate: dict[str, Any],
    *,
    exclude_token_categories: set[str] | None = None,
    exclude_step_types: set[str] | None = None,
    exclude_position_buckets: set[str] | None = None,
    exclude_confidence_buckets: set[str] | None = None,
) -> bool:
    token_category = str(candidate.get("token_features", {}).get("token_category", "unknown"))
    step_type = str(candidate.get("step_features", {}).get("step_type", "unknown"))
    if exclude_token_categories and token_category in exclude_token_categories:
        return True
    if exclude_step_types and step_type in exclude_step_types:
        return True
    if exclude_position_buckets and position_bucket(candidate) in exclude_position_buckets:
        return True
    if exclude_confidence_buckets and confidence_buckets(candidate) & exclude_confidence_buckets:
        return True
    return False

