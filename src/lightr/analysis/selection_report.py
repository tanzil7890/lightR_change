"""Selection statistics for candidate and selected-sample JSONL files."""

from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Any

from lightr.data.jsonl import read_jsonl


def _position_bucket(normalized_position: float | None) -> str:
    if normalized_position is None:
        return "unknown"
    if normalized_position < 0.33:
        return "early"
    if normalized_position < 0.66:
        return "middle"
    return "late"


def _safe_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def _selected_sample_metadata(record: dict[str, Any]) -> tuple[str, str, str]:
    metadata = record.get("selection_metadata", {})
    return (
        str(metadata.get("selector_name", "unknown")),
        str(metadata.get("token_category", "unknown")),
        str(metadata.get("step_type", "unknown")),
    )


def build_selection_report(path: str) -> dict[str, Any]:
    total = 0
    candidate_records = 0
    selected_sample_records = 0
    kl_values: list[float] = []
    expert_entropies: list[float] = []
    amateur_entropies: list[float] = []
    entropy_gaps: list[float] = []
    token_categories = Counter()
    step_types = Counter()
    position_buckets = Counter()
    selectors = Counter()
    fixed_kl_selected_flags = Counter()
    top1_match_flags = Counter()
    selected_by_rule = Counter()

    for record in read_jsonl(path):
        total += 1
        if "distribution_features" in record:
            candidate_records += 1
            distribution = record["distribution_features"]
            token_features = record.get("token_features", {})
            step_features = record.get("step_features", {})

            kl_values.append(float(distribution["kl_expert_amateur"]))
            expert_entropies.append(float(distribution.get("expert_entropy", 0.0)))
            amateur_entropies.append(float(distribution.get("amateur_entropy", 0.0)))
            entropy_gaps.append(float(distribution.get("entropy_gap", 0.0)))
            token_categories[str(token_features.get("token_category", "unknown"))] += 1
            step_types[str(step_features.get("step_type", "unknown"))] += 1
            position_buckets[_position_bucket(record.get("normalized_position"))] += 1
            fixed_flag = bool(record.get("fixed_kl_selected", False))
            fixed_kl_selected_flags[str(fixed_flag)] += 1
            top1_match_flags[str(bool(distribution.get("top1_match", False)))] += 1
            selected_by_rule["fixed_kl_selected" if fixed_flag else "fixed_kl_rejected"] += 1
        else:
            selected_sample_records += 1
            kl_values.append(float(record.get("kl_divergence", 0.0)))
            selector_name, token_category, step_type = _selected_sample_metadata(record)
            selectors[selector_name] += 1
            token_categories[token_category] += 1
            step_types[step_type] += 1
            position_buckets["unknown"] += 1

    fixed_selected = fixed_kl_selected_flags.get("True", 0)
    return {
        "input": path,
        "total_records": total,
        "candidate_records": candidate_records,
        "selected_sample_records": selected_sample_records,
        "selection_rate": fixed_selected / candidate_records if candidate_records else None,
        "kl": _safe_summary(kl_values),
        "expert_entropy": _safe_summary(expert_entropies),
        "amateur_entropy": _safe_summary(amateur_entropies),
        "entropy_gap": _safe_summary(entropy_gaps),
        "token_categories": dict(token_categories.most_common()),
        "step_types": dict(step_types.most_common()),
        "prefix_position_buckets": dict(position_buckets.most_common()),
        "selector_names": dict(selectors.most_common()),
        "fixed_kl_selected_flags": dict(fixed_kl_selected_flags.most_common()),
        "top1_match_flags": dict(top1_match_flags.most_common()),
        "selected_by_rule": dict(selected_by_rule.most_common()),
    }

