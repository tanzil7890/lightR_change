"""Schema validation and conversion helpers for adaptive LightReasoner data."""

from __future__ import annotations

from typing import Any


CANDIDATE_SCHEMA_VERSION = "candidate_v1"

REQUIRED_SELECTED_FIELDS = {
    "prompt_id",
    "step",
    "prefix",
    "tokens",
    "token_ids",
    "weights",
    "kl_divergence",
}

REQUIRED_CANDIDATE_FIELDS = {
    "schema_version",
    "prompt_id",
    "step",
    "prefix",
    "distribution_features",
    "token_features",
    "step_features",
    "contrastive_target",
}


def require_fields(record: dict[str, Any], required_fields: set[str], record_name: str) -> None:
    missing = sorted(field for field in required_fields if field not in record)
    if missing:
        raise ValueError(f"{record_name} missing required fields: {', '.join(missing)}")


def validate_selected_sample(record: dict[str, Any]) -> None:
    require_fields(record, REQUIRED_SELECTED_FIELDS, "selected sample")
    if len(record["token_ids"]) != len(record["weights"]):
        raise ValueError("selected sample token_ids and weights lengths differ")
    if len(record["tokens"]) != len(record["token_ids"]):
        raise ValueError("selected sample tokens and token_ids lengths differ")


def validate_candidate_record(record: dict[str, Any]) -> None:
    require_fields(record, REQUIRED_CANDIDATE_FIELDS, "candidate record")
    if record["schema_version"] != CANDIDATE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported candidate schema {record['schema_version']!r}; "
            f"expected {CANDIDATE_SCHEMA_VERSION!r}"
        )
    target = record["contrastive_target"]
    require_fields(target, {"tokens", "token_ids", "weights"}, "candidate contrastive_target")
    if len(target["token_ids"]) != len(target["weights"]):
        raise ValueError("candidate token_ids and weights lengths differ")
    if len(target["tokens"]) != len(target["token_ids"]):
        raise ValueError("candidate tokens and token_ids lengths differ")


def candidate_to_selected_sample(
    candidate: dict[str, Any],
    *,
    selector_name: str,
    selector_score: float | None = None,
    prompt_text_field: str = "question",
) -> dict[str, Any]:
    """Convert a candidate record to the current LightReasoner training schema.

    `LightR_finetuning.py` treats `prompt_id` as the actual user question. For
    compatibility, this function prefers candidate["question"] when present and
    falls back to candidate["prompt_id"].
    """
    validate_candidate_record(candidate)
    target = candidate["contrastive_target"]
    distribution_features = candidate["distribution_features"]
    token_features = candidate.get("token_features", {})
    step_features = candidate.get("step_features", {})

    question_text = candidate.get(prompt_text_field) or candidate["prompt_id"]
    selected = {
        "prompt_id": question_text,
        "step": candidate["step"],
        "prefix": candidate["prefix"],
        "tokens": target["tokens"],
        "token_ids": target["token_ids"],
        "weights": target["weights"],
        "kl_divergence": distribution_features["kl_expert_amateur"],
        "selection_metadata": {
            "selector_name": selector_name,
            "selector_score": selector_score,
            "original_prompt_id": candidate["prompt_id"],
            "token_category": token_features.get("token_category"),
            "step_type": step_features.get("step_type"),
        },
    }
    validate_selected_sample(selected)
    return selected

