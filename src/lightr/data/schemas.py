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


def filter_target_for_training_vocab(
    target: dict[str, Any],
    *,
    max_token_id_exclusive: int | None,
) -> tuple[dict[str, Any] | None, int]:
    """Drop target tokens that the current trainer cannot represent.

    `LightR_finetuning.py` uses `tokenizer.vocab_size` as the label dimension.
    Some Qwen special tokens, such as eos/pad id 151643, are valid tokenizer ids
    but are outside `tokenizer.vocab_size`. Those ids must be dropped before
    selected samples are sent to the existing trainer.
    """
    if max_token_id_exclusive is None:
        return target, 0

    kept_tokens = []
    kept_token_ids = []
    kept_weights = []
    dropped = 0
    for token, token_id, weight in zip(target["tokens"], target["token_ids"], target["weights"]):
        if 0 <= int(token_id) < max_token_id_exclusive:
            kept_tokens.append(token)
            kept_token_ids.append(token_id)
            kept_weights.append(float(weight))
        else:
            dropped += 1

    if not kept_token_ids:
        return None, dropped

    weight_sum = sum(kept_weights)
    if weight_sum <= 0:
        return None, dropped

    return {
        "tokens": kept_tokens,
        "token_ids": kept_token_ids,
        "weights": [weight / weight_sum for weight in kept_weights],
    }, dropped


def candidate_to_selected_sample(
    candidate: dict[str, Any],
    *,
    selector_name: str,
    selector_score: float | None = None,
    prompt_text_field: str = "question",
    max_token_id_exclusive: int | None = None,
) -> dict[str, Any] | None:
    """Convert a candidate record to the current LightReasoner training schema.

    `LightR_finetuning.py` treats `prompt_id` as the actual user question. For
    compatibility, this function prefers candidate["question"] when present and
    falls back to candidate["prompt_id"].
    """
    validate_candidate_record(candidate)
    target, dropped_token_count = filter_target_for_training_vocab(
        candidate["contrastive_target"],
        max_token_id_exclusive=max_token_id_exclusive,
    )
    if target is None:
        return None
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
            "dropped_out_of_vocab_target_tokens": dropped_token_count,
            "max_token_id_exclusive": max_token_id_exclusive,
        },
    }
    validate_selected_sample(selected)
    return selected
