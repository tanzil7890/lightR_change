"""Distribution-level features for expert-amateur comparisons."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected for full sampling, not pure utility tests.
    torch = None


def _to_float_list(values: Any) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().float().cpu().tolist()
    return [float(value) for value in values]


def _normalize(values: Sequence[float], epsilon: float) -> list[float]:
    clipped = [max(float(value), epsilon) for value in values]
    total = sum(clipped)
    if total <= 0:
        raise ValueError("distribution total must be positive")
    return [value / total for value in clipped]


def entropy(probs: Sequence[float], *, epsilon: float = 1e-12) -> float:
    normalized = _normalize(probs, epsilon)
    return -sum(prob * math.log(prob) for prob in normalized)


def kl_divergence(p_expert: Sequence[float], p_amateur: Sequence[float], *, epsilon: float = 1e-12) -> float:
    size = min(len(p_expert), len(p_amateur))
    expert = _normalize(p_expert[:size], epsilon)
    amateur = _normalize(p_amateur[:size], epsilon)
    return sum(p_e * (math.log(p_e) - math.log(p_a)) for p_e, p_a in zip(expert, amateur))


def top_values(probs: Sequence[float]) -> tuple[int, float, int | None, float]:
    indexed = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)
    if not indexed:
        raise ValueError("cannot compute top values for empty distribution")
    top1_id, top1_prob = indexed[0]
    if len(indexed) == 1:
        return top1_id, float(top1_prob), None, 0.0
    top2_id, top2_prob = indexed[1]
    return top1_id, float(top1_prob), top2_id, float(top2_prob)


def compute_distribution_features(
    expert_dist: Any,
    amateur_dist: Any,
    *,
    target_token_id: int | None = None,
    epsilon: float = 1e-12,
) -> dict[str, float | bool | int | None]:
    if torch is not None and hasattr(expert_dist, "detach") and hasattr(amateur_dist, "detach"):
        return _compute_torch_distribution_features(
            expert_dist,
            amateur_dist,
            target_token_id=target_token_id,
            epsilon=epsilon,
        )

    expert = _to_float_list(expert_dist)
    amateur = _to_float_list(amateur_dist)
    size = min(len(expert), len(amateur))
    expert = _normalize(expert[:size], epsilon)
    amateur = _normalize(amateur[:size], epsilon)

    expert_top1_id, expert_top1_prob, _expert_top2_id, expert_top2_prob = top_values(expert)
    amateur_top1_id, amateur_top1_prob, _amateur_top2_id, _amateur_top2_prob = top_values(amateur)

    target_expert_prob = None
    target_amateur_prob = None
    target_prob_gap = None
    if target_token_id is not None and 0 <= target_token_id < size:
        target_expert_prob = expert[target_token_id]
        target_amateur_prob = amateur[target_token_id]
        target_prob_gap = target_expert_prob - target_amateur_prob

    expert_entropy = entropy(expert, epsilon=epsilon)
    amateur_entropy = entropy(amateur, epsilon=epsilon)
    return {
        "kl_expert_amateur": kl_divergence(expert, amateur, epsilon=epsilon),
        "expert_entropy": expert_entropy,
        "amateur_entropy": amateur_entropy,
        "entropy_gap": amateur_entropy - expert_entropy,
        "expert_top1_id": expert_top1_id,
        "amateur_top1_id": amateur_top1_id,
        "expert_top1_prob": expert_top1_prob,
        "amateur_top1_prob": amateur_top1_prob,
        "expert_top2_margin": expert_top1_prob - expert_top2_prob,
        "top1_match": expert_top1_id == amateur_top1_id,
        "expert_amateur_top1_prob_gap": expert_top1_prob - amateur_top1_prob,
        "target_expert_prob": target_expert_prob,
        "target_amateur_prob": target_amateur_prob,
        "target_prob_gap": target_prob_gap,
    }


def _compute_torch_distribution_features(
    expert_dist,
    amateur_dist,
    *,
    target_token_id: int | None,
    epsilon: float,
) -> dict[str, float | bool | int | None]:
    size = min(expert_dist.size(0), amateur_dist.size(0))
    expert = expert_dist[:size].float().clamp_min(epsilon)
    amateur = amateur_dist[:size].float().clamp_min(epsilon)
    expert = expert / expert.sum()
    amateur = amateur / amateur.sum()

    expert_entropy = -torch.sum(expert * torch.log(expert)).item()
    amateur_entropy = -torch.sum(amateur * torch.log(amateur)).item()
    kl_value = torch.sum(expert * (torch.log(expert) - torch.log(amateur))).item()

    expert_top_values, expert_top_indices = torch.topk(expert, k=min(2, size))
    amateur_top_values, amateur_top_indices = torch.topk(amateur, k=min(2, size))
    expert_top1_id = int(expert_top_indices[0].item())
    amateur_top1_id = int(amateur_top_indices[0].item())
    expert_top1_prob = float(expert_top_values[0].item())
    amateur_top1_prob = float(amateur_top_values[0].item())
    expert_top2_prob = float(expert_top_values[1].item()) if expert_top_values.numel() > 1 else 0.0

    target_expert_prob = None
    target_amateur_prob = None
    target_prob_gap = None
    if target_token_id is not None and 0 <= target_token_id < size:
        target_expert_prob = float(expert[target_token_id].item())
        target_amateur_prob = float(amateur[target_token_id].item())
        target_prob_gap = target_expert_prob - target_amateur_prob

    return {
        "kl_expert_amateur": kl_value,
        "expert_entropy": expert_entropy,
        "amateur_entropy": amateur_entropy,
        "entropy_gap": amateur_entropy - expert_entropy,
        "expert_top1_id": expert_top1_id,
        "amateur_top1_id": amateur_top1_id,
        "expert_top1_prob": expert_top1_prob,
        "amateur_top1_prob": amateur_top1_prob,
        "expert_top2_margin": expert_top1_prob - expert_top2_prob,
        "top1_match": expert_top1_id == amateur_top1_id,
        "expert_amateur_top1_prob_gap": expert_top1_prob - amateur_top1_prob,
        "target_expert_prob": target_expert_prob,
        "target_amateur_prob": target_amateur_prob,
        "target_prob_gap": target_prob_gap,
    }
