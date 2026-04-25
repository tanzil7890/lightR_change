"""Candidate-level feature helpers."""

from __future__ import annotations


def position_features(step: int, sequence_length: int) -> dict[str, bool | float | int]:
    if sequence_length <= 0:
        normalized_position = 0.0
    else:
        normalized_position = step / sequence_length
    return {
        "step": step,
        "sequence_length": sequence_length,
        "normalized_position": normalized_position,
        "is_early": normalized_position < 0.33,
        "is_middle": 0.33 <= normalized_position < 0.66,
        "is_late": normalized_position >= 0.66,
    }

