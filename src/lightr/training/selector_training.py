"""Train lightweight selector artifacts without extra dependencies."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

from lightr.data.jsonl import read_jsonl
from lightr.selection.fixed_kl import FixedKLSelector
from lightr.selection.learned_selector import default_feature_names, sigmoid, vectorize
from lightr.selection.rule_selector import RuleSelector


def _label_candidate(candidate: dict, label_source: str, beta: float) -> int:
    if label_source == "rule":
        return int(RuleSelector().decide(candidate).selected)
    if label_source == "fixed_kl":
        return int(FixedKLSelector(beta=beta).decide(candidate).selected)
    if label_source == "candidate_fixed_kl_flag":
        return int(bool(candidate.get("fixed_kl_selected", False)))
    raise ValueError(f"Unsupported label source: {label_source}")


def _standardize_matrix(matrix: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    if not matrix:
        return matrix, [], []
    width = len(matrix[0])
    means = []
    scales = []
    for column in range(width):
        values = [row[column] for row in matrix]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values))
        scale = math.sqrt(variance) or 1.0
        means.append(mean)
        scales.append(scale)
    standardized = [
        [(row[column] - means[column]) / scales[column] for column in range(width)]
        for row in matrix
    ]
    return standardized, means, scales


def _apply_standardization(features: list[float], means: list[float], scales: list[float]) -> list[float]:
    return [(value - mean) / scale for value, mean, scale in zip(features, means, scales)]


def train_linear_selector(
    *,
    candidate_path: str,
    output_path: str,
    label_source: str,
    beta: float = 0.4,
    epochs: int = 20,
    learning_rate: float = 0.05,
    l2: float = 1e-4,
    threshold: float = 0.5,
    max_records: int | None = None,
) -> dict:
    feature_names = default_feature_names()
    records = []
    for index, candidate in enumerate(read_jsonl(candidate_path)):
        if max_records is not None and index >= max_records:
            break
        records.append(candidate)
    if not records:
        raise ValueError("No candidate records found for selector training")

    matrix = [vectorize(candidate, feature_names) for candidate in records]
    matrix, means, scales = _standardize_matrix(matrix)
    labels = [_label_candidate(candidate, label_source, beta) for candidate in records]

    weights = [0.0 for _ in feature_names]
    bias = 0.0
    for _epoch in range(epochs):
        for features, label in zip(matrix, labels):
            logit = bias + sum(weight * value for weight, value in zip(weights, features))
            pred = sigmoid(logit)
            error = pred - label
            bias -= learning_rate * error
            for index, value in enumerate(features):
                gradient = error * value + l2 * weights[index]
                weights[index] -= learning_rate * gradient

    correct = 0
    for features, label in zip(matrix, labels):
        pred = sigmoid(bias + sum(weight * value for weight, value in zip(weights, features)))
        correct += int((pred >= threshold) == bool(label))

    artifact = {
        "name": f"learned_linear_{label_source}",
        "model_type": "linear_logistic_sgd",
        "feature_names": feature_names,
        "weights": weights,
        "bias": bias,
        "threshold": threshold,
        "standardization": {
            "means": means,
            "scales": scales,
        },
        "training": {
            "candidate_path": candidate_path,
            "label_source": label_source,
            "beta": beta,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "records": len(records),
            "positive_labels": sum(labels),
            "training_accuracy": correct / len(records),
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def standardized_vectorize(candidate: dict, artifact: dict) -> list[float]:
    raw = vectorize(candidate, list(artifact["feature_names"]))
    standardization = artifact.get("standardization", {})
    means = standardization.get("means")
    scales = standardization.get("scales")
    if not means or not scales:
        return raw
    return _apply_standardization(raw, means, scales)

