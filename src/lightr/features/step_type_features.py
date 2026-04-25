"""Heuristic reasoning step type features."""

from __future__ import annotations

import re


FINAL_ANSWER_RE = re.compile(r"(####|boxed|final answer|answer is|the answer is)", re.IGNORECASE)
CALCULATION_RE = re.compile(r"\d[\d,.\s]*(\+|-|\*|/|=|%|\^| times | divided by | equals )", re.IGNORECASE)
SETUP_RE = re.compile(
    r"\b(let|suppose|we need|given|define|denote|total number|cost|rate|assume)\b",
    re.IGNORECASE,
)
TRANSFORMATION_RE = re.compile(
    r"\b(so|therefore|then|thus|which means|simplify|substitute|rearrange|combine)\b",
    re.IGNORECASE,
)
VERIFICATION_RE = re.compile(r"\b(check|verify|since|because|confirm)\b", re.IGNORECASE)


def classify_step_type(prefix: str, token_text: str = "", *, window_chars: int = 240) -> tuple[str, float]:
    """Return a coarse reasoning step type and a heuristic confidence."""
    local_text = (prefix[-window_chars:] + token_text).lower()
    if FINAL_ANSWER_RE.search(local_text):
        return "final_answer", 0.85
    if CALCULATION_RE.search(local_text):
        return "calculation", 0.75
    if SETUP_RE.search(local_text):
        return "setup", 0.70
    if TRANSFORMATION_RE.search(local_text):
        return "transformation", 0.65
    if VERIFICATION_RE.search(local_text):
        return "verification", 0.60
    return "other", 0.30


def step_type_features(prefix: str, token_text: str = "") -> dict[str, float | str]:
    step_type, confidence = classify_step_type(prefix, token_text)
    return {
        "step_type": step_type,
        "step_type_confidence": confidence,
    }

