"""Deterministic token category features."""

from __future__ import annotations

import re


OPERATOR_TOKENS = {"+", "-", "*", "/", "^", "<", ">"}
EQUALS_TOKENS = {"=", "=="}
UNIT_PATTERN = re.compile(
    r"(\$|%|cent|cents|dollar|dollars|kg|g|cm|mm|meter|meters|mile|miles|hour|hours|minute|minutes)",
    re.IGNORECASE,
)
FINAL_ANSWER_PATTERN = re.compile(r"(####|boxed|final answer|answer is|therefore)", re.IGNORECASE)
VARIABLE_PATTERN = re.compile(r"^[A-Za-z]$")


def classify_token(token_text: str) -> str:
    stripped = token_text.strip()
    if token_text == "":
        return "other"
    if "\n" in token_text:
        return "newline"
    if stripped == "":
        return "space"
    if FINAL_ANSWER_PATTERN.search(token_text):
        return "final_answer_marker"
    if UNIT_PATTERN.search(token_text):
        return "currency_or_unit"
    if any(char.isdigit() for char in stripped):
        return "number"
    if stripped in EQUALS_TOKENS:
        return "equals"
    if stripped in OPERATOR_TOKENS:
        return "operator"
    if VARIABLE_PATTERN.match(stripped):
        return "variable"
    if all(not char.isalnum() for char in stripped):
        return "punctuation"
    if any(char.isalpha() for char in stripped):
        return "word"
    return "other"


def token_features(token_text: str) -> dict[str, bool | str]:
    category = classify_token(token_text)
    return {
        "token_category": category,
        "is_number": category == "number",
        "is_operator": category == "operator",
        "is_equals": category == "equals",
        "is_variable": category == "variable",
        "is_punctuation": category == "punctuation",
        "is_newline": category == "newline",
        "is_space": category == "space",
        "is_currency_or_unit": category == "currency_or_unit",
        "is_final_answer_marker": category == "final_answer_marker",
    }
