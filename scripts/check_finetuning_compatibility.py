#!/usr/bin/env python
"""Validate selected samples against LightR_finetuning.py data expectations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lightr.data.jsonl import read_jsonl
from lightr.data.schemas import validate_selected_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Selected LightReasoner sample JSONL")
    parser.add_argument("--model-path", default=None, help="Optional tokenizer/model path for vocab-size validation")
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def _load_vocab_size(model_path: str | None) -> int | None:
    if model_path is None:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer.vocab_size


def main() -> None:
    args = parse_args()
    vocab_size = _load_vocab_size(args.model_path)
    checked = 0
    max_token_id = -1
    metadata_records = 0
    errors = []

    for line_index, record in enumerate(read_jsonl(args.input), start=1):
        if args.max_records is not None and checked >= args.max_records:
            break
        try:
            validate_selected_sample(record)
            if not isinstance(record["prompt_id"], str) or not record["prompt_id"].strip():
                raise ValueError("prompt_id must contain the question text for current LightR_finetuning.py")
            if not isinstance(record["prefix"], str):
                raise ValueError("prefix must be a string")
            if abs(sum(float(value) for value in record["weights"]) - 1.0) > 1e-3:
                raise ValueError("weights should sum to approximately 1.0")
            record_max = max(int(token_id) for token_id in record["token_ids"])
            max_token_id = max(max_token_id, record_max)
            if vocab_size is not None and record_max >= vocab_size:
                raise ValueError(f"token id {record_max} exceeds tokenizer vocab size {vocab_size}")
            if "selection_metadata" in record:
                metadata_records += 1
        except Exception as exc:  # noqa: BLE001 - report all compatibility issues together.
            errors.append({"line": line_index, "error": str(exc)})
        checked += 1

    summary = {
        "input": args.input,
        "records_checked": checked,
        "metadata_records": metadata_records,
        "max_token_id": max_token_id,
        "vocab_size": vocab_size,
        "compatible": not errors,
        "errors": errors[:20],
        "error_count": len(errors),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

