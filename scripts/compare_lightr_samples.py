#!/usr/bin/env python
"""Compare original and regenerated LightReasoner selected sample JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lightr.data.jsonl import read_jsonl
from lightr.data.schemas import validate_selected_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, help="Original LightReasoner selected JSONL")
    parser.add_argument("--candidate-selected", required=True, help="New pipeline selected JSONL")
    parser.add_argument("--output", default=None, help="Optional JSON report output path")
    return parser.parse_args()


def _load(path: str) -> list[dict]:
    records = list(read_jsonl(path))
    for record in records:
        validate_selected_sample(record)
    return records


def _key(record: dict) -> tuple[str, int, str]:
    return (str(record["prompt_id"]), int(record["step"]), str(record["prefix"]))


def _summary(records: list[dict]) -> dict:
    by_prompt = Counter(str(record["prompt_id"]) for record in records)
    kl_values = [float(record["kl_divergence"]) for record in records]
    target_sizes = [len(record["token_ids"]) for record in records]
    return {
        "records": len(records),
        "prompts": len(by_prompt),
        "records_per_prompt_mean": mean(by_prompt.values()) if by_prompt else 0.0,
        "kl_mean": mean(kl_values) if kl_values else None,
        "kl_median": median(kl_values) if kl_values else None,
        "kl_min": min(kl_values) if kl_values else None,
        "kl_max": max(kl_values) if kl_values else None,
        "target_tokens_total": sum(target_sizes),
        "target_tokens_mean": mean(target_sizes) if target_sizes else 0.0,
    }


def main() -> None:
    args = parse_args()
    original = _load(args.original)
    regenerated = _load(args.candidate_selected)
    original_keys = {_key(record) for record in original}
    regenerated_keys = {_key(record) for record in regenerated}

    overlap = original_keys & regenerated_keys
    report = {
        "original": {
            "path": args.original,
            **_summary(original),
        },
        "candidate_selected": {
            "path": args.candidate_selected,
            **_summary(regenerated),
        },
        "overlap": {
            "matching_prompt_step_prefix": len(overlap),
            "original_only": len(original_keys - regenerated_keys),
            "candidate_selected_only": len(regenerated_keys - original_keys),
            "original_overlap_rate": len(overlap) / len(original_keys) if original_keys else None,
            "candidate_selected_overlap_rate": len(overlap) / len(regenerated_keys) if regenerated_keys else None,
        },
    }
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

