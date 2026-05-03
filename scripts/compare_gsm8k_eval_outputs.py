#!/usr/bin/env python3
"""Compare GSM8K evaluation JSONL outputs from base and adapter runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_outputs(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = str(row.get("id", row.get("index")))
            rows[key] = row
    return rows


def accuracy(rows: dict[str, dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows.values() if row.get("correct")) / len(rows)


def example_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": row.get("index"),
        "id": row.get("id"),
        "gold": row.get("gold"),
        "prediction": row.get("prediction"),
        "response_preview": str(row.get("response", ""))[:300],
    }


def compare(base_path: str, fixed_path: str, rule_path: str, max_examples: int = 5) -> dict[str, Any]:
    base = read_outputs(base_path)
    fixed = read_outputs(fixed_path)
    rule = read_outputs(rule_path)
    common_keys = sorted(set(base) & set(fixed) & set(rule))
    if not common_keys:
        raise ValueError("No shared examples found across the three evaluation files")

    def correct_set(rows: dict[str, dict[str, Any]]) -> set[str]:
        return {key for key in common_keys if rows[key].get("correct")}

    base_correct = correct_set(base)
    fixed_correct = correct_set(fixed)
    rule_correct = correct_set(rule)

    fixed_regressions = sorted(base_correct - fixed_correct)
    fixed_improvements = sorted(fixed_correct - base_correct)
    rule_regressions = sorted(base_correct - rule_correct)
    rule_improvements = sorted(rule_correct - base_correct)
    rule_over_fixed = sorted(rule_correct - fixed_correct)
    fixed_over_rule = sorted(fixed_correct - rule_correct)

    report = {
        "total_common": len(common_keys),
        "accuracy": {
            "base": accuracy({key: base[key] for key in common_keys}),
            "fixed_kl_equal_budget": accuracy({key: fixed[key] for key in common_keys}),
            "rule_balanced": accuracy({key: rule[key] for key in common_keys}),
        },
        "correct_counts": {
            "base": len(base_correct),
            "fixed_kl_equal_budget": len(fixed_correct),
            "rule_balanced": len(rule_correct),
        },
        "overlap_counts": {
            "all_correct": len(base_correct & fixed_correct & rule_correct),
            "all_wrong": len(set(common_keys) - (base_correct | fixed_correct | rule_correct)),
            "fixed_improves_over_base": len(fixed_improvements),
            "fixed_regresses_from_base": len(fixed_regressions),
            "rule_improves_over_base": len(rule_improvements),
            "rule_regresses_from_base": len(rule_regressions),
            "rule_correct_fixed_wrong": len(rule_over_fixed),
            "fixed_correct_rule_wrong": len(fixed_over_rule),
        },
        "examples": {
            "fixed_improves_over_base": [example_summary(fixed[key]) for key in fixed_improvements[:max_examples]],
            "fixed_regresses_from_base": [example_summary(fixed[key]) for key in fixed_regressions[:max_examples]],
            "rule_improves_over_base": [example_summary(rule[key]) for key in rule_improvements[:max_examples]],
            "rule_regresses_from_base": [example_summary(rule[key]) for key in rule_regressions[:max_examples]],
            "rule_correct_fixed_wrong": [example_summary(rule[key]) for key in rule_over_fixed[:max_examples]],
            "fixed_correct_rule_wrong": [example_summary(fixed[key]) for key in fixed_over_rule[:max_examples]],
        },
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--rule", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-examples", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare(args.base, args.fixed, args.rule, max_examples=args.max_examples)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
