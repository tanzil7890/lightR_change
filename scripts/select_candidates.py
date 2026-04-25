#!/usr/bin/env python
"""Select adaptive candidates and emit LightReasoner-compatible samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lightr.data.jsonl import read_jsonl, write_jsonl
from lightr.data.schemas import candidate_to_selected_sample
from lightr.selection.fixed_kl import FixedKLSelector
from lightr.selection.learned_selector import LinearSelector
from lightr.selection.rule_selector import RuleSelector


def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_selector(args: argparse.Namespace):
    config = load_config(args.config)
    selector_config = config.get("selector", config)
    if args.selector == "fixed_kl":
        return FixedKLSelector(beta=float(selector_config.get("beta", args.beta)))
    if args.selector == "rule":
        return RuleSelector.from_config(selector_config)
    if args.selector == "learned":
        if args.config is None:
            raise ValueError("--config must point to a learned selector artifact for --selector learned")
        return LinearSelector.from_path(args.config)
    raise ValueError(f"Unsupported selector: {args.selector}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Candidate JSONL path")
    parser.add_argument("--output", required=True, help="Selected LightReasoner sample JSONL path")
    parser.add_argument("--selector", choices=["fixed_kl", "rule", "learned"], default="fixed_kl")
    parser.add_argument("--config", default=None, help="Optional JSON selector config")
    parser.add_argument("--beta", type=float, default=0.4, help="Fallback beta for fixed_kl")
    parser.add_argument("--prompt-text-field", default="question")
    parser.add_argument("--max-samples", type=int, default=None, help="Keep at most this many selected samples")
    parser.add_argument(
        "--max-target-tokens",
        type=int,
        default=None,
        help="Keep selected samples until the total contrastive target token count reaches this budget",
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=None,
        help="Keep only the top selected samples by selector score, as a percent in (0, 100]",
    )
    return parser.parse_args()


def _target_size(candidate: dict[str, Any]) -> int:
    return len(candidate["contrastive_target"]["token_ids"])


def _apply_budgets(rows: list[tuple[dict[str, Any], Any]], args: argparse.Namespace) -> list[tuple[dict[str, Any], Any]]:
    if args.top_percent is not None:
        if args.top_percent <= 0 or args.top_percent > 100:
            raise ValueError("--top-percent must be in (0, 100]")
        keep = max(1, int(len(rows) * (args.top_percent / 100.0)))
        rows = sorted(rows, key=lambda row: row[1].score if row[1].score is not None else 0.0, reverse=True)[:keep]

    if args.max_samples is not None:
        rows = sorted(rows, key=lambda row: row[1].score if row[1].score is not None else 0.0, reverse=True)[: args.max_samples]

    if args.max_target_tokens is not None:
        budgeted = []
        used = 0
        for candidate, decision in sorted(
            rows,
            key=lambda row: row[1].score if row[1].score is not None else 0.0,
            reverse=True,
        ):
            size = _target_size(candidate)
            if used + size > args.max_target_tokens:
                continue
            budgeted.append((candidate, decision))
            used += size
        rows = budgeted
    return rows


def main() -> None:
    args = parse_args()
    selector = build_selector(args)
    total = 0
    reason_counts: dict[str, int] = {}
    selected_rows = []

    for candidate in read_jsonl(args.input):
        total += 1
        decision = selector.decide(candidate)
        reason_counts[decision.reason] = reason_counts.get(decision.reason, 0) + 1
        if decision.selected:
            selected_rows.append((candidate, decision))

    pre_budget_selected = len(selected_rows)
    selected_rows = _apply_budgets(selected_rows, args)

    selected_count = len(selected_rows)
    selected_target_tokens = sum(_target_size(candidate) for candidate, _decision in selected_rows)
    write_jsonl(
        args.output,
        (
            candidate_to_selected_sample(
                candidate,
                selector_name=selector.name,
                selector_score=decision.score,
                prompt_text_field=args.prompt_text_field,
            )
            for candidate, decision in selected_rows
        ),
    )
    summary = {
        "selector": selector.name,
        "input": args.input,
        "output": args.output,
        "candidates_seen": total,
        "samples_selected_before_budget": pre_budget_selected,
        "samples_selected": selected_count,
        "selected_target_tokens": selected_target_tokens,
        "selection_rate": selected_count / total if total else 0.0,
        "reason_counts": reason_counts,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
