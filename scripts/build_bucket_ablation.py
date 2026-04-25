#!/usr/bin/env python
"""Build selected samples after excluding candidate feature buckets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lightr.analysis.bucket_ablation import should_exclude_candidate
from lightr.data.jsonl import read_jsonl, write_jsonl
from lightr.data.schemas import candidate_to_selected_sample
from lightr.selection.fixed_kl import FixedKLSelector


def _split_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Candidate JSONL path")
    parser.add_argument("--output", required=True, help="Selected output JSONL path")
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--exclude-token-categories", default=None)
    parser.add_argument("--exclude-step-types", default=None)
    parser.add_argument("--exclude-position-buckets", default=None)
    parser.add_argument("--exclude-confidence-buckets", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selector = FixedKLSelector(beta=args.beta)
    exclude_token_categories = _split_csv(args.exclude_token_categories)
    exclude_step_types = _split_csv(args.exclude_step_types)
    exclude_position_buckets = _split_csv(args.exclude_position_buckets)
    exclude_confidence_buckets = _split_csv(args.exclude_confidence_buckets)

    total = 0
    selected = 0
    excluded = 0

    def records():
        nonlocal total, selected, excluded
        for candidate in read_jsonl(args.input):
            total += 1
            if should_exclude_candidate(
                candidate,
                exclude_token_categories=exclude_token_categories,
                exclude_step_types=exclude_step_types,
                exclude_position_buckets=exclude_position_buckets,
                exclude_confidence_buckets=exclude_confidence_buckets,
            ):
                excluded += 1
                continue
            decision = selector.decide(candidate)
            if not decision.selected:
                continue
            selected += 1
            yield candidate_to_selected_sample(
                candidate,
                selector_name=f"fixed_kl_ablation_beta_{args.beta}",
                selector_score=decision.score,
            )

    write_jsonl(args.output, records())
    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "candidates_seen": total,
                "candidates_excluded": excluded,
                "samples_selected": selected,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

