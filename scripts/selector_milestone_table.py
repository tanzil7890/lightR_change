#!/usr/bin/env python
"""Create the selector milestone table from an adaptive candidate JSONL.

The table compares:
- Fixed KL
- Rule strict
- Rule balanced

It can also write selected LightReasoner-compatible JSONL files for each
selector, so the same outputs can be reused for fine-tuning later.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lightr.data.jsonl import read_jsonl, write_jsonl
from lightr.data.schemas import candidate_to_selected_sample
from lightr.selection.fixed_kl import FixedKLSelector
from lightr.selection.rule_selector import RuleSelector


DEFAULT_STRICT_CONFIG = REPO_ROOT / "configs" / "adaptive_sampling" / "qwen15_gsm8k_rule_selector.json"
DEFAULT_BALANCED_CONFIG = REPO_ROOT / "configs" / "adaptive_sampling" / "qwen15_gsm8k_rule_selector_balanced.json"


def load_selector_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config.get("selector", config)


def target_size(candidate: dict[str, Any]) -> int:
    return len(candidate["contrastive_target"]["token_ids"])


def filtered_target_size(candidate: dict[str, Any], max_token_id_exclusive: int | None) -> int:
    if max_token_id_exclusive is None:
        return target_size(candidate)
    return sum(1 for token_id in candidate["contrastive_target"]["token_ids"] if 0 <= int(token_id) < max_token_id_exclusive)


def selected_rows(candidates: list[dict[str, Any]], selector) -> list[tuple[dict[str, Any], Any]]:
    rows = []
    for candidate in candidates:
        decision = selector.decide(candidate)
        if decision.selected:
            rows.append((candidate, decision))
    return rows


def summarize_rows(
    name: str,
    rows: list[tuple[dict[str, Any], Any]],
    total_candidates: int,
    max_token_id_exclusive: int | None,
) -> dict[str, Any]:
    trainable_rows = [
        (candidate, decision)
        for candidate, decision in rows
        if filtered_target_size(candidate, max_token_id_exclusive) > 0
    ]
    kl_values = [float(candidate["distribution_features"]["kl_expert_amateur"]) for candidate, _decision in trainable_rows]
    token_counts = Counter(candidate["token_features"]["token_category"] for candidate, _decision in trainable_rows)
    step_counts = Counter(candidate["step_features"]["step_type"] for candidate, _decision in trainable_rows)
    target_tokens = sum(filtered_target_size(candidate, max_token_id_exclusive) for candidate, _decision in trainable_rows)
    return {
        "selector": name,
        "samples": len(trainable_rows),
        "samples_before_vocab_filter": len(rows),
        "samples_dropped_by_vocab_filter": len(rows) - len(trainable_rows),
        "target_tokens": target_tokens,
        "selection_rate": len(trainable_rows) / total_candidates if total_candidates else 0.0,
        "kl_mean": mean(kl_values) if kl_values else 0.0,
        "kl_median": median(kl_values) if kl_values else 0.0,
        "main_token_types": ", ".join(f"{key}:{value}" for key, value in token_counts.most_common(5)),
        "main_step_types": ", ".join(f"{key}:{value}" for key, value in step_counts.most_common(5)),
    }


def markdown_table(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "| Selector | Samples | Target Tokens | Selection Rate | KL Mean | KL Median | Main Token Types | Main Step Types |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for summary in summaries:
        lines.append(
            "| {selector} | {samples} | {target_tokens} | {selection_rate:.2%} | "
            "{kl_mean:.4f} | {kl_median:.4f} | {main_token_types} | {main_step_types} |".format(**summary)
        )
    return "\n".join(lines)


def write_selected_outputs(
    output_dir: str | Path,
    selector_rows: dict[str, list[tuple[dict[str, Any], Any]]],
    *,
    max_token_id_exclusive: int | None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for selector_name, rows in selector_rows.items():
        safe_name = selector_name.lower().replace(" ", "_").replace("-", "_")
        samples = []
        for candidate, decision in rows:
            sample = candidate_to_selected_sample(
                candidate,
                selector_name=selector_name,
                selector_score=decision.score,
                max_token_id_exclusive=max_token_id_exclusive,
            )
            if sample is not None:
                samples.append(sample)
        write_jsonl(output_path / f"{safe_name}.jsonl", samples)


def load_max_token_id_exclusive(args: argparse.Namespace) -> int | None:
    if args.max_token_id_exclusive is not None:
        return args.max_token_id_exclusive
    if args.model_path is None:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    return tokenizer.vocab_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Adaptive candidate JSONL")
    parser.add_argument("--output-md", default=None, help="Optional markdown table output path")
    parser.add_argument("--output-json", default=None, help="Optional JSON summary output path")
    parser.add_argument("--selected-output-dir", default=None, help="Optional directory for selected JSONL outputs")
    parser.add_argument("--strict-config", default=str(DEFAULT_STRICT_CONFIG))
    parser.add_argument("--balanced-config", default=str(DEFAULT_BALANCED_CONFIG))
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--model-path", default=None, help="Optional tokenizer/model path used to filter trainer-vocab ids")
    parser.add_argument(
        "--max-token-id-exclusive",
        type=int,
        default=None,
        help="Drop selected target ids >= this value and renormalize weights",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = list(read_jsonl(args.input))
    max_token_id_exclusive = load_max_token_id_exclusive(args)
    selectors = {
        "Fixed KL": FixedKLSelector(beta=args.beta),
        "Rule strict": RuleSelector.from_config(load_selector_config(args.strict_config)),
        "Rule balanced": RuleSelector.from_config(load_selector_config(args.balanced_config)),
    }

    selector_rows = {
        name: selected_rows(candidates, selector)
        for name, selector in selectors.items()
    }
    summaries = [
        summarize_rows(
            name,
            rows,
            total_candidates=len(candidates),
            max_token_id_exclusive=max_token_id_exclusive,
        )
        for name, rows in selector_rows.items()
    ]
    table = markdown_table(summaries)

    result = {
        "input": args.input,
        "total_candidates": len(candidates),
        "max_token_id_exclusive": max_token_id_exclusive,
        "summaries": summaries,
    }

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(table + "\n", encoding="utf-8")
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.selected_output_dir:
        write_selected_outputs(
            args.selected_output_dir,
            selector_rows,
            max_token_id_exclusive=max_token_id_exclusive,
        )

    print(table)
    print()
    print(json.dumps({"total_candidates": len(candidates)}, indent=2))


if __name__ == "__main__":
    main()
