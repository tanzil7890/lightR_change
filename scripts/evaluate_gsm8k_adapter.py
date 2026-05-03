#!/usr/bin/env python3
"""Evaluate a base model or LoRA adapter on a GSM8K JSONL subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lightr.evaluation.gsm8k_eval import GSM8KEvalConfig, run_gsm8k_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Base model path")
    parser.add_argument("--adapter-path", default=None, help="Optional PEFT adapter path")
    parser.add_argument("--input-jsonl", required=True, help="GSM8K test JSONL path")
    parser.add_argument("--output-jsonl", required=True, help="Per-example output JSONL")
    parser.add_argument("--summary-json", default=None, help="Optional summary JSON path")
    parser.add_argument("--max-examples", type=int, default=100, help="Number of examples to evaluate; use -1 for all")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--system-prompt",
        default="Please reason step by step, and put your final answer within \\boxed{}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_examples = None if args.max_examples == -1 else args.max_examples
    summary = run_gsm8k_eval(
        GSM8KEvalConfig(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            input_jsonl=args.input_jsonl,
            output_jsonl=args.output_jsonl,
            summary_json=args.summary_json,
            max_examples=max_examples,
            start=args.start,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            seed=args.seed,
            system_prompt=args.system_prompt,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
