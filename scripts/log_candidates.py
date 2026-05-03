#!/usr/bin/env python
"""Log adaptive LightReasoner candidate records from expert/amateur models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lightr.sampling.candidate_logger import log_candidates, parse_torch_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-model", required=True)
    parser.add_argument("--amateur-model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset-name", default="gsm8k")
    parser.add_argument("--prompt-field", default="question")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--system-prompt", default="Please reason step by step.")
    parser.add_argument("--answer-field", default=None, help="Optional answer field used for correctness gating")
    parser.add_argument("--require-correct", action="store_true", help="Only log candidates from correct expert generations")
    parser.add_argument("--reject-degenerate", action="store_true", help="Reject obvious prompt-copying/repetition generations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = log_candidates(
        expert_model_path=args.expert_model,
        amateur_model_path=args.amateur_model,
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset_name,
        prompt_field=args.prompt_field,
        id_field=args.id_field,
        device=args.device,
        torch_dtype=parse_torch_dtype(args.torch_dtype),
        max_questions=args.max_questions,
        max_new_tokens=args.max_new_tokens,
        alpha=args.alpha,
        beta=args.beta,
        batch_size=args.batch_size,
        system_prompt=args.system_prompt,
        answer_field=args.answer_field,
        require_correct=args.require_correct,
        reject_degenerate=args.reject_degenerate,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
