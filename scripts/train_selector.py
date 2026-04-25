#!/usr/bin/env python
"""Train a lightweight learned selector from candidate JSONL records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lightr.training.selector_training import train_linear_selector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Candidate JSONL path")
    parser.add_argument("--output", required=True, help="Selector artifact JSON path")
    parser.add_argument("--label-source", choices=["rule", "fixed_kl", "candidate_fixed_kl_flag"], default="rule")
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = train_linear_selector(
        candidate_path=args.input,
        output_path=args.output,
        label_source=args.label_source,
        beta=args.beta,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        threshold=args.threshold,
        max_records=args.max_records,
    )
    print(json.dumps(artifact["training"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

