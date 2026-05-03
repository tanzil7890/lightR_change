#!/usr/bin/env python3
"""Run LightReasoner contrastive soft-label LoRA fine-tuning from CLI args."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lightr.training.contrastive_finetuning import ContrastiveFinetuneConfig, run_contrastive_finetune


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Base expert model path")
    parser.add_argument("--train-jsonl", required=True, help="Selected sample JSONL path")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoints and final adapter")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj", help="Comma-separated LoRA target modules")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--report-to", default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ContrastiveFinetuneConfig(
        model_path=args.model_path,
        train_jsonl=args.train_jsonl,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=tuple(module.strip() for module in args.lora_target_modules.split(",") if module.strip()),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to,
    )
    metrics = run_contrastive_finetune(config)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
