"""Configurable contrastive soft-label LoRA fine-tuning."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


@dataclass
class ContrastiveFinetuneConfig:
    model_path: str
    train_jsonl: str
    output_dir: str
    max_steps: int = 20
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    max_length: int = 512
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    logging_steps: int = 1
    save_steps: int = 50
    save_total_limit: int = 2
    seed: int = 0
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 0
    report_to: str = "none"


def parse_torch_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ContrastiveSoftLabelDataset(Dataset):
    """Dataset compatible with LightReasoner selected-sample JSONL records."""

    def __init__(self, jsonl_path: str | Path, tokenizer: Any, model_vocab_size: int, max_length: int = 512):
        self.records: list[dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.vocab_size = model_vocab_size
        self.max_length = max_length

        with Path(jsonl_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        if not self.records:
            raise ValueError(f"No selected samples found in {jsonl_path}")

    def __len__(self) -> int:
        return len(self.records)

    def _format_prompt(self, question: str) -> str:
        messages = [
            {"role": "system", "content": "Please reason step by step."},
            {"role": "user", "content": question},
        ]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"System: Please reason step by step.\nUser: {question}\nAssistant:"

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.records[index]
        full_input = self._format_prompt(item["prompt_id"]) + item["prefix"]

        encoding = self.tokenizer(
            full_input,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = torch.zeros(self.vocab_size, dtype=torch.float32)
        for token_id, weight in zip(item["token_ids"], item["weights"]):
            if 0 <= int(token_id) < self.vocab_size:
                labels[int(token_id)] = float(weight)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }


def build_collate_fn(tokenizer: Any):
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [item["input_ids"] for item in batch],
                batch_first=True,
                padding_value=pad_token_id,
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                [item["attention_mask"] for item in batch],
                batch_first=True,
                padding_value=0,
            ),
            "labels": torch.stack([item["labels"] for item in batch]),
        }

    return collate_fn


def load_lora_model(config: ContrastiveFinetuneConfig):
    from peft import LoraConfig, TaskType, get_peft_model

    dtype = parse_torch_dtype(config.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=dtype,
        device_map=config.device_map,
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.lora_target_modules),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, lora_config)


class SoftLabelKLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        vocab_size = inputs["labels"].size(-1)
        logits = outputs.logits[:, -1, :vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)
        loss = F.kl_div(log_probs, inputs["labels"], reduction="batchmean")
        return (loss, outputs) if return_outputs else loss


def save_config(config: ContrastiveFinetuneConfig) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload["lora_target_modules"] = list(config.lora_target_modules)
    (output_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_contrastive_finetune(config: ContrastiveFinetuneConfig) -> dict[str, Any]:
    set_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    save_config(config)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = ContrastiveSoftLabelDataset(
        config.train_jsonl,
        tokenizer,
        model_vocab_size=tokenizer.vocab_size,
        max_length=config.max_length,
    )
    model = load_lora_model(config)

    dtype = parse_torch_dtype(config.torch_dtype)
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=dtype is torch.bfloat16,
        fp16=dtype is torch.float16,
        report_to=config.report_to,
        disable_tqdm=False,
        remove_unused_columns=False,
        dataloader_num_workers=config.dataloader_num_workers,
        seed=config.seed,
    )

    trainer = SoftLabelKLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=build_collate_fn(tokenizer),
    )

    train_output = trainer.train()
    final_adapter_dir = Path(config.output_dir) / "final_adapter"
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(final_adapter_dir)

    metrics = dict(train_output.metrics)
    metrics["train_samples"] = len(train_dataset)
    metrics["final_adapter_dir"] = str(final_adapter_dir)
    (Path(config.output_dir) / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics
