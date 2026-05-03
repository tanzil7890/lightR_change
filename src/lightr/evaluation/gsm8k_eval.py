"""Lightweight GSM8K evaluation for base models and PEFT adapters."""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GSM8KEvalConfig:
    model_path: str
    input_jsonl: str
    output_jsonl: str
    summary_json: str | None = None
    adapter_path: str | None = None
    max_examples: int | None = 100
    start: int = 0
    max_new_tokens: int = 256
    batch_size: int = 1
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    seed: int = 0
    system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}."


def parse_torch_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def normalize_answer(value: str) -> str:
    cleaned = value.strip()
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace("%", "")
    cleaned = cleaned.rstrip(".")
    return cleaned.strip()


def extract_gsm8k_gold(answer: str) -> str:
    if "####" in answer:
        return normalize_answer(answer.split("####")[-1])
    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
    return normalize_answer(numbers[-1]) if numbers else ""


def _extract_boxed(text: str) -> str | None:
    match = re.search(r"\\boxed\{([^{}]+)\}", text)
    if match:
        return match.group(1).strip()
    return None


def extract_predicted_answer(text: str) -> str:
    boxed = _extract_boxed(text)
    if boxed:
        return normalize_answer(boxed)

    final_patterns = [
        r"(?:final answer|answer is|the answer is)\s*[:=]?\s*(-?\$?\d[\d,]*(?:\.\d+)?)",
        r"####\s*(-?\$?\d[\d,]*(?:\.\d+)?)",
    ]
    for pattern in final_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return normalize_answer(matches[-1])

    numbers = re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?", text)
    return normalize_answer(numbers[-1]) if numbers else ""


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_prompt(tokenizer: Any, question: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system_prompt}\nUser: {question}\nAssistant:"


def load_model_and_tokenizer(config: GSM8KEvalConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=parse_torch_dtype(config.torch_dtype),
        device_map=config.device_map,
    )
    if config.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, config.adapter_path)
    model.eval()
    return model, tokenizer


def run_gsm8k_eval(config: GSM8KEvalConfig) -> dict[str, Any]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    records = load_jsonl(config.input_jsonl)
    end = None if config.max_examples is None else config.start + config.max_examples
    eval_records = records[config.start:end]
    if not eval_records:
        raise ValueError("No GSM8K records selected for evaluation")

    output_path = Path(config.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(config.summary_json) if config.summary_json else output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(config)
    correct = 0
    results = []

    with output_path.open("w", encoding="utf-8") as handle:
        progress = tqdm(range(0, len(eval_records), config.batch_size), desc="Evaluating GSM8K")
        for batch_start in progress:
            batch_records = eval_records[batch_start : batch_start + config.batch_size]
            prompts = [
                build_prompt(tokenizer, record["question"], config.system_prompt)
                for record in batch_records
            ]
            model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            completion_start = model_inputs.input_ids.shape[1]

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            for item_offset, record in enumerate(batch_records):
                index = config.start + batch_start + item_offset
                completion_ids = generated_ids[item_offset][completion_start:]
                response = tokenizer.decode(completion_ids, skip_special_tokens=True)

                gold = extract_gsm8k_gold(record["answer"])
                prediction = extract_predicted_answer(response)
                is_correct = prediction == gold
                correct += int(is_correct)

                result = {
                    "index": index,
                    "id": record.get("id", record["question"]),
                    "question": record["question"],
                    "gold": gold,
                    "prediction": prediction,
                    "correct": is_correct,
                    "response": response,
                }
                results.append(result)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = {
        "config": asdict(config),
        "total": len(results),
        "correct": correct,
        "accuracy": correct / len(results),
        "output_jsonl": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
