"""Candidate logging for adaptive LightReasoner.

This module mirrors the original LightR sampling path but writes every
candidate prefix with reusable features, so selectors can be rerun without
recomputing expert and amateur model distributions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lightr.data.schemas import CANDIDATE_SCHEMA_VERSION
from lightr.evaluation.gsm8k_eval import extract_gsm8k_gold, extract_predicted_answer
from lightr.features.candidate_features import position_features
from lightr.features.distribution_features import compute_distribution_features
from lightr.features.step_type_features import step_type_features
from lightr.features.token_features import token_features


def parse_torch_dtype(dtype_name: str):
    dtype_name = dtype_name.lower()
    if dtype_name in {"auto", "torch.auto"}:
        return "auto"
    if dtype_name in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if dtype_name in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if dtype_name in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def load_prompts(path: str | Path, *, prompt_field: str, id_field: str, max_questions: int | None) -> list[dict[str, Any]]:
    prompts = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if prompt_field not in record:
                raise ValueError(f"Prompt field {prompt_field!r} missing from input record")
            prompt_id = record.get(id_field) or record[prompt_field]
            prompts.append(
                {
                    "prompt_id": prompt_id,
                    "question": record[prompt_field],
                    "raw": record,
                }
            )
            if max_questions is not None and len(prompts) >= max_questions:
                break
    return prompts


def load_processed_ids(checkpoint_path: str | Path | None) -> set[str]:
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        return set()
    processed = set()
    with Path(checkpoint_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            processed.add(str(record.get("prompt_id")))
    return processed


def _append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")


def build_contrastive_target(expert_dist, amateur_dist, tokenizer, *, alpha: float) -> dict[str, Any] | None:
    min_vocab_size = min(expert_dist.size(0), amateur_dist.size(0))
    expert_dist = expert_dist[:min_vocab_size]
    amateur_dist = amateur_dist[:min_vocab_size]

    mask = expert_dist >= (alpha * torch.max(expert_dist))
    if not mask.any():
        return None

    log_expert = torch.log(expert_dist + 1e-12)
    log_amateur = torch.log(amateur_dist + 1e-12)
    weights = torch.softmax((log_expert - log_amateur)[mask], dim=-1)
    token_ids = torch.arange(min_vocab_size, device=expert_dist.device)[mask].tolist()
    return {
        "token_ids": token_ids,
        "tokens": [tokenizer.decode([token_id]) for token_id in token_ids],
        "weights": weights.detach().float().cpu().tolist(),
    }


def _normalized_for_matching(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def detect_degenerate_generation(
    generation: str,
    *,
    question: str,
    system_prompt: str,
) -> tuple[bool, list[str]]:
    """Detect obvious prompt-copying and repetition artifacts.

    This is intentionally conservative. It is a gate for candidate logging, not
    a learned quality classifier.
    """
    reasons: list[str] = []
    normalized = _normalized_for_matching(generation)
    normalized_question = _normalized_for_matching(question)
    normalized_system = _normalized_for_matching(system_prompt)

    if not normalized:
        reasons.append("empty_generation")

    if normalized_system and normalized.count(normalized_system[:80]) >= 1:
        reasons.append("copies_system_prompt")

    if len(normalized_question) >= 80 and normalized_question[:80] in normalized:
        reasons.append("copies_question")

    artifact_patterns = {
        "ager": r"\bager\w*\b",
        "highlight": r"\bhighlight\b",
    }
    for name, pattern in artifact_patterns.items():
        if len(re.findall(pattern, normalized)) >= 3:
            reasons.append(f"repeated_artifact:{name}")

    if re.search(r"\b([a-zA-Z]{3,})\b(?:\s+\1\b){4,}", normalized):
        reasons.append("repeated_word")

    lines = [line.strip() for line in generation.splitlines() if len(line.strip()) >= 24]
    line_counts: dict[str, int] = {}
    for line in lines:
        key = _normalized_for_matching(line)
        line_counts[key] = line_counts.get(key, 0) + 1
    if any(count >= 3 for count in line_counts.values()):
        reasons.append("repeated_line")

    return bool(reasons), reasons


def log_candidates(
    *,
    expert_model_path: str,
    amateur_model_path: str,
    input_path: str | Path,
    output_path: str | Path,
    checkpoint_path: str | Path | None,
    dataset_name: str,
    prompt_field: str,
    id_field: str,
    device: str,
    torch_dtype,
    max_questions: int | None,
    max_new_tokens: int,
    alpha: float,
    beta: float,
    batch_size: int,
    system_prompt: str,
    answer_field: str | None = None,
    require_correct: bool = False,
    reject_degenerate: bool = False,
) -> dict[str, int]:
    prompts = load_prompts(input_path, prompt_field=prompt_field, id_field=id_field, max_questions=max_questions)
    processed_ids = load_processed_ids(checkpoint_path)

    expert_model = AutoModelForCausalLM.from_pretrained(
        expert_model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
    ).eval()
    amateur_model = AutoModelForCausalLM.from_pretrained(
        amateur_model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(expert_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_count = 0
    candidate_count = 0
    skipped_count = 0
    rejected_incorrect_count = 0
    rejected_degenerate_count = 0

    for prompt_obj in tqdm(prompts, desc="Logging candidates"):
        prompt_id = str(prompt_obj["prompt_id"])
        if prompt_id in processed_ids:
            skipped_count += 1
            continue

        question = prompt_obj["question"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = expert_model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )

        expert_targets = outputs.sequences[0][model_inputs.input_ids.shape[1] :].tolist()
        expert_generation = tokenizer.decode(expert_targets, skip_special_tokens=True)

        raw_record = prompt_obj.get("raw", {})
        gold_answer = None
        predicted_answer = None
        is_correct = None
        if answer_field:
            if answer_field not in raw_record:
                raise ValueError(f"Answer field {answer_field!r} missing from input record")
            gold_answer = extract_gsm8k_gold(str(raw_record[answer_field]))
            predicted_answer = extract_predicted_answer(expert_generation)
            is_correct = predicted_answer == gold_answer
            if require_correct and not is_correct:
                rejected_incorrect_count += 1
                if checkpoint_path is not None:
                    _append_jsonl(
                        checkpoint_path,
                        {
                            "prompt_id": prompt_id,
                            "num_steps": len(expert_targets),
                            "num_candidates": 0,
                            "accepted": False,
                            "rejection_reason": "incorrect_expert_generation",
                            "gold_answer": gold_answer,
                            "predicted_answer": predicted_answer,
                        },
                    )
                continue

        is_degenerate, degeneration_reasons = detect_degenerate_generation(
            expert_generation,
            question=question,
            system_prompt=system_prompt,
        )
        if reject_degenerate and is_degenerate:
            rejected_degenerate_count += 1
            if checkpoint_path is not None:
                _append_jsonl(
                    checkpoint_path,
                    {
                        "prompt_id": prompt_id,
                        "num_steps": len(expert_targets),
                        "num_candidates": 0,
                        "accepted": False,
                        "rejection_reason": "degenerate_expert_generation",
                        "degeneration_reasons": degeneration_reasons,
                        "gold_answer": gold_answer,
                        "predicted_answer": predicted_answer,
                    },
                )
            continue

        expert_probs = [F.softmax(score[0], dim=-1) for score in outputs.scores]
        sequence_length = len(expert_targets)

        input_ids_batch = []
        prefixes = []
        for step in range(1, sequence_length):
            prefix = tokenizer.decode(expert_targets[:step])
            formatted = text + prefix
            input_ids = tokenizer(formatted, return_tensors="pt").input_ids[0]
            input_ids_batch.append(input_ids)
            prefixes.append(prefix)

        amateur_probs_all = []
        with torch.no_grad():
            for start in range(0, len(input_ids_batch), batch_size):
                batch_chunk = input_ids_batch[start : start + batch_size]
                padded = torch.nn.utils.rnn.pad_sequence(
                    batch_chunk,
                    batch_first=True,
                    padding_value=tokenizer.pad_token_id,
                ).to(device, non_blocking=True)
                attention_mask = (padded != tokenizer.pad_token_id).long()
                logits = amateur_model(padded, attention_mask=attention_mask).logits
                for item_index, seq_len in enumerate([item.size(0) for item in batch_chunk]):
                    amateur_probs_all.append(F.softmax(logits[item_index, seq_len - 1, :], dim=-1))

        prompt_candidates = 0
        for offset, step in enumerate(range(1, sequence_length)):
            expert_dist = expert_probs[step]
            amateur_dist = amateur_probs_all[offset]
            target_token_id = expert_targets[step]
            target_token_text = tokenizer.decode([target_token_id])
            contrastive_target = build_contrastive_target(expert_dist, amateur_dist, tokenizer, alpha=alpha)
            if contrastive_target is None:
                continue

            distribution = compute_distribution_features(
                expert_dist,
                amateur_dist,
                target_token_id=target_token_id,
            )
            position = position_features(step, sequence_length)
            candidate = {
                "schema_version": CANDIDATE_SCHEMA_VERSION,
                "prompt_id": prompt_id,
                "question": question,
                "dataset": dataset_name,
                "model_pair": {
                    "expert": expert_model_path,
                    "amateur": amateur_model_path,
                },
                "step": step,
                "prefix": prefixes[offset],
                "target_token_id": target_token_id,
                "target_token_text": target_token_text,
                "sequence_length": sequence_length,
                "normalized_position": position["normalized_position"],
                "position_features": position,
                "distribution_features": distribution,
                "token_features": token_features(target_token_text),
                "step_features": step_type_features(prefixes[offset], target_token_text),
                "contrastive_target": contrastive_target,
                "fixed_kl_selected": bool(distribution["kl_expert_amateur"] >= beta),
                "expert_generation_metadata": {
                    "gold_answer": gold_answer,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "is_degenerate": is_degenerate,
                    "degeneration_reasons": degeneration_reasons,
                },
            }
            _append_jsonl(output_path, candidate)
            prompt_candidates += 1
            candidate_count += 1

        if checkpoint_path is not None:
            _append_jsonl(
                checkpoint_path,
                {
                    "prompt_id": prompt_id,
                    "num_steps": sequence_length,
                    "num_candidates": prompt_candidates,
                    "accepted": True,
                    "gold_answer": gold_answer,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "is_degenerate": is_degenerate,
                    "degeneration_reasons": degeneration_reasons,
                },
            )
        prompt_count += 1

    return {
        "prompts_processed": prompt_count,
        "prompts_skipped": skipped_count,
        "prompts_rejected_incorrect": rejected_incorrect_count,
        "prompts_rejected_degenerate": rejected_degenerate_count,
        "candidates_written": candidate_count,
    }
