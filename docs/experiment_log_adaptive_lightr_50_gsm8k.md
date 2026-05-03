# Experiment Log: Adaptive Step-Aware LightReasoner, GSM8K 50-Question Milestone

Date: 2026-04-25

## Purpose

This log records the first real CUDA run for the Adaptive Step-Aware
LightReasoner extension.

The immediate research milestone was:

> Before fine-tuning, compare Fixed KL, Rule strict, and Rule balanced selectors
> on a 50-question GSM8K candidate log and verify that the selected outputs are
> compatible with the existing `LightR_finetuning.py` training schema.

The goal is not yet to claim accuracy improvement. The goal is to show that the
new candidate pipeline works and that adaptive selectors produce meaningfully
different supervision subsets from the fixed-KL baseline.

## Machine and Environment

Cloud provider: RunPod

GPU:

```text
NVIDIA GeForce RTX 4090
VRAM: 24564 MiB
Driver Version: 570.195.03
CUDA Version reported by nvidia-smi: 12.8
```

Python environment:

```text
Python virtual environment: .venv
PyTorch: 2.11.0+cu128
PyTorch CUDA build: 12.8
CUDA available: True
GPU detected by PyTorch: NVIDIA GeForce RTX 4090
```

Verification command:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

Observed output:

```text
torch: 2.11.0+cu128
torch cuda build: 12.8
cuda available: True
gpu: NVIDIA GeForce RTX 4090
```

## Models

Expert:

```text
Qwen/Qwen2.5-Math-1.5B
Local path: ./Qwen2.5-Math-1.5B
```

Amateur:

```text
Qwen/Qwen2.5-0.5B
Local path: ./Qwen2.5-0.5B
```

Downloaded with:

```bash
hf download Qwen/Qwen2.5-Math-1.5B --local-dir ./Qwen2.5-Math-1.5B
hf download Qwen/Qwen2.5-0.5B --local-dir ./Qwen2.5-0.5B
```

Verified files:

```text
LICENSE
README.md
config.json
generation_config.json
merges.txt
model.safetensors
tokenizer.json
tokenizer_config.json
vocab.json
```

## Data

Prepared with:

```bash
python data_prep.py
```

Generated files:

```text
gsm8k_train.jsonl: 7473 examples
gsm8k_test.jsonl: 1319 examples
math_train.jsonl: 7500 examples
math_test.jsonl: 5000 examples
```

This milestone uses:

```text
Training source: gsm8k_train.jsonl
Number of source questions: 50
```

## Code Paths Used

Candidate generation:

```text
scripts/log_candidates.py
src/lightr/sampling/candidate_logger.py
src/lightr/features/distribution_features.py
src/lightr/features/token_features.py
src/lightr/features/step_type_features.py
src/lightr/features/candidate_features.py
```

Selector table:

```text
scripts/selector_milestone_table.py
src/lightr/selection/fixed_kl.py
src/lightr/selection/rule_selector.py
configs/adaptive_sampling/qwen15_gsm8k_rule_selector.json
configs/adaptive_sampling/qwen15_gsm8k_rule_selector_balanced.json
```

Compatibility validation:

```text
scripts/check_finetuning_compatibility.py
src/lightr/data/schemas.py
```

Reporting:

```text
scripts/report_selection_stats.py
src/lightr/analysis/selection_report.py
```

## Candidate Logging Command

Command:

```bash
python scripts/log_candidates.py \
  --expert-model ./Qwen2.5-Math-1.5B \
  --amateur-model ./Qwen2.5-0.5B \
  --input gsm8k_train.jsonl \
  --output artifacts/candidates/qwen15_gsm8k_50_candidates.jsonl \
  --checkpoint artifacts/candidates/qwen15_gsm8k_50_checkpoint.jsonl \
  --max-questions 50 \
  --max-new-tokens 64 \
  --device cuda \
  --torch-dtype bfloat16
```

Observed runtime:

```text
50 prompts completed in approximately 11 minutes 17 seconds.
Average wall time: approximately 13.55 seconds per prompt.
```

Observed output:

```json
{
  "prompts_processed": 50,
  "prompts_skipped": 0,
  "candidates_written": 3150
}
```

Interpretation:

- The candidate logger successfully processed 50 GSM8K training questions.
- It produced 3150 candidate prefix positions.
- Unlike original `LightR_sampling.py`, this file contains both selected and
  rejected candidates, enabling repeated selector experiments without re-running
  model inference.

## Selector Table Command

Command:

```bash
python scripts/selector_milestone_table.py \
  --input artifacts/candidates/qwen15_gsm8k_50_candidates.jsonl \
  --output-md artifacts/reports/qwen15_gsm8k_50_selector_milestone.md \
  --output-json artifacts/reports/qwen15_gsm8k_50_selector_milestone.json \
  --selected-output-dir artifacts/selected_samples/qwen15_gsm8k_50_milestone
```

Observed output:

| Selector | Samples | Target Tokens | Selection Rate | KL Mean | KL Median | Main Token Types | Main Step Types |
|---|---:|---:|---:|---:|---:|---|---|
| Fixed KL | 515 | 1133 | 16.35% | 1.2827 | 0.7988 | word:368, punctuation:51, newline:32, currency_or_unit:24, number:14 | setup:266, other:135, calculation:99, transformation:13, verification:2 |
| Rule strict | 37 | 40 | 1.17% | 1.0608 | 0.4934 | number:15, currency_or_unit:14, operator:4, variable:2, final_answer_marker:1 | calculation:21, setup:13, other:2, transformation:1 |
| Rule balanced | 378 | 639 | 12.00% | 0.8862 | 0.6487 | word:320, currency_or_unit:22, number:18, operator:9, variable:6 | setup:223, calculation:96, other:47, transformation:10, verification:2 |

Interpretation:

- Fixed KL selected 515 samples and 1133 contrastive target tokens.
- Rule strict selected only 37 samples and 40 target tokens, making it too small
  for the next fine-tuning comparison. It is useful as a high-precision
  diagnostic selector.
- Rule balanced selected 378 samples and 639 target tokens.
- Rule balanced used approximately 56.4% of the Fixed KL target-token budget:

```text
639 / 1133 = 56.4%
```

- Rule balanced selected approximately 73.4% of Fixed KL's sample count:

```text
378 / 515 = 73.4%
```

Research signal:

- Fixed KL selected many broad tokens, including punctuation and newline tokens.
- Rule balanced removed punctuation/newline/space by design while retaining a
  large number of word, unit, number, operator, and variable tokens.
- This supports the working hypothesis that KL alone is a broad disagreement
  signal, while step/token-aware filtering can compress supervision toward more
  structured reasoning tokens.

## Trainer-Vocabulary Filtering

The first compatibility check revealed that selected targets could contain
Qwen special token id `151643`.

Issue:

```text
tokenizer.vocab_size == 151643
valid trainer label ids: 0 through 151642
special token id 151643 is outside the current LightR_finetuning.py label space
```

Fix:

- `candidate_to_selected_sample(...)` now supports `max_token_id_exclusive`.
- Selected target ids `>= max_token_id_exclusive` are dropped.
- Remaining weights are renormalized.
- `scripts/selector_milestone_table.py` and `scripts/select_candidates.py` can
  use `--model-path ./Qwen2.5-Math-1.5B` to infer `tokenizer.vocab_size`.

Filtered selector table command:

```bash
python scripts/selector_milestone_table.py \
  --input artifacts/candidates/qwen15_gsm8k_50_candidates.jsonl \
  --output-md artifacts/reports/qwen15_gsm8k_50_selector_milestone_filtered.md \
  --output-json artifacts/reports/qwen15_gsm8k_50_selector_milestone_filtered.json \
  --selected-output-dir artifacts/selected_samples/qwen15_gsm8k_50_milestone_filtered \
  --model-path ./Qwen2.5-Math-1.5B
```

Observed filtered output:

| Selector | Samples | Target Tokens | Selection Rate | KL Mean | KL Median | Main Token Types | Main Step Types |
|---|---:|---:|---:|---:|---:|---|---|
| Fixed KL | 515 | 1129 | 16.35% | 1.2827 | 0.7988 | word:368, punctuation:51, newline:32, currency_or_unit:24, number:14 | setup:266, other:135, calculation:99, transformation:13, verification:2 |
| Rule strict | 37 | 39 | 1.17% | 1.0608 | 0.4934 | number:15, currency_or_unit:14, operator:4, variable:2, final_answer_marker:1 | calculation:21, setup:13, other:2, transformation:1 |
| Rule balanced | 378 | 638 | 12.00% | 0.8862 | 0.6487 | word:320, currency_or_unit:22, number:18, operator:9, variable:6 | setup:223, calculation:96, other:47, transformation:10, verification:2 |

Interpretation:

- Filtering dropped only a small number of target tokens.
- All selected outputs are now compatible with the current trainer schema.
- The main research pattern remains unchanged.

## Equal-Budget Fixed-KL Baseline

Rule balanced selected 638 trainable target tokens after filtering.

To create a roughly equal-budget Fixed KL baseline, the following command was
run:

```bash
python scripts/select_candidates.py \
  --input artifacts/candidates/qwen15_gsm8k_50_candidates.jsonl \
  --output artifacts/selected_samples/qwen15_fixedkl_budget639_50_filtered.jsonl \
  --selector fixed_kl \
  --beta 0.4 \
  --max-target-tokens 639 \
  --model-path ./Qwen2.5-Math-1.5B
```

Observed output:

```json
{
  "candidates_seen": 3150,
  "input": "artifacts/candidates/qwen15_gsm8k_50_candidates.jsonl",
  "max_token_id_exclusive": 151643,
  "output": "artifacts/selected_samples/qwen15_fixedkl_budget639_50_filtered.jsonl",
  "reason_counts": {
    "kl_above_beta": 515,
    "kl_below_beta": 2635
  },
  "samples_dropped_after_vocab_filter": 0,
  "samples_selected": 258,
  "samples_selected_before_budget": 515,
  "selected_target_tokens": 635,
  "selection_rate": 0.08190476190476191,
  "selector": "fixed_kl"
}
```

Interpretation:

- Equal-budget Fixed KL uses 258 selected samples and 635 target tokens.
- Rule balanced uses 378 selected samples and 638 target tokens.
- These two are the most important next fine-tuning comparison:

```text
Fixed KL budgeted: 258 samples, 635 target tokens
Rule balanced:     378 samples, 638 target tokens
```

This comparison asks:

> At the same target-token budget, does adaptive step-aware selection beat
> KL-only selection?

## Equal-Budget Selector Reports

The equal-budget Fixed KL dataset was validated and summarized after creation.

Compatibility command:

```bash
python scripts/check_finetuning_compatibility.py \
  --input artifacts/selected_samples/qwen15_fixedkl_budget639_50_filtered.jsonl \
  --model-path ./Qwen2.5-Math-1.5B
```

Observed output:

```json
{
  "compatible": true,
  "error_count": 0,
  "errors": [],
  "input": "artifacts/selected_samples/qwen15_fixedkl_budget639_50_filtered.jsonl",
  "max_token_id": 147601,
  "metadata_records": 258,
  "records_checked": 258,
  "vocab_size": 151643
}
```

Equal-budget Fixed KL report command:

```bash
python scripts/report_selection_stats.py \
  --input artifacts/selected_samples/qwen15_fixedkl_budget639_50_filtered.jsonl
```

Observed output summary:

```json
{
  "selected_sample_records": 258,
  "kl": {
    "max": 14.111364364624023,
    "mean": 1.9876189735508705,
    "median": 1.2580153346061707,
    "min": 0.789900004863739
  },
  "step_types": {
    "calculation": 55,
    "other": 79,
    "setup": 115,
    "transformation": 7,
    "verification": 2
  },
  "token_categories": {
    "currency_or_unit": 11,
    "equals": 1,
    "final_answer_marker": 1,
    "newline": 12,
    "number": 6,
    "punctuation": 36,
    "space": 4,
    "variable": 7,
    "word": 180
  }
}
```

Rule balanced report command:

```bash
python scripts/report_selection_stats.py \
  --input artifacts/selected_samples/qwen15_gsm8k_50_milestone_filtered/rule_balanced.jsonl
```

Observed output summary:

```json
{
  "selected_sample_records": 378,
  "kl": {
    "max": 14.111364364624023,
    "mean": 0.8862482703946255,
    "median": 0.6487404704093933,
    "min": 0.21320560574531555
  },
  "step_types": {
    "calculation": 96,
    "other": 47,
    "setup": 223,
    "transformation": 10,
    "verification": 2
  },
  "token_categories": {
    "currency_or_unit": 22,
    "equals": 2,
    "final_answer_marker": 1,
    "number": 18,
    "operator": 9,
    "variable": 6,
    "word": 320
  }
}
```

Direct comparison:

| Selector | Samples | Target Tokens | KL Mean | KL Median | Punctuation/Newline/Space | Numeric/Operator/Variable/Unit |
|---|---:|---:|---:|---:|---:|---:|
| Fixed KL equal-budget | 258 | 635 | 1.9876 | 1.2580 | 52 | 25 |
| Rule balanced | 378 | 638 | 0.8862 | 0.6487 | 0 | 55 |

Interpretation:

- Equal-budget Fixed KL keeps fewer samples but higher-KL disagreements.
- Rule balanced keeps more distinct training positions at nearly the same
  target-token budget.
- Rule balanced removes punctuation, newline, and space tokens from the selected
  set.
- Rule balanced selects more arithmetic or structure-adjacent tokens:
  number, operator, variable, and currency/unit.
- This is now a clean fine-tuning comparison because both files are
  trainer-compatible and nearly matched by target-token budget.

## Compatibility Checks

Fixed KL full:

```bash
python scripts/check_finetuning_compatibility.py \
  --input artifacts/selected_samples/qwen15_gsm8k_50_milestone_filtered/fixed_kl.jsonl \
  --model-path ./Qwen2.5-Math-1.5B
```

Output:

```json
{
  "compatible": true,
  "error_count": 0,
  "max_token_id": 147601,
  "metadata_records": 515,
  "records_checked": 515,
  "vocab_size": 151643
}
```

Rule strict:

```bash
python scripts/check_finetuning_compatibility.py \
  --input artifacts/selected_samples/qwen15_gsm8k_50_milestone_filtered/rule_strict.jsonl \
  --model-path ./Qwen2.5-Math-1.5B
```

Output:

```json
{
  "compatible": true,
  "error_count": 0,
  "max_token_id": 71518,
  "metadata_records": 37,
  "records_checked": 37,
  "vocab_size": 151643
}
```

Rule balanced:

```bash
python scripts/check_finetuning_compatibility.py \
  --input artifacts/selected_samples/qwen15_gsm8k_50_milestone_filtered/rule_balanced.jsonl \
  --model-path ./Qwen2.5-Math-1.5B
```

Output:

```json
{
  "compatible": true,
  "error_count": 0,
  "max_token_id": 79688,
  "metadata_records": 378,
  "records_checked": 378,
  "vocab_size": 151643
}
```

Conclusion:

- All three filtered milestone selected datasets are compatible with the
  current fine-tuning schema.
- The equal-budget Fixed KL selected dataset is also compatible.
- The special-token issue is resolved.

## Current Research State

The current evidence supports the following preliminary claim:

> Fixed KL identifies broad expert-amateur disagreement, but that disagreement
> includes tokens that are not obviously reasoning-critical. A step/token-aware
> selector can compress supervision by reducing target-token budget while
> preserving a large number of structured reasoning positions.

What has been shown:

- The candidate logger works on real CUDA models.
- The new candidate JSONL contains reusable candidate-level features.
- Fixed KL, Rule strict, and Rule balanced select very different supervision
  subsets.
- Rule balanced uses roughly 56% of the Fixed KL target-token budget.
- Rule balanced removes punctuation/newline/space from selected supervision.
- Filtered selected outputs are compatible with `LightR_finetuning.py`.
- The equal-budget Fixed KL baseline is now built, summarized, and compatible.
- The main next comparison has a clean input pair:
  - Fixed KL equal-budget: 258 samples, 635 target tokens.
  - Rule balanced: 378 samples, 638 target tokens.

What has not yet been shown:

- That Rule balanced improves downstream reasoning accuracy.
- That Rule balanced beats equal-budget Fixed KL.
- That the trend holds at 1000-question scale.
- That the method transfers to MATH, SVAMP, ASDiv, or other benchmarks.

## Next Experiment

Do not use Rule strict for fine-tuning yet. It is too small.

The next fine-tuning comparison should use:

1. Fixed KL full:

```text
artifacts/selected_samples/qwen15_gsm8k_50_milestone_filtered/fixed_kl.jsonl
```

2. Fixed KL equal-budget:

```text
artifacts/selected_samples/qwen15_fixedkl_budget639_50_filtered.jsonl
```

3. Rule balanced:

```text
artifacts/selected_samples/qwen15_gsm8k_50_milestone_filtered/rule_balanced.jsonl
```

Primary research comparison:

```text
Fixed KL equal-budget vs Rule balanced
```

Research question:

> At the same target-token budget, does adaptive step-aware selection provide
> better fine-tuning signal than KL-only selection?

## Suggested Next Commands

The pre-fine-tuning selector reports and compatibility checks are complete.

The configurable runner is:

```text
scripts/run_contrastive_finetune.py
src/lightr/training/contrastive_finetuning.py
```

It keeps `LightR_finetuning.py` unchanged and writes:

```text
run_config.json
train_metrics.json
final_adapter/
```

The first smoke test should train on the two equal-budget files for a very small
number of steps:

```text
artifacts/selected_samples/qwen15_fixedkl_budget639_50_filtered.jsonl
artifacts/selected_samples/qwen15_gsm8k_50_milestone_filtered/rule_balanced.jsonl
```

Smoke run, Fixed KL equal-budget:

```bash
python scripts/run_contrastive_finetune.py \
  --model-path ./Qwen2.5-Math-1.5B \
  --train-jsonl artifacts/selected_samples/qwen15_fixedkl_budget639_50_filtered.jsonl \
  --output-dir artifacts/finetune_smoke/qwen15_fixedkl_budget635_50_seed0 \
  --max-steps 5 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 5e-5 \
  --max-length 512 \
  --torch-dtype bfloat16 \
  --logging-steps 1 \
  --save-steps 5 \
  --seed 0
```

Smoke run, Rule balanced:

```bash
python scripts/run_contrastive_finetune.py \
  --model-path ./Qwen2.5-Math-1.5B \
  --train-jsonl artifacts/selected_samples/qwen15_gsm8k_50_milestone_filtered/rule_balanced.jsonl \
  --output-dir artifacts/finetune_smoke/qwen15_rule_balanced_budget638_50_seed0 \
  --max-steps 5 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 5e-5 \
  --max-length 512 \
  --torch-dtype bfloat16 \
  --logging-steps 1 \
  --save-steps 5 \
  --seed 0
```

Observed smoke results:

| Selector | Steps | Train Samples | Train Loss | Runtime | Samples/Sec | Adapter Output |
|---|---:|---:|---:|---:|---:|---|
| Fixed KL equal-budget | 5 | 258 | 0.2947 | 12.86s | 0.389 | `artifacts/finetune_smoke/qwen15_fixedkl_budget635_50_seed0/final_adapter` |
| Rule balanced | 5 | 378 | 0.1374 | 11.10s | 0.451 | `artifacts/finetune_smoke/qwen15_rule_balanced_budget638_50_seed0/final_adapter` |

Fixed KL equal-budget observed metrics:

```json
{
  "epoch": 0.01937984496124031,
  "final_adapter_dir": "artifacts/finetune_smoke/qwen15_fixedkl_budget635_50_seed0/final_adapter",
  "total_flos": 5421452215296.0,
  "train_loss": 0.29469547867774964,
  "train_runtime": 12.8563,
  "train_samples": 258,
  "train_samples_per_second": 0.389,
  "train_steps_per_second": 0.389
}
```

Rule balanced observed metrics:

```json
{
  "epoch": 0.013227513227513227,
  "final_adapter_dir": "artifacts/finetune_smoke/qwen15_rule_balanced_budget638_50_seed0/final_adapter",
  "total_flos": 4091662049280.0,
  "train_loss": 0.13740580775775016,
  "train_runtime": 11.0981,
  "train_samples": 378,
  "train_samples_per_second": 0.451,
  "train_steps_per_second": 0.451
}
```

Interpretation:

- Both selected datasets can be loaded by the runner.
- Both LoRA smoke runs complete and save adapters.
- Both runs produce finite training loss.
- The lower 5-step training loss for Rule balanced is not yet an accuracy
  result; it only confirms the training path works and suggests the selected
  targets are learnable.

Since both smoke runs finished and `train_loss` is finite, run the matched
comparison with more steps:

```bash
python scripts/run_contrastive_finetune.py \
  --model-path ./Qwen2.5-Math-1.5B \
  --train-jsonl artifacts/selected_samples/qwen15_fixedkl_budget639_50_filtered.jsonl \
  --output-dir artifacts/finetune/qwen15_fixedkl_budget635_50_seed0 \
  --max-steps 100 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-5 \
  --max-length 512 \
  --torch-dtype bfloat16 \
  --logging-steps 5 \
  --save-steps 50 \
  --seed 0
```

```bash
python scripts/run_contrastive_finetune.py \
  --model-path ./Qwen2.5-Math-1.5B \
  --train-jsonl artifacts/selected_samples/qwen15_gsm8k_50_milestone_filtered/rule_balanced.jsonl \
  --output-dir artifacts/finetune/qwen15_rule_balanced_budget638_50_seed0 \
  --max-steps 100 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-5 \
  --max-length 512 \
  --torch-dtype bfloat16 \
  --logging-steps 5 \
  --save-steps 50 \
  --seed 0
```

Observed 100-step matched training results:

| Selector | Steps | Train Samples | Train Loss | Runtime | Samples/Sec | Adapter Output |
|---|---:|---:|---:|---:|---:|---|
| Fixed KL equal-budget | 100 | 258 | 2.5245 | 343.83s | 1.163 | `artifacts/finetune/qwen15_fixedkl_budget635_50_seed0/final_adapter` |
| Rule balanced | 100 | 378 | 0.8736 | 351.82s | 1.137 | `artifacts/finetune/qwen15_rule_balanced_budget638_50_seed0/final_adapter` |

Fixed KL equal-budget observed metrics:

```json
{
  "epoch": 1.5426356589147288,
  "final_adapter_dir": "artifacts/finetune/qwen15_fixedkl_budget635_50_seed0/final_adapter",
  "total_flos": 351269186930688.0,
  "train_loss": 2.5245076608657837,
  "train_runtime": 343.8342,
  "train_samples": 258,
  "train_samples_per_second": 1.163,
  "train_steps_per_second": 0.291
}
```

Rule balanced observed metrics:

```json
{
  "epoch": 1.052910052910053,
  "final_adapter_dir": "artifacts/finetune/qwen15_rule_balanced_budget638_50_seed0/final_adapter",
  "total_flos": 358657784361984.0,
  "train_loss": 0.8736045193672181,
  "train_runtime": 351.8211,
  "train_samples": 378,
  "train_samples_per_second": 1.137,
  "train_steps_per_second": 0.284
}
```

Interpretation:

- Both 100-step matched training runs completed successfully.
- The Rule balanced training loss is substantially lower than the equal-budget
  Fixed KL training loss.
- This is still not enough to claim a reasoning improvement. Training loss is
  measured on different selected examples, so the decisive result must come
  from held-out GSM8K or other benchmark accuracy.

## Evaluation Commands

For the first accuracy check, use the lightweight GSM8K adapter evaluator:

```text
scripts/evaluate_gsm8k_adapter.py
src/lightr/evaluation/gsm8k_eval.py
```

Evaluate the base expert on the first 100 GSM8K test examples:

```bash
python scripts/evaluate_gsm8k_adapter.py \
  --model-path ./Qwen2.5-Math-1.5B \
  --input-jsonl gsm8k_test.jsonl \
  --output-jsonl artifacts/eval/gsm8k_100/base_qwen15.jsonl \
  --summary-json artifacts/eval/gsm8k_100/base_qwen15_summary.json \
  --max-examples 100 \
  --max-new-tokens 256 \
  --torch-dtype bfloat16 \
  --seed 0
```

Evaluate Fixed KL equal-budget adapter:

```bash
python scripts/evaluate_gsm8k_adapter.py \
  --model-path ./Qwen2.5-Math-1.5B \
  --adapter-path artifacts/finetune/qwen15_fixedkl_budget635_50_seed0/final_adapter \
  --input-jsonl gsm8k_test.jsonl \
  --output-jsonl artifacts/eval/gsm8k_100/fixedkl_budget635.jsonl \
  --summary-json artifacts/eval/gsm8k_100/fixedkl_budget635_summary.json \
  --max-examples 100 \
  --max-new-tokens 256 \
  --torch-dtype bfloat16 \
  --seed 0
```

Observed evaluation results so far:

| Model | Adapter | Correct | Total | Accuracy | Runtime |
|---|---|---:|---:|---:|---:|
| Base Qwen2.5-Math-1.5B | none | 19 | 100 | 19.00% | 1:15:07 |
| Fixed KL equal-budget | `artifacts/finetune/qwen15_fixedkl_budget635_50_seed0/final_adapter` | 18 | 100 | 18.00% | 1:43:45 |

Base observed summary:

```json
{
  "accuracy": 0.19,
  "correct": 19,
  "output_jsonl": "artifacts/eval/gsm8k_100/base_qwen15.jsonl",
  "total": 100
}
```

Fixed KL equal-budget observed summary:

```json
{
  "accuracy": 0.18,
  "correct": 18,
  "output_jsonl": "artifacts/eval/gsm8k_100/fixedkl_budget635.jsonl",
  "total": 100
}
```

Current interpretation:

- On the first 100 GSM8K test examples, equal-budget Fixed KL is slightly below
  the base model: 18% vs 19%.
- This does not support a Fixed KL improvement at this small 50-question
  training scale.
- The key comparison is still incomplete until Rule balanced is evaluated with
  the same settings.

Evaluate Rule balanced adapter:

```bash
python scripts/evaluate_gsm8k_adapter.py \
  --model-path ./Qwen2.5-Math-1.5B \
  --adapter-path artifacts/finetune/qwen15_rule_balanced_budget638_50_seed0/final_adapter \
  --input-jsonl gsm8k_test.jsonl \
  --output-jsonl artifacts/eval/gsm8k_100/rule_balanced_budget638.jsonl \
  --summary-json artifacts/eval/gsm8k_100/rule_balanced_budget638_summary.json \
  --max-examples 100 \
  --max-new-tokens 256 \
  --torch-dtype bfloat16 \
  --seed 0
```

The first evaluation comparison should be:

```text
Base expert vs Fixed KL equal-budget vs Rule balanced
```

Observed GSM8K-100 evaluation results:

| Model | Adapter | Correct | Total | Accuracy |
|---|---|---:|---:|---:|
| Base Qwen2.5-Math-1.5B | none | 19 | 100 | 19% |
| Fixed KL equal-budget | `qwen15_fixedkl_budget635_50_seed0/final_adapter` | 18 | 100 | 18% |
| Rule balanced | `qwen15_rule_balanced_budget638_50_seed0/final_adapter` | 16 | 100 | 16% |

Evaluation runtimes:

```text
Base: 1:15:07
Fixed KL equal-budget: 1:43:45
Rule balanced: 1:48:13
```

Interpretation:

- The 50-question pilot did not improve held-out GSM8K accuracy.
- Equal-budget Fixed KL slightly underperformed the base model on this
  100-example subset.
- Rule balanced underperformed both the base model and the equal-budget Fixed KL
  adapter on this 100-example subset.
- The lower training loss for Rule balanced did not translate into better
  held-out accuracy in this pilot.
- This does not invalidate the research idea, but it means the current selector
  and tiny 50-question training setup are not enough.

Likely explanations:

- The training set is extremely small: only 50 GSM8K source questions.
- The LoRA run may overfit because each method trained for roughly one epoch or
  more over a very small selected set.
- Rule balanced still selects many `word` tokens, so it is not yet strongly
  focused on arithmetic or equation-critical disagreements.
- The lightweight evaluator is useful for iteration, but final paper numbers
  should use a stronger and standardized evaluation path.
- The base evaluation accuracy is low enough that prompt format and answer
  extraction should be inspected before drawing strong conclusions.

Immediate next diagnostic:

```bash
python scripts/compare_gsm8k_eval_outputs.py \
  --base artifacts/eval/gsm8k_100/base_qwen15.jsonl \
  --fixed artifacts/eval/gsm8k_100/fixedkl_budget635.jsonl \
  --rule artifacts/eval/gsm8k_100/rule_balanced_budget638.jsonl \
  --output artifacts/eval/gsm8k_100/comparison_summary.json \
  --max-examples 5
```

Observed comparison summary:

```json
{
  "accuracy": {
    "base": 0.19,
    "fixed_kl_equal_budget": 0.18,
    "rule_balanced": 0.16
  },
  "correct_counts": {
    "base": 19,
    "fixed_kl_equal_budget": 18,
    "rule_balanced": 16
  },
  "overlap_counts": {
    "all_correct": 10,
    "all_wrong": 73,
    "fixed_correct_rule_wrong": 6,
    "fixed_improves_over_base": 4,
    "fixed_regresses_from_base": 5,
    "rule_correct_fixed_wrong": 4,
    "rule_improves_over_base": 6,
    "rule_regresses_from_base": 9
  },
  "total_common": 100
}
```

Comparison interpretation:

- The adapters are not simply failing on exactly the same examples.
- Fixed KL improves 4 base-wrong examples but regresses 5 base-correct
  examples.
- Rule balanced improves 6 base-wrong examples but regresses 9 base-correct
  examples.
- Rule balanced solves 4 examples that Fixed KL misses, but Fixed KL solves 6
  examples that Rule balanced misses.
- This means there is useful signal in the adaptive-selected examples, but the
  current selector/training recipe introduces too many regressions.

Important qualitative failure mode:

- Several adapter outputs contain repeated prompt-like or artifact strings such
  as `Please reason step by step...`, `ager`, and repeated copied questions.
- Some examples produce the correct final number despite strange formatting,
  but others degrade into prompt copying or unrelated problem text.
- This points to overfitting or distribution shift from the tiny selected
  training set, not just answer-extraction noise.

Next experimental direction:

- Do not scale Rule balanced unchanged.
- First inspect the comparison summary and a few generated outputs.
- Create a more reasoning-focused selector variant that reduces generic `word`
  tokens and emphasizes number, operator, equals, variable, unit, and final
  answer tokens.
- Try a smaller learning rate or fewer steps for 50-question training, such as
  20 to 50 steps, before scaling.
- Then run a larger candidate log, ideally 500 to 1000 GSM8K questions, because
  the current pilot is too small for a stable accuracy claim.

Reasoning-focused selector config:

```text
configs/adaptive_sampling/qwen15_gsm8k_rule_selector_reasoning_focus.json
```

This selector removes generic `word` tokens and relaxes confidence/KL thresholds
relative to Rule strict, aiming to produce more than Rule strict while avoiding
the word-heavy behavior of Rule balanced.

Run it on the existing 50-question candidate log:

```bash
python scripts/select_candidates.py \
  --input artifacts/candidates/qwen15_gsm8k_50_candidates.jsonl \
  --output artifacts/selected_samples/qwen15_reasoning_focus_50.jsonl \
  --selector rule \
  --config configs/adaptive_sampling/qwen15_gsm8k_rule_selector_reasoning_focus.json \
  --model-path ./Qwen2.5-Math-1.5B
```

Summarize it:

```bash
python scripts/report_selection_stats.py \
  --input artifacts/selected_samples/qwen15_reasoning_focus_50.jsonl
```

Check compatibility:

```bash
python scripts/check_finetuning_compatibility.py \
  --input artifacts/selected_samples/qwen15_reasoning_focus_50.jsonl \
  --model-path ./Qwen2.5-Math-1.5B
```

Observed reasoning-focused selection result:

```json
{
  "candidates_seen": 3150,
  "samples_selected": 83,
  "samples_selected_before_budget": 83,
  "selected_target_tokens": 112,
  "selection_rate": 0.026349206349206348,
  "selector": "rule_v3_reasoning_focus"
}
```

Observed reasoning-focused report:

```json
{
  "selected_sample_records": 83,
  "kl": {
    "max": 14.111364364624023,
    "mean": 0.7953808890229248,
    "median": 0.41369885206222534,
    "min": 0.10757744312286377
  },
  "step_types": {
    "calculation": 46,
    "other": 7,
    "setup": 28,
    "transformation": 2
  },
  "token_categories": {
    "currency_or_unit": 33,
    "equals": 4,
    "final_answer_marker": 2,
    "number": 25,
    "operator": 10,
    "variable": 9
  }
}
```

Compatibility result:

```json
{
  "compatible": true,
  "error_count": 0,
  "max_token_id": 71518,
  "metadata_records": 83,
  "records_checked": 83,
  "vocab_size": 151643
}
```

Interpretation:

- The selector behavior is now much cleaner than Rule balanced.
- It selected no generic `word`, `punctuation`, `space`, or `newline` tokens.
- Most selected samples are calculation/setup tokens.
- The selected categories are directly aligned with the intended hypothesis:
  number, operator, equals, variable, unit, and final-answer marker.
- However, the dataset is small: 83 samples and 112 target tokens.
- This should be tested only as a conservative diagnostic, not as the next
  paper-scale run.

Recommended diagnostic training run:

```bash
python scripts/run_contrastive_finetune.py \
  --model-path ./Qwen2.5-Math-1.5B \
  --train-jsonl artifacts/selected_samples/qwen15_reasoning_focus_50.jsonl \
  --output-dir artifacts/finetune/qwen15_reasoning_focus_50_steps20_seed0 \
  --max-steps 20 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-5 \
  --max-length 512 \
  --torch-dtype bfloat16 \
  --logging-steps 2 \
  --save-steps 20 \
  --seed 0
```

Observed reasoning-focused 20-step training result:

```json
{
  "epoch": 0.963855421686747,
  "final_adapter_dir": "artifacts/finetune/qwen15_reasoning_focus_50_steps20_seed0/final_adapter",
  "total_flos": 73130590550016.0,
  "train_loss": 0.6986170038580894,
  "train_runtime": 75.1375,
  "train_samples": 83,
  "train_samples_per_second": 1.065,
  "train_steps_per_second": 0.266
}
```

Recommended diagnostic evaluation:

```bash
python scripts/evaluate_gsm8k_adapter.py \
  --model-path ./Qwen2.5-Math-1.5B \
  --adapter-path artifacts/finetune/qwen15_reasoning_focus_50_steps20_seed0/final_adapter \
  --input-jsonl gsm8k_test.jsonl \
  --output-jsonl artifacts/eval/gsm8k_100/reasoning_focus_steps20.jsonl \
  --summary-json artifacts/eval/gsm8k_100/reasoning_focus_steps20_summary.json \
  --max-examples 100 \
  --max-new-tokens 256 \
  --torch-dtype bfloat16 \
  --seed 0
```

Faster diagnostic evaluation option:

The full `--max-examples 100 --max-new-tokens 256` evaluator is slow because it
generates long chain-of-thought outputs. For quick iteration, use 30 examples,
128 new tokens, and batched generation:

```bash
python scripts/evaluate_gsm8k_adapter.py \
  --model-path ./Qwen2.5-Math-1.5B \
  --adapter-path artifacts/finetune/qwen15_reasoning_focus_50_steps20_seed0/final_adapter \
  --input-jsonl gsm8k_test.jsonl \
  --output-jsonl artifacts/eval/gsm8k_30/reasoning_focus_steps20.jsonl \
  --summary-json artifacts/eval/gsm8k_30/reasoning_focus_steps20_summary.json \
  --max-examples 30 \
  --max-new-tokens 128 \
  --batch-size 4 \
  --torch-dtype bfloat16 \
  --seed 0
```

Use the 100-example evaluation only after the 30-example diagnostic does not
show obvious degeneration.

Observed reasoning-focused fast evaluation result:

```json
{
  "accuracy": 0.06666666666666667,
  "correct": 2,
  "total": 30,
  "output_jsonl": "artifacts/eval/gsm8k_30/reasoning_focus_steps20.jsonl"
}
```

Interpretation:

- The reasoning-focused selector trained successfully, but its first fast
  diagnostic accuracy is poor: 2/30.
- This result should not be compared directly against the previous GSM8K-100
  results because this evaluation used a different setting:
  - 30 examples instead of 100,
  - 128 generated tokens instead of 256,
  - batched generation.
- The next fair diagnostic is to evaluate Base, Fixed KL equal-budget, and Rule
  balanced on the same GSM8K-30 setting. If the base model is also low on this
  shortened setting, the fast diagnostic is too harsh. If only reasoning-focus
  is low, then `rule_v3_reasoning_focus` is not useful at the 50-question scale.

Fair GSM8K-30 comparison commands:

```bash
python scripts/evaluate_gsm8k_adapter.py \
  --model-path ./Qwen2.5-Math-1.5B \
  --input-jsonl gsm8k_test.jsonl \
  --output-jsonl artifacts/eval/gsm8k_30/base_qwen15.jsonl \
  --summary-json artifacts/eval/gsm8k_30/base_qwen15_summary.json \
  --max-examples 30 \
  --max-new-tokens 128 \
  --batch-size 4 \
  --torch-dtype bfloat16 \
  --seed 0
```

```bash
python scripts/evaluate_gsm8k_adapter.py \
  --model-path ./Qwen2.5-Math-1.5B \
  --adapter-path artifacts/finetune/qwen15_fixedkl_budget635_50_seed0/final_adapter \
  --input-jsonl gsm8k_test.jsonl \
  --output-jsonl artifacts/eval/gsm8k_30/fixedkl_budget635.jsonl \
  --summary-json artifacts/eval/gsm8k_30/fixedkl_budget635_summary.json \
  --max-examples 30 \
  --max-new-tokens 128 \
  --batch-size 4 \
  --torch-dtype bfloat16 \
  --seed 0
```

```bash
python scripts/evaluate_gsm8k_adapter.py \
  --model-path ./Qwen2.5-Math-1.5B \
  --adapter-path artifacts/finetune/qwen15_rule_balanced_budget638_50_seed0/final_adapter \
  --input-jsonl gsm8k_test.jsonl \
  --output-jsonl artifacts/eval/gsm8k_30/rule_balanced_budget638.jsonl \
  --summary-json artifacts/eval/gsm8k_30/rule_balanced_budget638_summary.json \
  --max-examples 30 \
  --max-new-tokens 128 \
  --batch-size 4 \
  --torch-dtype bfloat16 \
  --seed 0
```

Observed fair GSM8K-30 comparison:

| Model | Correct | Total | Accuracy | Runtime |
|---|---:|---:|---:|---:|
| Base Qwen2.5-Math-1.5B | 1 | 30 | 3.33% | 3:40 |
| Fixed KL equal-budget | 2 | 30 | 6.67% | 4:49 |
| Rule balanced | 1 | 30 | 3.33% | 5:04 |
| Rule reasoning-focus | 2 | 30 | 6.67% | 4:55 |

Interpretation:

- The fast GSM8K-30 setting is too harsh to use as a quality decision by
  itself.
- Base accuracy drops from 19/100 under `max_new_tokens=256` to 1/30 under
  `max_new_tokens=128`.
- That suggests many examples need longer generation to reach the final answer,
  or the shortened run cuts off answer formatting.
- Reasoning-focus does not look worse than the other adapters under the fast
  diagnostic, but the metric is too low to distinguish methods reliably.
- Future quick checks should use batched generation with `max_new_tokens=256`,
  not 128.

Recommended next quick evaluation:

Use the first 30 examples again, but restore `max_new_tokens=256` and keep
`batch_size=4`:

```bash
python scripts/evaluate_gsm8k_adapter.py \
  --model-path ./Qwen2.5-Math-1.5B \
  --input-jsonl gsm8k_test.jsonl \
  --output-jsonl artifacts/eval/gsm8k_30_tok256/base_qwen15.jsonl \
  --summary-json artifacts/eval/gsm8k_30_tok256/base_qwen15_summary.json \
  --max-examples 30 \
  --max-new-tokens 256 \
  --batch-size 4 \
  --torch-dtype bfloat16 \
  --seed 0
```

```bash
python scripts/evaluate_gsm8k_adapter.py \
  --model-path ./Qwen2.5-Math-1.5B \
  --adapter-path artifacts/finetune/qwen15_reasoning_focus_50_steps20_seed0/final_adapter \
  --input-jsonl gsm8k_test.jsonl \
  --output-jsonl artifacts/eval/gsm8k_30_tok256/reasoning_focus_steps20.jsonl \
  --summary-json artifacts/eval/gsm8k_30_tok256/reasoning_focus_steps20_summary.json \
  --max-examples 30 \
  --max-new-tokens 256 \
  --batch-size 4 \
  --torch-dtype bfloat16 \
  --seed 0
```

Observed GSM8K-30, 256-token diagnostic:

| Model | Correct | Total | Accuracy | Runtime |
|---|---:|---:|---:|---:|
| Base Qwen2.5-Math-1.5B | 3 | 30 | 10.00% | 7:29 |
| Fixed KL equal-budget | 5 | 30 | 16.67% | 9:56 |
| Rule balanced | 4 | 30 | 13.33% | 9:47 |
| Rule reasoning-focus, 20 steps | 4 | 30 | 13.33% | 9:55 |

Interpretation:

- Restoring `max_new_tokens=256` improves the fast diagnostic compared with
  the 128-token setting.
- Fixed KL equal-budget is best on this small same-setting subset: 5/30.
- Rule balanced and reasoning-focus tie at 4/30.
- Reasoning-focus is +1 correct over base, but it does not beat the fixed-KL
  baseline.
- This is not enough for a research claim.
- The best interpretation is that adaptive selection has some signal, but the
  current 50-question setup does not yet outperform KL-only selection.

Completed same-setting commands:

```bash
python scripts/evaluate_gsm8k_adapter.py \
  --model-path ./Qwen2.5-Math-1.5B \
  --adapter-path artifacts/finetune/qwen15_fixedkl_budget635_50_seed0/final_adapter \
  --input-jsonl gsm8k_test.jsonl \
  --output-jsonl artifacts/eval/gsm8k_30_tok256/fixedkl_budget635.jsonl \
  --summary-json artifacts/eval/gsm8k_30_tok256/fixedkl_budget635_summary.json \
  --max-examples 30 \
  --max-new-tokens 256 \
  --batch-size 4 \
  --torch-dtype bfloat16 \
  --seed 0
```

```bash
python scripts/evaluate_gsm8k_adapter.py \
  --model-path ./Qwen2.5-Math-1.5B \
  --adapter-path artifacts/finetune/qwen15_rule_balanced_budget638_50_seed0/final_adapter \
  --input-jsonl gsm8k_test.jsonl \
  --output-jsonl artifacts/eval/gsm8k_30_tok256/rule_balanced_budget638.jsonl \
  --summary-json artifacts/eval/gsm8k_30_tok256/rule_balanced_budget638_summary.json \
  --max-examples 30 \
  --max-new-tokens 256 \
  --batch-size 4 \
  --torch-dtype bfloat16 \
  --seed 0
```

Decision from this diagnostic:

- Do not tune further on the 50-question setup.
- Do not scale Rule balanced unchanged.
- Do not claim reasoning-focus improves over Fixed KL.
- The next meaningful experiment is a larger candidate log, preferably 500
  GSM8K training questions, followed by equal-budget selector comparisons.

Do not claim the research result until both have been fine-tuned and evaluated
with the same settings.

## Pivot: Correctness-Gated Candidate Logging

The 50-question pilot showed that selector design alone is not enough. A
reasoning-critical token can still be harmful if it comes from an incorrect or
degenerate expert trajectory.

The candidate logger now supports two gates:

```text
--require-correct
--reject-degenerate
```

Correctness gate:

- Uses the input `answer` field.
- Extracts the GSM8K gold answer.
- Extracts the expert generation's predicted answer.
- Logs candidates only when the expert final answer is correct.

Degeneration gate:

- Rejects obvious prompt-copying and repetition artifacts.
- Detects repeated system prompt text.
- Detects copied question text.
- Detects repeated artifact strings such as `ager`.
- Detects repeated lines and repeated words.

This is the next publishability-oriented version of the method:

> Correctness-gated, step-aware disagreement selection.

Recommended 500-question candidate logging command:

```bash
python scripts/log_candidates.py \
  --expert-model ./Qwen2.5-Math-1.5B \
  --amateur-model ./Qwen2.5-0.5B \
  --input gsm8k_train.jsonl \
  --output artifacts/candidates/qwen15_gsm8k_500_candidates_correct_gated.jsonl \
  --checkpoint artifacts/candidates/qwen15_gsm8k_500_checkpoint_correct_gated.jsonl \
  --max-questions 500 \
  --max-new-tokens 128 \
  --device cuda \
  --torch-dtype bfloat16 \
  --answer-field answer \
  --require-correct \
  --reject-degenerate
```

Expected behavior:

- The number of accepted prompts will be less than 500.
- This is intentional; accepted prompts should have cleaner supervision.
- The summary will include:
  - `prompts_processed`
  - `prompts_skipped`
  - `prompts_rejected_incorrect`
  - `prompts_rejected_degenerate`
  - `candidates_written`

After candidate logging, run selector summaries before training:

```bash
python scripts/selector_milestone_table.py \
  --input artifacts/candidates/qwen15_gsm8k_500_candidates_correct_gated.jsonl \
  --output-md artifacts/reports/qwen15_gsm8k_500_correct_gated_selector_milestone.md \
  --output-json artifacts/reports/qwen15_gsm8k_500_correct_gated_selector_milestone.json \
  --selected-output-dir artifacts/selected_samples/qwen15_gsm8k_500_correct_gated_milestone \
  --model-path ./Qwen2.5-Math-1.5B
```

Also run the reasoning-focused selector:

```bash
python scripts/select_candidates.py \
  --input artifacts/candidates/qwen15_gsm8k_500_candidates_correct_gated.jsonl \
  --output artifacts/selected_samples/qwen15_reasoning_focus_500_correct_gated.jsonl \
  --selector rule \
  --config configs/adaptive_sampling/qwen15_gsm8k_rule_selector_reasoning_focus.json \
  --model-path ./Qwen2.5-Math-1.5B
```

Do not fine-tune until the selector table shows enough selected samples and a
reasonable token distribution.
