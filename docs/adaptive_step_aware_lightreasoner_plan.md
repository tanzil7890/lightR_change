# Adaptive Step-Aware LightReasoner Implementation Plan

## Purpose

This document is a project-local research and engineering guide for extending
LightReasoner into an adaptive, step-aware contrastive reasoning distillation
framework.

The target research question is:

> Can we learn when expert-amateur disagreement is actually useful, instead of
> selecting reasoning tokens with a fixed KL threshold?

The target paper framing is:

> Learning to select reasoning-critical tokens for contrastive distillation.

The current LightReasoner implementation is intentionally compact. This guide
keeps that project intact and proposes an additive implementation path. Existing
scripts should continue to run as before. New functionality should be introduced
through separate modules, configuration files, and optional command-line flags.

## Implementation Status

Last updated: 2026-04-25

Completed in the current project:

- Added isolated `src/lightr/` package without modifying original
  `LightR_sampling.py` or `LightR_finetuning.py`.
- Added JSONL helpers and schema validation:
  - `src/lightr/data/jsonl.py`
  - `src/lightr/data/schemas.py`
- Added deterministic feature extraction:
  - `src/lightr/features/distribution_features.py`
  - `src/lightr/features/token_features.py`
  - `src/lightr/features/step_type_features.py`
  - `src/lightr/features/candidate_features.py`
- Added selectors:
  - `src/lightr/selection/fixed_kl.py`
  - `src/lightr/selection/rule_selector.py`
- Added real model-backed candidate logging entrypoint:
  - `src/lightr/sampling/candidate_logger.py`
  - `scripts/log_candidates.py`
- Added selector and reporting entrypoints:
  - `scripts/select_candidates.py`
  - `scripts/report_selection_stats.py`
- Added fixed-KL reproduction and trainer-compatibility checks:
  - `scripts/compare_lightr_samples.py`
  - `scripts/check_finetuning_compatibility.py`
- Added bucket-ablation infrastructure:
  - `src/lightr/analysis/bucket_ablation.py`
  - `scripts/build_bucket_ablation.py`
- Added lightweight learned-selector infrastructure without new dependencies:
  - `src/lightr/selection/learned_selector.py`
  - `src/lightr/training/selector_training.py`
  - `scripts/train_selector.py`
- Added example selector configs:
  - `configs/adaptive_sampling/qwen15_gsm8k_fixed_kl.json`
  - `configs/adaptive_sampling/qwen15_gsm8k_rule_selector.json`
- Added smoke/unit tests:
  - `tests/test_adaptive_features_and_selection.py`
- Added `artifacts/`, `__pycache__/`, and `*.pyc` to `.gitignore` for
  generated experiment outputs and Python cache files.

Verified:

- `python -m unittest discover -s tests -v` passes.
- `python -m compileall src scripts tests` passes.
- `scripts/select_candidates.py --help` works.
- `scripts/log_candidates.py --help` works.
- `scripts/train_selector.py --help` works.
- `scripts/build_bucket_ablation.py --help` works.
- `scripts/report_selection_stats.py` can read real pre-collected selected
  LightReasoner samples from `LRsamples/LR_samples.zip`; verified on
  `LR_Qw1.5_gsm8k_torch2.8.txt` with 16,719 selected records.
- `scripts/compare_lightr_samples.py` self-comparison on the same real sample
  file reports 100% overlap across 16,719 records.
- `scripts/check_finetuning_compatibility.py` validates the first 100 real
  pre-collected selected samples as compatible with the current trainer schema.

Pending:

- A real candidate logging run with local expert and amateur model paths.
- A direct fixed-KL reproduction comparison against `LightR_sampling.py` on the
  same prompts.
- LoRA fine-tuning and benchmark evaluation of adaptive selected samples.
- Learned selector training on real candidate logs.

Environment note:

- No local model weight folders were found in this workspace.
- The current machine reports no CUDA and no Apple MPS device through PyTorch.
  Full candidate generation and fine-tuning should be run on a GPU machine or
  after local model paths are added.

## Current Project Understanding

### High-level method

LightReasoner improves a stronger expert model using a weaker amateur model.
The amateur is not used as a teacher in the normal knowledge-distillation sense.
Instead, the amateur exposes where the expert has a useful advantage.

The existing flow is:

1. Load a reasoning dataset such as GSM8K.
2. Let the expert generate a reasoning trajectory.
3. For each generated prefix, compare the expert and amateur next-token
   distributions.
4. Keep prefixes where the expert-amateur KL divergence is above a fixed
   threshold.
5. Convert the expert-amateur contrast into a soft target distribution.
6. Fine-tune the expert with LoRA using KL loss against those soft labels.
7. Merge the LoRA adapter if a standalone model is needed.
8. Evaluate with the Qwen2.5-Math evaluation toolkit.

The key limitation for this research direction is that token selection is
currently controlled by a fixed scalar threshold:

```python
if kl_div < beta:
    continue
```

This treats all disagreements as equally meaningful once they cross the
threshold. In practice, disagreement usefulness likely depends on confidence,
reasoning position, token type, and local reasoning role.

### Repository file-by-file map

#### README.md

Primary project documentation.

Important content:

- Defines the LightReasoner idea.
- Describes expert-amateur KL-guided selection.
- Reports main results across GSM8K, MATH, SVAMP, ASDiv, Minerva Math,
  OlympiadBench, and MMLU STEM.
- Describes the current fixed-threshold sampling workflow.
- Emphasizes that model pairing matters: the amateur should be weaker but still
  competent.
- Describes the pre-collected samples in `LRsamples/`.

Role in the extension:

- Keep unchanged during early development.
- Later, add a short section only after the adaptive selector is stable.
- Do not rewrite the main README while experiments are still exploratory.

#### README-zh.md

Chinese version of the README.

Role in the extension:

- Leave unchanged until the English README update is final.
- If the project becomes public-facing, mirror only stable documentation changes.

#### Communication.md

Community/contact information.

Role in the extension:

- No changes needed.

#### requirements.txt

Current dependencies:

- `transformers`
- `accelerate`
- `datasets`
- `huggingface_hub`
- `tqdm`
- `peft`

Role in the extension:

- Avoid adding heavy dependencies early.
- Prefer standard library plus PyTorch and Transformers.
- If a learned selector uses scikit-learn, add it only when the selector is
  actually implemented and justified.
- For an industry-standard implementation, use optional dependency groups later
  rather than growing the base requirements unnecessarily.

#### data_prep.py

Downloads and serializes GSM8K and MATH data into JSONL files.

Current behavior:

- Writes `gsm8k_train.jsonl` and `gsm8k_test.jsonl`.
- Writes `math_train.jsonl` and `math_test.jsonl`.
- Uses question/problem text as `id`.

Important caution:

- GSM8K examples use `question`.
- MATH examples use `problem`.
- Current sampling code expects `question`, so MATH support requires either
  normalization or a dataset adapter.

Role in the extension:

- Do not change this file at first.
- Add a future dataset normalization module instead of making this script handle
  every dataset format.

#### LightR_sampling.py

Core sampling script.

Current behavior:

- Loads expert and amateur models.
- Loads prompts from JSONL.
- Generates expert reasoning tokens.
- For each expert prefix, runs the amateur model.
- Computes expert-amateur KL divergence.
- Filters steps with fixed `beta`.
- Builds contrastive soft labels from `log P_expert - log P_amateur`.
- Saves JSONL samples containing:
  - `prompt_id`
  - `step`
  - `prefix`
  - `tokens`
  - `token_ids`
  - `weights`
  - `kl_divergence`

Important implementation details:

- `alpha` controls the plausibility mask.
- `beta` controls KL filtering.
- KL is computed over the truncated shared vocabulary length.
- The output only stores selected examples, not rejected candidates.

Main limitation for this research:

- Rejected candidates are discarded. A learned selector needs feature records for
  both selected and rejected candidates.
- Token selection currently has no access to token type, entropy gap, prefix
  position, top-1 mismatch, or step-type features.

Role in the extension:

- Preserve existing behavior.
- Introduce a new sampler script or a new optional mode rather than replacing
  this script immediately.
- The first production-quality path should be additive:
  `adaptive_lightr_sampling.py` or `src/lightr/sampling_adaptive.py`.

#### LightR_finetuning.py

Core fine-tuning script.

Current behavior:

- Loads contrastive soft-label JSONL.
- Reconstructs the chat prompt from `prompt_id` and `prefix`.
- Creates sparse soft-label vectors over the model vocabulary.
- Applies LoRA to `q_proj` and `v_proj`.
- Uses a custom `Trainer` that computes KL loss on the final token.

Important implementation details:

- `prompt_id` is being used as the question text.
- The loss is only applied at the final position of the provided prefix.
- The script expects the sampling model and fine-tuning model to match.

Role in the extension:

- No change required for the first adaptive selector if the adaptive sampler
  writes the same JSONL schema as current LightReasoner.
- Later, metadata fields can be added to the JSONL without breaking the trainer,
  as long as required fields stay the same.

#### merge.py

Merges a LoRA adapter into a base model and saves a standalone model.

Role in the extension:

- No change required.
- Keep model merging separate from adaptive sampling and selector training.

#### LRsamples/LR_samples.zip

Pre-collected LightReasoner samples for:

- Qwen2.5-Math-7B
- DeepSeek-R1-Distill-Qwen-1.5B
- Qwen2.5-Math-1.5B

Role in the extension:

- Useful for testing trainer compatibility.
- Not sufficient for training a selector because these samples only contain
  selected records, not all candidate records.

#### evaluation/README.md

Explains how to evaluate using the official Qwen2.5-Math evaluation stack.

Role in the extension:

- Use this as the canonical benchmark path.
- Do not build a separate evaluator unless the Qwen evaluation stack cannot
  support a new dataset.

#### evaluation/manual_eval/evaluation_dataset.py

Manual GSM8K evaluation prototype.

Important caution:

- This is not the official evaluation path.
- It contains hard-coded model and output behavior.
- It ends with `open("wrong_answers.txt", "x")`, which can fail if the file
  already exists.

Role in the extension:

- Treat as exploratory only.
- Do not use it for paper numbers.

#### analysis/preliminary_analysis.py

Logs token-level KL, entropy, and top-1 mismatch information.

Current output fields:

- `prompt_id`
- `step`
- `token`
- `kl_divergence`
- `expert_entropy`
- `amateur_entropy`
- `expert_top1_token`
- `amateur_top1_token`

Role in the extension:

- This is the closest existing script to the feature-logging path needed for the
  adaptive selector.
- Instead of modifying it heavily, create a richer candidate logger that borrows
  its metrics.

#### analysis/pre_analysis_stats.py

Computes descriptive statistics from KL logs.

Current behavior:

- Compares top-1 match vs mismatch KL.
- Counts high-confidence expert and uncertain-amateur cases.
- Prints KL histogram.

Role in the extension:

- Use as a starting point for selector analysis reports.
- Future analysis should add breakdowns by token category, prefix position, and
  reasoning-step type.

#### analysis/PPL_analysis.py

Evaluates perplexity across LoRA checkpoints.

Role in the extension:

- Optional diagnostic.
- Not a primary metric for reasoning accuracy.
- Useful for detecting catastrophic degradation.

#### analysis/case_study.py

Compares evaluation JSONL outputs from base and fine-tuned models.

Current behavior:

- Finds examples where base is wrong and fine-tuned model is correct.

Role in the extension:

- Useful for qualitative analysis.
- Extend later to compare fixed-KL LightReasoner against adaptive LightReasoner.

#### analysis/testspace/check_vocab_alignment.py

Checks whether expert and amateur tokenizers share token id mappings.

Role in the extension:

- Important safety check before computing by-index KL.
- Adaptive selector work should keep this check as a preflight validation.
- Cross-tokenizer support should be treated as a separate future project.

#### analysis/testspace/early_test.py

Prototype for checking Qwen tokenizer behavior and next-token probabilities.

Role in the extension:

- Exploratory only.
- Do not use directly in production experiments.

#### analysis/testspace/llama_KL.py

Prototype for KL computation between Llama-family models with shared-vocabulary
alignment.

Role in the extension:

- Useful reference for future cross-tokenizer research.
- Do not mix it into the first adaptive selector implementation.

#### assets/

Figures used in README.

Role in the extension:

- No changes during implementation.
- Add new figures only when final experiment plots are stable.

## Research Direction

### Proposed title

When Does Disagreement Teach? Adaptive Token Selection for Contrastive
Reasoning Distillation

### Core claim

Expert-amateur disagreement is not uniformly useful. Its value depends on:

- KL magnitude.
- Expert confidence.
- Amateur uncertainty.
- Whether the expert and amateur top predictions differ.
- Prefix position.
- Token category.
- Reasoning-step role.

A learned or adaptive selector can identify better contrastive supervision than
a fixed KL threshold.

### Baseline

The baseline is original LightReasoner:

- Fixed plausibility threshold `alpha`.
- Fixed KL threshold `beta`.
- Contrastive weights from `log P_expert - log P_amateur`.
- LoRA fine-tuning with KL loss.

### Proposed extension

Replace the fixed `beta` decision with a selector:

```text
candidate prefix -> feature extractor -> selector -> keep or reject
```

The selected examples must still be serialized into the current LightReasoner
training schema so the existing fine-tuning script remains compatible.

## Design Principles

1. Preserve current behavior.
   Existing scripts should continue to run without changes.

2. Add new code in isolated modules.
   Avoid turning `LightR_sampling.py` into a large experimental script.

3. Keep output schemas backward compatible.
   New metadata fields are allowed, but existing required fields must remain.

4. Separate candidate logging from sample selection.
   The project should support re-running selectors without re-running expensive
   model inference.

5. Separate selector training from expert fine-tuning.
   Selector artifacts should be versioned independently from LoRA checkpoints.

6. Prefer simple, inspectable selectors first.
   Start with rules and logistic regression before using neural selectors.

7. Make all experiments reproducible.
   Record model names, dataset path, alpha, selector config, seed, and git or
   snapshot information when available.

8. Keep future expansion possible.
   The architecture should support more datasets, more selectors, more feature
   groups, and eventually cross-tokenizer model pairs.

## Proposed Directory Layout

Add the following structure over time:

```text
configs/
  adaptive_sampling/
    qwen15_gsm8k_fixed_kl.yaml
    qwen15_gsm8k_rule_selector.yaml
    qwen15_gsm8k_learned_selector.yaml

docs/
  adaptive_step_aware_lightreasoner_plan.md

src/
  lightr/
    __init__.py
    data/
      __init__.py
      schemas.py
      jsonl.py
      adapters.py
    features/
      __init__.py
      token_features.py
      distribution_features.py
      step_type_features.py
      candidate_features.py
    selection/
      __init__.py
      base.py
      fixed_kl.py
      rule_selector.py
      learned_selector.py
    sampling/
      __init__.py
      candidate_logger.py
      adaptive_sampler.py
    training/
      __init__.py
      selector_training.py
    analysis/
      __init__.py
      selection_report.py
      bucket_ablation.py

scripts/
  log_candidates.py
  select_candidates.py
  train_selector.py
  build_lightr_dataset.py
  run_adaptive_sampling.py
  report_selection_stats.py

artifacts/
  candidates/
  selected_samples/
  selectors/
  reports/
```

Do not create all of this at once unless implementing the full system. The first
step should be a small subset that proves the pipeline.

## Data Schemas

### Current LightReasoner training sample

This schema must remain compatible with `LightR_finetuning.py`:

```json
{
  "prompt_id": "question text or id",
  "step": 17,
  "prefix": "expert generated prefix",
  "tokens": [" token text"],
  "token_ids": [123],
  "weights": [1.0],
  "kl_divergence": 0.82
}
```

### Proposed candidate record

A candidate record should be written for every prefix considered, including
prefixes that would be rejected by fixed KL.

```json
{
  "schema_version": "candidate_v1",
  "prompt_id": "stable prompt id",
  "question": "question text",
  "dataset": "gsm8k",
  "model_pair": {
    "expert": "Qwen/Qwen2.5-Math-1.5B",
    "amateur": "Qwen/Qwen2.5-0.5B"
  },
  "step": 17,
  "prefix": "expert generated prefix",
  "target_token_id": 123,
  "target_token_text": " 42",
  "sequence_length": 96,
  "normalized_position": 0.177,
  "distribution_features": {
    "kl_expert_amateur": 0.82,
    "expert_entropy": 0.71,
    "amateur_entropy": 2.94,
    "entropy_gap": 2.23,
    "expert_top1_prob": 0.61,
    "amateur_top1_prob": 0.24,
    "expert_top2_margin": 0.19,
    "top1_match": false
  },
  "token_features": {
    "token_category": "number",
    "is_number": true,
    "is_operator": false,
    "is_punctuation": false,
    "is_newline": false
  },
  "step_features": {
    "step_type": "calculation",
    "step_type_confidence": 0.75
  },
  "contrastive_target": {
    "token_ids": [123, 456],
    "tokens": [" 42", " forty"],
    "weights": [0.82, 0.18]
  },
  "fixed_kl_selected": true
}
```

### Proposed selected sample with metadata

This can be consumed by the current trainer because required fields remain:

```json
{
  "prompt_id": "question text or id",
  "step": 17,
  "prefix": "expert generated prefix",
  "tokens": [" 42", " forty"],
  "token_ids": [123, 456],
  "weights": [0.82, 0.18],
  "kl_divergence": 0.82,
  "selection_metadata": {
    "selector_name": "rule_v1",
    "selector_score": 0.91,
    "token_category": "number",
    "step_type": "calculation"
  }
}
```

## Feature Design

### Distribution features

Compute these from expert and amateur next-token distributions:

- `kl_expert_amateur`
- `expert_entropy`
- `amateur_entropy`
- `entropy_gap`
- `expert_top1_prob`
- `amateur_top1_prob`
- `expert_top2_margin`
- `top1_match`
- `expert_amateur_top1_prob_gap`
- `target_expert_prob`
- `target_amateur_prob`
- `target_prob_gap`

Rationale:

- KL alone measures distributional disagreement.
- Expert entropy tells whether the expert is confident.
- Amateur entropy tells whether the amateur is uncertain.
- Top-1 mismatch detects explicit prediction conflict.
- Margin detects whether the expert preference is sharp.

### Prefix position features

Compute:

- `step`
- `sequence_length`
- `normalized_position = step / sequence_length`
- `is_early`
- `is_middle`
- `is_late`

Suggested initial buckets:

- early: `normalized_position < 0.33`
- middle: `0.33 <= normalized_position < 0.66`
- late: `normalized_position >= 0.66`

Rationale:

- Early tokens often set up the reasoning path.
- Middle tokens often perform calculations and transformations.
- Late tokens may contain final-answer formatting or accumulated error.

### Token category features

Start with deterministic categories:

- `number`
- `operator`
- `equals`
- `variable`
- `punctuation`
- `newline`
- `word`
- `space`
- `currency_or_unit`
- `final_answer_marker`
- `other`

Implementation should use decoded token text and simple regex rules.

Example rules:

- Contains a digit: `number`.
- One of `+`, `-`, `*`, `/`, `=`, `<`, `>`: operator or equals.
- Contains `$`, `%`, `kg`, `cm`, `miles`, `dollars`: unit.
- Contains `####`, `boxed`, `answer`: final answer marker.
- Only punctuation: punctuation.

Rationale:

- High KL on punctuation may be less useful.
- Moderate KL on numbers and operators may be highly useful.
- Final-answer markers may need special handling.

### Step-type features

Start simple and deterministic. Do not make the first version depend on a large
LLM classifier.

Initial step types:

- `setup`
- `calculation`
- `transformation`
- `verification`
- `final_answer`
- `other`

Suggested heuristics:

- `setup`: prefix window contains `let`, `suppose`, `we need`, `given`,
  `define`, `total number`, `cost`, `rate`.
- `calculation`: local window contains digits plus arithmetic operators.
- `transformation`: local window contains `so`, `therefore`, `then`, `thus`,
  `which means`, `simplify`.
- `verification`: local window contains `check`, `verify`, `since`, `because`.
- `final_answer`: local window contains `answer is`, `final answer`, `####`,
  `boxed`.

Rationale:

- These labels are imperfect but interpretable.
- ACL-style analysis can report which step types carry the strongest training
  signal.

## Selector Designs

### Selector 0: Fixed KL baseline

Equivalent to current LightReasoner:

```text
select if kl_expert_amateur >= beta
```

Purpose:

- Reproduce current behavior.
- Ensure new pipeline is compatible with old results.

### Selector 1: Rule-based adaptive selector

Example:

```text
select if:
  expert_entropy <= 1.0
  and entropy_gap >= 1.0
  and token_category in {number, operator, equals, variable, final_answer_marker}
  and kl_expert_amateur >= beta_for_step_type
```

Step-type thresholds:

```text
setup: 0.30
calculation: 0.20
transformation: 0.35
verification: 0.40
final_answer: 0.25
other: 0.50
```

Purpose:

- Validate that step-aware selection can beat fixed KL without learned labels.
- Provide an interpretable baseline.

### Selector 2: Budgeted top-k selector

Select a fixed percentage of candidates per problem or per dataset:

```text
score = weighted feature combination
select top k percent by score
```

Purpose:

- Control token budget.
- Compare methods at equal selected-token counts.

### Selector 3: Learned lightweight selector

Use a logistic regression, shallow MLP, or gradient-boosted model.

Input:

- Distribution features.
- Prefix position features.
- Token category one-hot features.
- Step-type one-hot features.

Output:

- Probability that a candidate should be selected.

Purpose:

- Replace manual thresholding with learned selection.
- Enable feature ablations.

Important:

- The first learned selector should be simple enough to interpret.
- Do not start with a large neural selector unless simple selectors fail.

## Labeling Strategies for the Learned Selector

The hardest part is defining "useful disagreement." Below are feasible options.

### Option A: Outcome-based weak labels

Procedure:

1. Log all candidate records.
2. Bucket candidates by feature groups.
3. Build multiple selected datasets by bucket or selector variant.
4. Fine-tune small LoRA adapters.
5. Evaluate on a validation set.
6. Assign positive labels to candidate buckets that improve validation accuracy.

Pros:

- Strongest scientific signal.
- Directly tied to downstream model performance.

Cons:

- Expensive.
- Labels are noisy and group-level, not perfect token-level labels.

### Option B: Leave-one-bucket-out attribution

Procedure:

1. Start from fixed-KL selected samples.
2. Remove one bucket at a time, such as punctuation, late tokens, or
   high-entropy expert tokens.
3. Fine-tune and evaluate.
4. Infer which buckets are harmful or helpful.

Pros:

- Practical.
- Good for analysis and ablation.

Cons:

- Does not directly create clean per-token labels.

### Option C: Heuristic pseudo-labels

Procedure:

1. Define positive candidates as:
   - high or moderate KL,
   - low expert entropy,
   - high amateur entropy,
   - meaningful token category,
   - useful step type.
2. Define negative candidates as:
   - punctuation-only,
   - high expert entropy,
   - very late non-answer formatting,
   - low-confidence expert disagreement.
3. Train selector on these pseudo-labels.
4. Validate with downstream fine-tuning.

Pros:

- Fast.
- Easy to start.

Cons:

- Novelty is weaker unless downstream results are strong.

### Option D: Budgeted bandit selector

Procedure:

1. Treat token selection as a budgeted decision.
2. Reward selectors based on validation improvements after fine-tuning.
3. Update selector policy over multiple rounds.

Pros:

- Strong research angle.

Cons:

- Too ambitious for the first version.
- Expensive and harder to debug.

Recommended first path:

1. Implement fixed-KL reproduction.
2. Implement rule-based adaptive selector.
3. Implement bucket ablations.
4. Use bucket results to train a simple learned selector.

## Phase-by-Phase Development Plan

### Phase 0: Preserve and document baseline

Goal:

Create a stable understanding of the current project before modifying behavior.

Tasks:

1. Keep original scripts unchanged.
2. Record the current baseline command path:
   - `data_prep.py`
   - `LightR_sampling.py`
   - `LightR_finetuning.py`
   - `merge.py`
   - Qwen2.5-Math evaluation
3. Add this document under `docs/`.
4. Create an experiment notes file later, such as
   `docs/experiment_log_template.md`.

Exit criteria:

- Existing project is untouched except documentation.
- The adaptive project has a clear implementation plan.

### Phase 1: Add structured schemas and JSONL utilities

Goal:

Avoid ad hoc JSON manipulation across future scripts.

Status: completed for the first implementation.

Suggested files:

- `src/lightr/data/schemas.py`
- `src/lightr/data/jsonl.py`

Tasks:

1. Define dataclasses or typed dictionaries for:
   - candidate records,
   - selected samples,
   - model-pair metadata,
   - distribution features,
   - token features,
   - step features.
2. Implement `read_jsonl(path)`.
3. Implement `write_jsonl(path, records)`.
4. Implement schema validation for required fields.

Non-breaking rule:

- Do not modify `LightR_finetuning.py`.
- Selected sample output must stay compatible with existing trainer.

Exit criteria:

- Can read current LightReasoner sample JSONL.
- Can write selected sample JSONL in the same schema.

### Phase 2: Build feature extraction modules

Goal:

Compute reusable features for adaptive selection.

Status: completed for deterministic first-pass features.

Suggested files:

- `src/lightr/features/distribution_features.py`
- `src/lightr/features/token_features.py`
- `src/lightr/features/step_type_features.py`
- `src/lightr/features/candidate_features.py`

Tasks:

1. Implement entropy:
   ```text
   H(P) = -sum P log P
   ```
2. Implement top-1 and top-2 margin.
3. Implement top-1 match/mismatch.
4. Implement decoded-token category detection.
5. Implement prefix-position features.
6. Implement rule-based step-type detection from local prefix windows.

Testing:

- Unit test token category rules on examples:
  - `" 42"` -> number
  - `" +"` -> operator
  - `" ="` -> equals
  - `"\n"` -> newline
  - `" therefore"` -> word or transformation cue
  - `"####"` -> final_answer_marker

Exit criteria:

- Features can be computed without loading any model.
- Feature logic is deterministic and inspectable.

### Phase 3: Candidate logger

Goal:

Run expensive expert-amateur inference once and store all candidate records.

Status: implemented. Real model/GPU execution is pending.

Suggested script:

- `scripts/log_candidates.py`

Suggested module:

- `src/lightr/sampling/candidate_logger.py`

Tasks:

1. Load expert and amateur models.
2. Load prompts.
3. Generate expert trajectories.
4. For every prefix, compute:
   - expert distribution,
   - amateur distribution,
   - distribution features,
   - token features,
   - step features,
   - contrastive target under the plausibility mask.
5. Save every candidate, not only selected ones.
6. Preserve checkpointing so interrupted runs can resume.

Important changes from current `LightR_sampling.py`:

- Current script stores only selected records.
- Candidate logger must store selected and rejected candidates.
- Candidate logger should include enough metadata to run different selectors
  later without reloading models.

Non-breaking rule:

- Do not replace `LightR_sampling.py`.
- The candidate logger is a new entrypoint.

Exit criteria:

- Given 10 GSM8K examples, produces a candidate JSONL.
- The candidate JSONL includes all features needed for selection.
- Runtime and memory behavior are acceptable on a small test.

### Phase 4: Fixed-KL selector reproduction

Goal:

Prove the new candidate pipeline can reproduce current LightReasoner behavior.

Status: selector implemented and smoke-tested. Full reproduction against the
original sampler is pending a real candidate log.

Additional implementation completed:

- `scripts/compare_lightr_samples.py` compares original LightReasoner selected
  JSONL against regenerated selected JSONL by prompt, step, prefix overlap, KL
  summaries, prompt counts, and target-token counts.

Suggested files:

- `src/lightr/selection/base.py`
- `src/lightr/selection/fixed_kl.py`
- `scripts/select_candidates.py`

Tasks:

1. Load candidate JSONL.
2. Select candidates using:
   ```text
   kl_expert_amateur >= beta
   ```
3. Convert selected candidate records into current LightReasoner sample schema.
4. Compare selected-count statistics with original `LightR_sampling.py`.

Exit criteria:

- Selected records can be consumed by `LightR_finetuning.py`.
- Counts are close to the original sampler for the same prompts.
- Any differences are explained by implementation details.

### Phase 5: Rule-based adaptive selector

Goal:

Create the first step-aware selector without learned labels.

Status: implemented and smoke-tested. Budget controls are also implemented in
`scripts/select_candidates.py` via:

- `--max-samples`
- `--max-target-tokens`
- `--top-percent`

Suggested file:

- `src/lightr/selection/rule_selector.py`

Tasks:

1. Implement a configurable rule selector.
2. Support per-step-type thresholds.
3. Support token-category allowlists and denylists.
4. Support expert-confidence constraints.
5. Support selected-token budget constraints.

Example configuration:

```yaml
selector:
  name: rule_v1
  min_expert_confidence: 0.30
  max_expert_entropy: 1.5
  min_entropy_gap: 0.5
  allowed_token_categories:
    - number
    - operator
    - equals
    - variable
    - final_answer_marker
  step_type_thresholds:
    setup: 0.30
    calculation: 0.20
    transformation: 0.35
    verification: 0.40
    final_answer: 0.25
    other: 0.50
```

Exit criteria:

- Rule selector produces fewer or equal selected tokens than fixed KL at a
  comparable quality target.
- Selection statistics show interpretable shifts toward meaningful token types.

### Phase 6: Selection analysis reports

Goal:

Make selector behavior visible before expensive fine-tuning.

Suggested script:

- `scripts/report_selection_stats.py`

Suggested module:

- `src/lightr/analysis/selection_report.py`

Status: implemented and tested on real pre-collected selected samples.

Report sections:

1. Total candidates.
2. Total selected samples.
3. Selection rate.
4. Mean and median KL.
5. Selection by prefix position.
6. Selection by token category.
7. Selection by step type.
8. Expert entropy distribution.
9. Amateur entropy distribution.
10. Top-1 match vs mismatch.

Exit criteria:

- Every selected dataset has a matching report.
- Reports are saved under `artifacts/reports/`.

### Phase 7: Fine-tuning compatibility test

Goal:

Verify adaptive selected samples can train the expert without modifying
`LightR_finetuning.py`.

Status: schema-level compatibility checker implemented and tested on real
selected samples from the 50-question RunPod candidate run. A configurable LoRA
fine-tuning runner has also been implemented so adaptive datasets can be trained
without editing `LightR_finetuning.py`. Five-step RunPod smoke runs completed
for Fixed KL equal-budget and Rule balanced, with finite training loss and saved
LoRA adapters. The 100-step matched training runs also completed. Downstream
GSM8K-100 evaluation completed, but the 50-question pilot did not improve
accuracy: base 19%, equal-budget Fixed KL 18%, Rule balanced 16%.
Comparison analysis showed Rule balanced improves some base-wrong examples but
regresses more base-correct examples, with visible prompt-copying/artifact
generation in some outputs. A reasoning-focused selector config has been added
for the next diagnostic. On the existing 50-question candidate log,
`rule_v3_reasoning_focus` selected 83 samples and 112 target tokens, with no
generic word/punctuation/space/newline tokens; it is compatible with the
trainer. A conservative 20-step run completed, but the first GSM8K-30 fast
evaluation with 128 generated tokens was poor: 2/30. A fair same-setting
comparison showed the fast 128-token diagnostic is too harsh overall: base
1/30, Fixed KL 2/30, Rule balanced 1/30, reasoning-focus 2/30. Future quick
checks should use batched 30-example evaluation with 256 generated tokens.
Under that setting, base scored 3/30 and reasoning-focus scored 4/30, a weak
positive signal. The completed same-setting comparison was: base 3/30,
equal-budget Fixed KL 5/30, Rule balanced 4/30, reasoning-focus 4/30. This
means reasoning-focus does not beat Fixed KL in the 50-question pilot; the next
meaningful step is a larger 500-question candidate log with equal-budget
selector comparisons.

Current pivot:

- Add correctness-gated candidate logging.
- Add degeneration filtering before candidate records are written.
- Only train on expert trajectories that are correct and not obviously
  degenerate.
- This addresses the main failure mode from the 50-question pilot: selected
  tokens can be reasoning-critical but still come from bad expert trajectories.

Implemented files:

- `scripts/check_finetuning_compatibility.py`
- `scripts/log_candidates.py`
- `scripts/run_contrastive_finetune.py`
- `src/lightr/training/contrastive_finetuning.py`
- `scripts/evaluate_gsm8k_adapter.py`
- `scripts/compare_gsm8k_eval_outputs.py`
- `src/lightr/evaluation/gsm8k_eval.py`
- `src/lightr/sampling/candidate_logger.py`
- `configs/adaptive_sampling/qwen15_gsm8k_rule_selector_reasoning_focus.json`

Tasks:

1. Run a tiny selected dataset through the existing trainer.
2. Confirm the trainer loads records successfully.
3. Confirm loss decreases on a small run.
4. Confirm no metadata fields break the data loader.

Exit criteria:

- Existing trainer runs unchanged.
- Adaptive samples are backward compatible.

### Phase 8: First controlled experiments

Goal:

Compare fixed KL and adaptive rule selection under controlled conditions.

Minimum experiment matrix:

```text
Expert: Qwen2.5-Math-1.5B
Amateur: Qwen2.5-0.5B
Training data: GSM8K train subset
Evaluation: GSM8K test, MATH, SVAMP, ASDiv

Methods:
1. Base expert
2. Original LightReasoner fixed KL
3. New pipeline fixed KL
4. Rule selector v1
5. Rule selector v1 with equal token budget
```

Metrics:

- Accuracy.
- Average gain.
- Number of selected tokens.
- Number of source problems.
- Training time.
- Sampling time.
- Selection rate by token category and step type.

Exit criteria:

- New fixed-KL pipeline matches original fixed-KL behavior closely.
- Rule selector shows either better accuracy at the same token budget or similar
  accuracy with fewer selected tokens.

### Phase 9: Bucket ablations

Goal:

Estimate which disagreement types are helpful or harmful.

Suggested module:

- `src/lightr/analysis/bucket_ablation.py`

Status: infrastructure implemented. Full ablation experiments are pending real
candidate logs and fine-tuning runs.

Implemented files:

- `src/lightr/analysis/bucket_ablation.py`
- `scripts/build_bucket_ablation.py`

Candidate buckets:

- token category:
  - number
  - operator
  - punctuation
  - word
  - final answer marker
- prefix position:
  - early
  - middle
  - late
- step type:
  - setup
  - calculation
  - transformation
  - final_answer
- confidence:
  - low expert entropy
  - high expert entropy
  - high entropy gap
  - top-1 mismatch

Experiment examples:

- Fixed KL minus punctuation.
- Fixed KL minus late non-answer tokens.
- Fixed KL plus moderate-KL arithmetic tokens.
- Rule selector without step-type features.
- Rule selector without token-category features.

Exit criteria:

- Produce evidence for the paper claim that KL magnitude alone is incomplete.
- Identify feature groups that matter most.

### Phase 10: Learned selector

Goal:

Train a lightweight selector from weak labels or bucket attribution.

Suggested files:

- `src/lightr/selection/learned_selector.py`
- `src/lightr/training/selector_training.py`
- `scripts/train_selector.py`

Status: first lightweight implementation completed and unit-tested. It trains a
dependency-free linear logistic selector from candidate JSONL using weak labels
from the rule selector, fixed-KL selector, or the candidate fixed-KL flag. Real
training/evaluation is pending candidate logs.

Initial model:

- Logistic regression or a small MLP.

Input features:

- Numeric distribution features.
- One-hot token category.
- One-hot step type.
- Prefix position features.

Labels:

- Start with heuristic pseudo-labels.
- Later replace or augment with outcome-based bucket labels.

Output:

- Selector artifact under `artifacts/selectors/`.
- Feature normalization metadata.
- Selector configuration.

Exit criteria:

- Learned selector can score candidate JSONL.
- Learned selector can generate selected sample JSONL.
- Learned selector is evaluated against fixed KL and rule selector.

### Phase 11: Generalization tests

Goal:

Show the selector does not overfit GSM8K.

Experiments:

1. Train selector policy on GSM8K.
2. Apply selector without retuning to MATH or other datasets.
3. Evaluate transfer on:
   - MATH
   - SVAMP
   - ASDiv
   - Minerva Math if available
   - MMLU STEM if evaluation setup supports it

Strong evidence:

- Learned or rule selector beats fixed KL on average.
- Or matches fixed KL while selecting fewer tokens.
- Or improves out-of-domain transfer even if in-domain gains are modest.

Exit criteria:

- Results support a robust ACL-style claim.

### Phase 12: Paper-ready analysis

Goal:

Turn implementation into a defensible research contribution.

Required analysis:

1. KL-only selection vs adaptive selection.
2. Feature ablation:
   - remove entropy features,
   - remove token-category features,
   - remove step-type features,
   - remove prefix-position features.
3. Token budget curve:
   - 5k,
   - 10k,
   - 20k,
   - 40k selected samples.
4. Step-type distribution plots.
5. Case studies:
   - fixed KL fails but adaptive succeeds,
   - adaptive rejects high-KL low-value punctuation,
   - adaptive keeps moderate-KL arithmetic token.
6. Robustness:
   - different `beta`,
   - different source problem counts,
   - different random seeds where feasible.

Exit criteria:

- The method has a clear story, not just a leaderboard result.

## Implementation Order for the First Development Sprint

Do this first:

1. Create `src/lightr/features/token_features.py`. Completed.
2. Create `src/lightr/features/step_type_features.py`. Completed.
3. Create `src/lightr/features/distribution_features.py`. Completed.
4. Create `scripts/log_candidates.py` for a very small subset. Implemented;
   real model run pending local model paths/GPU.
5. Create `scripts/select_candidates.py` with fixed-KL selection. Completed.
6. Confirm selected output works with `LightR_finetuning.py`. Schema
   compatibility checker completed; configurable LoRA smoke runner completed;
   real loss-decrease result pending RunPod execution.
7. Add rule selector. Completed.
8. Add selection report. Completed.

Do not do this first:

- Do not rewrite `LightR_sampling.py`.
- Do not add a neural selector before candidate logging is stable.
- Do not change evaluation infrastructure before baseline reproduction.
- Do not add cross-tokenizer support until same-tokenizer adaptive selection is
  working.

## Testing Strategy

### Unit tests

Add tests later under:

```text
tests/
  test_token_features.py
  test_step_type_features.py
  test_distribution_features.py
  test_sample_schema.py
  test_selectors.py
```

Minimum tests:

1. Token category classification.
2. Step-type heuristic classification.
3. Entropy and margin calculations.
4. Fixed-KL selector behavior.
5. Backward-compatible selected sample schema.

### Smoke tests

Use tiny data before full experiments:

```text
max_questions = 2
max_new_tokens = 32
```

Check:

- Candidate JSONL is created.
- Selected sample JSONL is created.
- Existing fine-tuning dataset loader can read selected records.
- Selection report is created.

### Integration tests

Use small but meaningful runs:

```text
max_questions = 50
max_new_tokens = 128
```

Check:

- Fixed-KL new pipeline selected-count roughly matches old pipeline.
- Rule selector produces expected distribution changes.
- LoRA training starts and logs loss.

### Experiment tests

Use full or paper-scale runs only after smoke and integration tests pass.

## Reproducibility Checklist

For every run, save:

- Date.
- Machine/GPU.
- Python version.
- PyTorch version.
- Transformers version.
- Expert model name and revision if available.
- Amateur model name and revision if available.
- Dataset name and split.
- Number of source problems.
- Max new tokens.
- Alpha.
- Selector name and config.
- Random seed.
- Number of candidates.
- Number of selected samples.
- Output paths.
- Evaluation command.

## Suggested Experiment Naming

Use stable names:

```text
qwen15_gsm8k_fixedkl_beta04_seed0
qwen15_gsm8k_rulev1_budget20k_seed0
qwen15_gsm8k_learnedv1_budget20k_seed0
qwen7_gsm8k_rulev1_budget20k_seed0
ds15_gsm8k_rulev1_budget20k_seed0
```

## Risks and Mitigations

### Risk: Selector overfits GSM8K

Mitigation:

- Train selector on GSM8K.
- Test without retuning on MATH, SVAMP, and ASDiv.
- Report out-of-domain transfer.

### Risk: Step-type heuristics are noisy

Mitigation:

- Treat step type as one feature group, not the sole decision signal.
- Include ablations with and without step-type features.
- Keep heuristic rules transparent.

### Risk: Learned selector labels are weak

Mitigation:

- Start with rule selector as a strong interpretable baseline.
- Use bucket ablations to justify labels.
- Report label construction honestly.

### Risk: Adaptive selector selects too few tokens

Mitigation:

- Add budgeted selection mode.
- Compare at equal token budgets.

### Risk: Adaptive selector selects too many easy tokens

Mitigation:

- Add expert entropy and margin constraints.
- Track token category and step-type distribution.

### Risk: Candidate logging consumes too much disk

Mitigation:

- Store only top-k distributions and contrastive target candidates, not full
  vocab distributions.
- Compress JSONL when needed.
- Make full distribution dumps optional.

### Risk: Different tokenizers invalidate by-index KL

Mitigation:

- Use `analysis/testspace/check_vocab_alignment.py` before same-vocab runs.
- Keep cross-tokenizer alignment as future work.

## Future Extensions

### Multi-amateur selection

Use multiple amateurs to expose different expert strengths:

- weak general amateur,
- weak math amateur,
- instruction amateur,
- multilingual amateur.

Candidate features can include disagreement with each amateur and agreement
patterns across amateurs.

### Cross-tokenizer LightReasoner

Generalize expert-amateur comparison across model families by aligning tokens
through text spans, byte-level units, or candidate string rescoring.

This should be a separate research phase because it changes the distribution
comparison assumptions.

### LLM-assisted step-type labeling

Use an LLM to label reasoning step types after the heuristic system is stable.

Important:

- Keep deterministic heuristics as the default.
- Use LLM labels for analysis or comparison first, not as a hard dependency.

### Active selection

Treat candidate selection as a budgeted learning problem where the selector
chooses examples expected to maximize downstream validation improvement.

This is promising but should come after fixed, rule, and learned selectors.

## Related Work Anchors

Use these papers to frame the research:

1. LightReasoner: Can Small Language Models Teach Large Language Models
   Reasoning?
   - Main project paper.
   - URL: https://huggingface.co/papers/2510.07962

2. Contrastive Decoding: Open-ended Text Generation as Optimization.
   - ACL 2023.
   - Uses expert-amateur contrast at inference time.
   - URL: https://aclanthology.org/2023.acl-long.687/

3. The First Few Tokens Are All You Need: An Efficient and Effective
   Unsupervised Prefix Fine-Tuning Method for Reasoning Models.
   - NeurIPS 2025 poster.
   - Relevant for prefix position and token efficiency.
   - URL: https://openreview.net/forum?id=1SCMFCGliM

## Recommended First Milestone

The first milestone should be:

> Produce a candidate JSONL for 50 GSM8K examples, run fixed-KL selection from
> that candidate file, and verify that the selected JSONL trains with the
> existing `LightR_finetuning.py`.

This milestone proves the new architecture without changing the original
project behavior.

The second milestone should be:

> Add a rule-based step-aware selector and show a selection report comparing it
> with fixed KL.

The third milestone should be:

> Fine-tune fixed-KL and rule-selected datasets under the same token budget and
> compare GSM8K validation performance.

Only after these milestones should the learned selector be implemented.
