from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lightr.data.schemas import validate_selected_sample
from lightr.features.distribution_features import compute_distribution_features
from lightr.evaluation.gsm8k_eval import GSM8KEvalConfig, extract_gsm8k_gold, extract_predicted_answer, normalize_answer
from lightr.features.step_type_features import classify_step_type
from lightr.features.token_features import classify_token
from lightr.selection.fixed_kl import FixedKLSelector
from lightr.selection.rule_selector import RuleSelector
from lightr.sampling.candidate_logger import detect_degenerate_generation
from lightr.training.contrastive_finetuning import (
    ContrastiveFinetuneConfig,
    ContrastiveSoftLabelDataset,
    build_collate_fn,
    parse_torch_dtype,
)


def candidate_record(**overrides):
    record = {
        "schema_version": "candidate_v1",
        "prompt_id": "q1",
        "question": "If x = 2 + 3, what is x?",
        "dataset": "fixture",
        "model_pair": {"expert": "expert", "amateur": "amateur"},
        "step": 4,
        "prefix": "Let x = 2 +",
        "target_token_id": 10,
        "target_token_text": " 3",
        "sequence_length": 12,
        "normalized_position": 0.33,
        "distribution_features": {
            "kl_expert_amateur": 0.45,
            "expert_entropy": 0.7,
            "amateur_entropy": 1.8,
            "entropy_gap": 1.1,
            "expert_top1_prob": 0.62,
            "amateur_top1_prob": 0.25,
            "expert_top2_margin": 0.2,
            "top1_match": False,
        },
        "token_features": {
            "token_category": "number",
            "is_number": True,
            "is_operator": False,
            "is_punctuation": False,
            "is_newline": False,
        },
        "step_features": {
            "step_type": "calculation",
            "step_type_confidence": 0.75,
        },
        "contrastive_target": {
            "token_ids": [10],
            "tokens": [" 3"],
            "weights": [1.0],
        },
        "fixed_kl_selected": True,
    }
    record.update(overrides)
    return record


class AdaptiveFeatureSelectionTest(unittest.TestCase):
    def test_token_categories(self):
        self.assertEqual(classify_token(" 42"), "number")
        self.assertEqual(classify_token(" +"), "operator")
        self.assertEqual(classify_token(" ="), "equals")
        self.assertEqual(classify_token("\n"), "newline")
        self.assertEqual(classify_token(" x"), "variable")
        self.assertEqual(classify_token("####"), "final_answer_marker")

    def test_step_types(self):
        self.assertEqual(classify_step_type("Let x be the total cost", "")[0], "setup")
        self.assertEqual(classify_step_type("Now 2 +", " 3")[0], "calculation")
        self.assertEqual(classify_step_type("Therefore the answer is", " 5")[0], "final_answer")

    def test_distribution_features(self):
        features = compute_distribution_features([0.7, 0.2, 0.1], [0.2, 0.7, 0.1], target_token_id=0)
        self.assertGreater(features["kl_expert_amateur"], 0)
        self.assertFalse(features["top1_match"])
        self.assertGreater(features["target_prob_gap"], 0)

    def test_selectors(self):
        candidate = candidate_record()
        self.assertTrue(FixedKLSelector(beta=0.4).decide(candidate).selected)
        self.assertFalse(FixedKLSelector(beta=0.5).decide(candidate).selected)
        self.assertTrue(RuleSelector().decide(candidate).selected)

        punctuation = candidate_record(
            token_features={"token_category": "punctuation"},
            target_token_text=",",
        )
        self.assertFalse(RuleSelector().decide(punctuation).selected)

    def test_select_candidates_cli_outputs_trainer_compatible_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "candidates.jsonl"
            output_path = Path(tmpdir) / "selected.jsonl"
            input_path.write_text(json.dumps(candidate_record()) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "select_candidates.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--selector",
                    "fixed_kl",
                    "--beta",
                    "0.4",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn('"samples_selected": 1', result.stdout)
            selected = json.loads(output_path.read_text(encoding="utf-8").strip())
            validate_selected_sample(selected)
            self.assertEqual(selected["prompt_id"], "If x = 2 + 3, what is x?")
            self.assertEqual(selected["token_ids"], [10])

    def test_select_candidates_cli_respects_max_samples_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "candidates.jsonl"
            output_path = Path(tmpdir) / "selected.jsonl"
            records = [
                candidate_record(prompt_id="q1", distribution_features={**candidate_record()["distribution_features"], "kl_expert_amateur": 0.45}),
                candidate_record(prompt_id="q2", distribution_features={**candidate_record()["distribution_features"], "kl_expert_amateur": 0.95}),
            ]
            input_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "select_candidates.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--selector",
                    "fixed_kl",
                    "--beta",
                    "0.4",
                    "--max-samples",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn('"samples_selected_before_budget": 2', result.stdout)
            self.assertIn('"samples_selected": 1', result.stdout)
            selected = json.loads(output_path.read_text(encoding="utf-8").strip())
            self.assertEqual(selected["selection_metadata"]["original_prompt_id"], "q2")

    def test_select_candidates_cli_filters_out_of_vocab_target_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "candidates.jsonl"
            output_path = Path(tmpdir) / "selected.jsonl"
            record = candidate_record(
                contrastive_target={
                    "token_ids": [10, 151643],
                    "tokens": [" 3", "<|endoftext|>"],
                    "weights": [0.25, 0.75],
                }
            )
            input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "select_candidates.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--selector",
                    "fixed_kl",
                    "--beta",
                    "0.4",
                    "--max-token-id-exclusive",
                    "151643",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            selected = json.loads(output_path.read_text(encoding="utf-8").strip())
            self.assertEqual(selected["token_ids"], [10])
            self.assertEqual(selected["weights"], [1.0])
            self.assertEqual(selected["selection_metadata"]["dropped_out_of_vocab_target_tokens"], 1)

    def test_train_and_use_learned_selector_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "candidates.jsonl"
            selector_path = Path(tmpdir) / "selector.json"
            output_path = Path(tmpdir) / "selected.jsonl"
            records = [
                candidate_record(prompt_id="q1"),
                candidate_record(
                    prompt_id="q2",
                    token_features={"token_category": "punctuation", "is_punctuation": True},
                    distribution_features={**candidate_record()["distribution_features"], "kl_expert_amateur": 0.41},
                ),
            ]
            input_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "train_selector.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(selector_path),
                    "--label-source",
                    "rule",
                    "--epochs",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(selector_path.exists())

            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "select_candidates.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--selector",
                    "learned",
                    "--config",
                    str(selector_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(output_path.exists())

    def test_selector_milestone_table_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "candidates.jsonl"
            output_md = Path(tmpdir) / "table.md"
            selected_dir = Path(tmpdir) / "selected"
            records = [
                candidate_record(prompt_id="q1"),
                candidate_record(
                    prompt_id="q2",
                    token_features={"token_category": "word", "is_number": False},
                    distribution_features={**candidate_record()["distribution_features"], "kl_expert_amateur": 0.85},
                    step_features={"step_type": "setup", "step_type_confidence": 0.7},
                ),
            ]
            input_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "selector_milestone_table.py"),
                    "--input",
                    str(input_path),
                    "--output-md",
                    str(output_md),
                    "--selected-output-dir",
                    str(selected_dir),
                    "--max-token-id-exclusive",
                    "151643",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("| Fixed KL |", result.stdout)
            self.assertIn("| Rule strict |", result.stdout)
            self.assertIn("| Rule balanced |", result.stdout)
            self.assertTrue(output_md.exists())
            self.assertTrue((selected_dir / "fixed_kl.jsonl").exists())
            self.assertTrue((selected_dir / "rule_strict.jsonl").exists())
            self.assertTrue((selected_dir / "rule_balanced.jsonl").exists())

    def test_contrastive_finetuning_dataset_and_collator(self):
        class FakeTokenizer:
            vocab_size = 20
            pad_token_id = 0
            eos_token_id = 1
            chat_template = None

            def __call__(self, text, truncation, max_length, return_tensors):
                import torch

                token_count = min(max_length, max(1, len(text.split())))
                return {
                    "input_ids": torch.arange(2, 2 + token_count).unsqueeze(0),
                    "attention_mask": torch.ones(1, token_count, dtype=torch.long),
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            selected_path = Path(tmpdir) / "selected.jsonl"
            record = {
                "prompt_id": "If x = 2 + 3, what is x?",
                "step": 4,
                "prefix": "Let x = 2 +",
                "tokens": [" 3"],
                "token_ids": [10],
                "weights": [1.0],
                "kl_divergence": 0.45,
            }
            selected_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            dataset = ContrastiveSoftLabelDataset(selected_path, FakeTokenizer(), model_vocab_size=20, max_length=8)
            item = dataset[0]
            self.assertEqual(len(dataset), 1)
            self.assertEqual(item["labels"].shape[0], 20)
            self.assertEqual(float(item["labels"][10]), 1.0)

            batch = build_collate_fn(FakeTokenizer())([item, item])
            self.assertEqual(batch["input_ids"].shape[0], 2)
            self.assertEqual(batch["labels"].shape, (2, 20))

    def test_contrastive_finetune_config_and_dtype_parser(self):
        self.assertEqual(str(parse_torch_dtype("bfloat16")), "torch.bfloat16")
        self.assertEqual(str(parse_torch_dtype("fp16")), "torch.float16")
        config = ContrastiveFinetuneConfig(
            model_path="./model",
            train_jsonl="./selected.jsonl",
            output_dir="./out",
            max_steps=2,
            lora_target_modules=("q_proj", "v_proj"),
        )
        self.assertEqual(config.max_steps, 2)
        self.assertEqual(config.lora_target_modules, ("q_proj", "v_proj"))

    def test_gsm8k_answer_extraction(self):
        self.assertEqual(normalize_answer("$1,234."), "1234")
        self.assertEqual(extract_gsm8k_gold("We compute it.\n#### 42"), "42")
        self.assertEqual(extract_predicted_answer("Therefore \\boxed{42}."), "42")
        self.assertEqual(extract_predicted_answer("The final answer is: $1,234."), "1234")
        self.assertEqual(extract_predicted_answer("2 + 3 = 5, so answer is 5"), "5")
        config = GSM8KEvalConfig(
            model_path="./model",
            input_jsonl="./gsm8k_test.jsonl",
            output_jsonl="./out.jsonl",
            batch_size=4,
            max_new_tokens=128,
        )
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.max_new_tokens, 128)

    def test_detect_degenerate_generation(self):
        clean = "Let x be the total. Then 2 + 3 = 5. The final answer is \\boxed{5}."
        is_degenerate, reasons = detect_degenerate_generation(
            clean,
            question="If x = 2 + 3, what is x?",
            system_prompt="Please reason step by step.",
        )
        self.assertFalse(is_degenerate)
        self.assertEqual(reasons, [])

        copied_prompt = "Please reason step by step. Please reason step by step. ager ager ager"
        is_degenerate, reasons = detect_degenerate_generation(
            copied_prompt,
            question="If x = 2 + 3, what is x?",
            system_prompt="Please reason step by step.",
        )
        self.assertTrue(is_degenerate)
        self.assertIn("copies_system_prompt", reasons)
        self.assertIn("repeated_artifact:ager", reasons)

    def test_compare_gsm8k_eval_outputs_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "base.jsonl"
            fixed_path = Path(tmpdir) / "fixed.jsonl"
            rule_path = Path(tmpdir) / "rule.jsonl"
            output_path = Path(tmpdir) / "comparison.json"

            def write(path, rows):
                path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            write(
                base_path,
                [
                    {"id": "a", "index": 0, "gold": "1", "prediction": "1", "correct": True, "response": "1"},
                    {"id": "b", "index": 1, "gold": "2", "prediction": "0", "correct": False, "response": "0"},
                ],
            )
            write(
                fixed_path,
                [
                    {"id": "a", "index": 0, "gold": "1", "prediction": "0", "correct": False, "response": "0"},
                    {"id": "b", "index": 1, "gold": "2", "prediction": "2", "correct": True, "response": "2"},
                ],
            )
            write(
                rule_path,
                [
                    {"id": "a", "index": 0, "gold": "1", "prediction": "1", "correct": True, "response": "1"},
                    {"id": "b", "index": 1, "gold": "2", "prediction": "2", "correct": True, "response": "2"},
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "compare_gsm8k_eval_outputs.py"),
                    "--base",
                    str(base_path),
                    "--fixed",
                    str(fixed_path),
                    "--rule",
                    str(rule_path),
                    "--output",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn('"rule_balanced": 1.0', result.stdout)
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(report["overlap_counts"]["fixed_improves_over_base"], 1)
            self.assertEqual(report["overlap_counts"]["fixed_regresses_from_base"], 1)


if __name__ == "__main__":
    unittest.main()
