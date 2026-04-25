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
from lightr.features.step_type_features import classify_step_type
from lightr.features.token_features import classify_token
from lightr.selection.fixed_kl import FixedKLSelector
from lightr.selection.rule_selector import RuleSelector


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


if __name__ == "__main__":
    unittest.main()
