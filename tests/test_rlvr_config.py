"""Validate the frozen RLVR data-design configuration."""

from __future__ import annotations

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

import yaml

from src import constraint_sampler


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "configs" / "rlvr.yaml"
OFFICIAL_IFEVAL = PROJECT_DIR / "third_party" / "Multi-IF" / "ifeval.py"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RlvrConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_ratios_and_sizes_are_frozen(self) -> None:
        data = self.config["data"]
        self.assertEqual(data["train_size"], 2000)
        self.assertEqual(data["validation_size"], 100)
        self.assertAlmostEqual(sum(data["task_bucket_ratios"].values()), 1.0)
        self.assertAlmostEqual(sum(data["context_turn_ratios"].values()), 1.0)
        self.assertAlmostEqual(
            sum(
                self.config["constraint_sampling"][
                    "constraints_per_prompt_ratios"
                ].values()
            ),
            1.0,
        )

    def test_evaluation_hashes_match_frozen_files(self) -> None:
        boundary = self.config["evaluation_boundary"]
        for name in ("source", "dev", "test"):
            path = PROJECT_DIR / boundary[name]
            self.assertEqual(sha256_file(path), boundary[f"{name}_sha256"])

    def test_constraint_ids_exist_in_official_checker(self) -> None:
        source = OFFICIAL_IFEVAL.read_text(encoding="utf-8")
        configured = set(self.config["constraint_sampling"]["constraints"])
        prefix_constants = {
            "keywords": "_KEYWORD",
            "length_constraints": "_LENGTH",
            "detectable_content": "_CONTENT",
            "detectable_format": "_FORMAT",
            "combination": "_COMBINATION",
            "startend": "_STARTEND",
            "punctuation": "_PUNCTUATION",
        }
        for instruction_id in configured:
            prefix, name = instruction_id.split(":", maxsplit=1)
            constant = prefix_constants[prefix]
            self.assertIn(f'{constant} + "{name}"', source)

        referenced = set()
        sampling = self.config["constraint_sampling"]
        for group in sampling["mutually_exclusive_groups"]:
            referenced.update(group)
        for pair in sampling["forbidden_pairs"]:
            referenced.update(pair)
        self.assertLessEqual(referenced, configured)

    def test_quality_pool_can_fill_every_bucket_quota(self) -> None:
        data = self.config["data"]
        total = data["train_size"] + data["validation_size"]
        counts: Counter[str] = Counter()
        source_path = PROJECT_DIR / data["source"]
        with source_path.open("r", encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                metadata = row["metadata"]
                if metadata["quality_score"] >= data["minimum_quality_score"]:
                    counts[metadata["task_bucket"]] += 1

        for bucket, ratio in data["task_bucket_ratios"].items():
            required = round(total * ratio)
            self.assertGreaterEqual(
                counts[bucket], required, f"{bucket}: {counts[bucket]} < {required}"
            )

    def test_t1_and_learning_progress_sampler_are_frozen(self) -> None:
        from datasets import load_from_disk

        t1 = self.config["t1"]
        self.assertEqual(self.config["model"]["primary"], "Qwen/Qwen3-4B-Base")
        self.assertEqual(self.config["model"]["teacher"], "Qwen/Qwen3-4B")
        self.assertEqual(self.config["rlvr"]["initialization"], "existing_adapter")
        self.assertEqual(
            self.config["rlvr"]["initial_adapter"],
            "outputs/distill/T1_v2/final_adapter",
        )
        self.assertTrue(self.config["rlvr"]["blocked_until_t1_v2_generation_gate"])
        self.assertTrue(self.config["rlvr"]["rollout_enable_thinking"])
        self.assertEqual(self.config["rlvr"]["maximum_completion_tokens"], 1024)
        self.assertEqual(self.config["rlvr"]["zero_variance_abort_patience"], 8)
        self.assertEqual(
            self.config["rlvr"]["zero_variance_policy"],
            "accept_all_batches_abort_on_consecutive_zero_signal",
        )
        self.assertEqual(t1["input"], "data/distill/t1_thinking_accepted.jsonl")
        self.assertEqual(t1["accepted_rows"], 1461)
        self.assertEqual(t1["reasoning_compression"], "disabled")
        self.assertEqual(t1["legacy_generation_gate"], "failed_free_generation_structure")
        self.assertEqual(t1["generation_gate"]["status"], "pending")
        self.assertEqual(
            t1["qlora"]["structure_token_strings"],
            ["<think>", "</think>", "<|im_end|>"],
        )
        self.assertNotIn("lm_head", t1["qlora"]["target_modules"])
        self.assertNotIn("lm_head", t1["qlora"]["legacy_target_modules"])
        self.assertEqual(t1["qlora"]["structure_token_loss_multiplier"], 4.0)
        self.assertEqual(t1["training"]["output_dir"], "outputs/distill/T1_v2")
        self.assertEqual(
            t1["rlvr_usage"],
            "blocked_until_t1_v2_generation_gate_passes",
        )
        with (PROJECT_DIR / t1["input"]).open("r", encoding="utf-8") as handle:
            ids = [json.loads(line)["id"] for line in handle if line.strip()]
        self.assertEqual(len(ids), t1["accepted_rows"])
        self.assertEqual(len(ids), len(set(ids)))
        cached = load_from_disk(str(PROJECT_DIR / t1["cache"]))
        cached_train_ids = {row["id"] for row in cached["t1_train"]}
        self.assertLessEqual(
            set(self.config["rlvr"]["smoke_prompt_ids"]), cached_train_ids
        )

        sampling = self.config["constraint_sampling"]
        self.assertEqual(sampling["ema_beta"], constraint_sampler.FROZEN_EMA_BETA)
        self.assertEqual(
            sampling["uniform_mixture_lambda"],
            constraint_sampler.FROZEN_MIXTURE_LAMBDA,
        )
        self.assertEqual(
            sampling["min_sampling_weight"], constraint_sampler.FROZEN_MIN_WEIGHT
        )
        self.assertEqual(
            sampling["max_sampling_weight"], constraint_sampler.FROZEN_MAX_WEIGHT
        )
        self.assertEqual(
            sampling["sampling_weight_update_steps"],
            constraint_sampler.FROZEN_UPDATE_STEPS,
        )
        self.assertEqual(
            sampling["fast_ema_beta"], constraint_sampler.FROZEN_FAST_EMA_BETA
        )
        self.assertEqual(
            sampling["stagnation_tolerance"],
            constraint_sampler.FROZEN_STAGNATION_TOLERANCE,
        )
        self.assertEqual(
            sampling["stagnation_patience"],
            constraint_sampler.FROZEN_STAGNATION_PATIENCE,
        )
        self.assertEqual(
            sampling["stagnation_scale"], constraint_sampler.FROZEN_STAGNATION_SCALE
        )


if __name__ == "__main__":
    unittest.main()
