"""Validate the frozen RLVR data-design configuration."""

from __future__ import annotations

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

import yaml


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


if __name__ == "__main__":
    unittest.main()
