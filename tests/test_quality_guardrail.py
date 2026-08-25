"""Validate the frozen Chinese quality guardrail and its manifest."""

from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts import build_chinese_quality_guardrail as builder  # noqa: E402


class ChineseQualityGuardrailTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = builder.read_jsonl(builder.OUTPUT_PATH)
        cls.manifest = json.loads(builder.MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_size_schema_and_quotas_are_frozen(self) -> None:
        self.assertEqual(len(self.rows), 50)
        self.assertEqual(
            Counter(row["category"] for row in self.rows),
            Counter(builder.QUOTAS),
        )
        self.assertEqual(len({row["id"] for row in self.rows}), 50)
        self.assertEqual(len({row["metadata"]["source_id"] for row in self.rows}), 50)
        for row in self.rows:
            self.assertEqual(len(row["messages"]), 1)
            self.assertEqual(row["messages"][0]["role"], "user")
            self.assertNotIn("assistant", row)

    def test_no_rlvr_source_or_multi_if_prompt_overlap(self) -> None:
        rlvr_ids = builder.load_rlvr_source_ids(builder.RLVR_PATHS)
        selected_ids = {row["metadata"]["source_id"] for row in self.rows}
        self.assertTrue(selected_ids.isdisjoint(rlvr_ids))

        multi_if_keys = builder.load_multi_if_prompt_keys(builder.MULTI_IF_PATHS)
        selected_keys = {
            builder.normalize_prompt(row["messages"][0]["content"])
            for row in self.rows
        }
        self.assertTrue(selected_keys.isdisjoint(multi_if_keys))

    def test_manifest_matches_frozen_output(self) -> None:
        self.assertEqual(self.manifest["status"], "frozen_before_r0")
        self.assertFalse(
            self.manifest["selection"]["reference_answers_included"]
        )
        selected_source_ids = {
            row["metadata"]["source_id"] for row in self.rows
        }
        self.assertTrue(
            selected_source_ids.isdisjoint(builder.MANUAL_EXCLUSIONS)
        )
        self.assertEqual(
            self.manifest["output"]["sha256"],
            builder.sha256_file(builder.OUTPUT_PATH),
        )
        self.assertEqual(self.manifest["checks"]["rlvr_source_id_overlap"], 0)
        self.assertEqual(
            self.manifest["checks"]["multi_if_normalized_prompt_overlap"], 0
        )


if __name__ == "__main__":
    unittest.main()
