"""Integration tests for the deterministic RLVR constraint-data builder."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_DIR / "scripts" / "build_constraint_rlvr_data.py"
CONFIG = PROJECT_DIR / "configs" / "rlvr.yaml"

sys.path.insert(0, str(PROJECT_DIR))
from scripts import build_constraint_rlvr_data as builder  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BuildConstraintRlvrDataTest(unittest.TestCase):
    def run_builder(
        self, config_path: Path, *, force: bool = False
    ) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(SCRIPT), "--config", str(config_path)]
        if force:
            command.append("--force")
        return subprocess.run(
            command,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

    def test_keyword_candidates_are_readable_prompt_phrases(self) -> None:
        ads = builder.prompt_term_candidates(
            "指点如何在Google AdWords上创建广告投放"
        )
        self.assertIn("Google AdWords", ads)
        self.assertNotIn("告投", ads)

        translation = builder.prompt_term_candidates("帮我把这句话翻译成文言文")
        self.assertEqual(translation, {"文言文"})

        urgent = builder.prompt_term_candidates("有什么事如此急迫")
        self.assertNotIn("此急迫", urgent)

        how = builder.prompt_term_candidates("如何对社会做出最有利的贡献")
        self.assertNotIn("何对社会做出最有利的贡献", how)

        self.assertEqual(
            builder.prompt_term_candidates(
                "创建一个包含20世纪出版的5本经典书籍的列表"
            ),
            set(),
        )
        self.assertEqual(
            builder.prompt_term_candidates("给出一个自然现象，请求解释其基本原理\n彩虹"),
            {"自然现象", "基本原理", "彩虹"},
        )
        self.assertTrue(builder.has_explicit_output_constraint("最多50个字"))
        self.assertTrue(builder.has_explicit_output_constraint("写一个单行简介"))

        multiple_choice = builder.prompt_term_candidates(
            "某建筑队借用A建筑公司的资质，以\nA. 有效\nB. 可撤销\nC. 无效\nD. 效力待定"
        )
        self.assertFalse(any(term.startswith("D.") for term in multiple_choice))

    def test_outputs_are_valid_disjoint_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
            config["data"]["source"] = str(
                PROJECT_DIR / config["data"]["source"]
            )
            for key in ("source", "dev", "test"):
                config["evaluation_boundary"][key] = str(
                    PROJECT_DIR / config["evaluation_boundary"][key]
                )

            paths = {
                "train": root / "constraint_train_2000.jsonl",
                "validation": root / "constraint_validation_100.jsonl",
                "review": root / "manual_review_queue_100.jsonl",
                "manifest": root / "manifest.json",
                "audit": root / "audit.md",
            }
            config["data"]["train_output"] = str(paths["train"])
            config["data"]["validation_output"] = str(paths["validation"])
            config["data"]["manual_review_output"] = str(paths["review"])
            config["data"]["manifest_output"] = str(paths["manifest"])
            config["data"]["audit_output"] = str(paths["audit"])
            config_path = root / "rlvr.test.yaml"
            config_path.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )

            first = self.run_builder(config_path)
            self.assertEqual(first.returncode, 0, first.stderr)

            train_rows = read_jsonl(paths["train"])
            validation_rows = read_jsonl(paths["validation"])
            review_rows = read_jsonl(paths["review"])
            all_rows = train_rows + validation_rows
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

            self.assertEqual(len(train_rows), 2000)
            self.assertEqual(len(validation_rows), 100)
            self.assertEqual(len(review_rows), 100)
            self.assertEqual(manifest["status"], "pending_manual_review")
            self.assertGreater(
                manifest["checks"]["source_intrinsic_constraint_removed"], 0
            )
            self.assertEqual(len({row["id"] for row in all_rows}), 2100)
            self.assertEqual(
                len({row["metadata"]["source_id"] for row in all_rows}), 2100
            )

            expected_roles = {
                1: ["user"],
                2: ["user", "assistant", "user"],
                3: ["user", "assistant", "user", "assistant", "user"],
            }
            forbidden_pairs = [
                set(pair)
                for pair in config["constraint_sampling"]["forbidden_pairs"]
            ]
            exclusive_groups = [
                set(group)
                for group in config["constraint_sampling"][
                    "mutually_exclusive_groups"
                ]
            ]
            checker_module = builder.load_official_checker()
            for row in all_rows:
                roles = [message["role"] for message in row["messages"]]
                self.assertEqual(
                    roles, expected_roles[row["metadata"]["context_turns"]]
                )
                self.assertEqual(roles[-1], "user")
                self.assertEqual(len(row["instruction_ids"]), len(row["kwargs"]))
                self.assertEqual(
                    len(row["instruction_ids"]), row["metadata"]["constraint_count"]
                )
                selected = set(row["instruction_ids"])
                self.assertTrue(
                    all(not pair <= selected for pair in forbidden_pairs)
                )
                self.assertTrue(
                    all(len(group & selected) <= 1 for group in exclusive_groups)
                )
                for instruction_id, kwargs in zip(
                    row["instruction_ids"], row["kwargs"]
                ):
                    self.assertNotIn("核心结论", json.dumps(kwargs, ensure_ascii=False))
                    checker = checker_module.INSTRUCTION_DICT[instruction_id](
                        instruction_id
                    )
                    checker.build_description(**kwargs)

            multi_if_prompts = builder.load_multi_if_prompts(
                [
                    Path(config["evaluation_boundary"]["dev"]),
                    Path(config["evaluation_boundary"]["test"]),
                ]
            )
            self.assertFalse(
                {
                    builder.normalize_prompt(row["messages"][-1]["content"])
                    for row in all_rows
                }
                & multi_if_prompts
            )

            tracked_paths = list(paths.values())
            first_hashes = {path.name: sha256_file(path) for path in tracked_paths}

            refused = self.run_builder(config_path)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("Refusing to overwrite", refused.stderr)

            rebuilt = self.run_builder(config_path, force=True)
            self.assertEqual(rebuilt.returncode, 0, rebuilt.stderr)
            rebuilt_hashes = {
                path.name: sha256_file(path) for path in tracked_paths
            }
            self.assertEqual(rebuilt_hashes, first_hashes)


if __name__ == "__main__":
    unittest.main()
