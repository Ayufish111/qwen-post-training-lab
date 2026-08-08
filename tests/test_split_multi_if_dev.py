"""Integration tests for the frozen Multi-IF dev/test split."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_DIR / "scripts" / "split_multi_if_dev.py"
INPUT = PROJECT_DIR / "data" / "eval" / "multi_if_zh.csv"


class SplitMultiIfDevTest(unittest.TestCase):
    def run_split(
        self, output_dir: Path, manifest: Path, *, force: bool = False
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(INPUT),
            "--output-dir",
            str(output_dir),
            "--manifest",
            str(manifest),
            "--seed",
            "42",
            "--dev-size",
            "80",
        ]
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

    def test_split_is_complete_disjoint_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "eval"
            manifest_path = root / "manifest.json"

            first = self.run_split(output_dir, manifest_path)
            self.assertEqual(first.returncode, 0, first.stderr)
            first_manifest_text = manifest_path.read_text(encoding="utf-8")
            manifest = json.loads(first_manifest_text)

            self.assertEqual(manifest["input"]["rows"], 454)
            self.assertEqual(manifest["algorithm"]["dev_size"], 80)
            self.assertEqual(manifest["algorithm"]["test_size"], 374)
            self.assertEqual(
                manifest["checks"]["dev_test_id_intersection_count"], 0
            )

            with (output_dir / "multi_if_dev.csv").open(
                "r", encoding="utf-8", newline=""
            ) as file:
                dev_rows = list(csv.DictReader(file))
            with (output_dir / "multi_if_test.csv").open(
                "r", encoding="utf-8", newline=""
            ) as file:
                test_rows = list(csv.DictReader(file))

            dev_ids = {row["key"] for row in dev_rows}
            test_ids = {row["key"] for row in test_rows}
            self.assertEqual(len(dev_rows), 80)
            self.assertEqual(len(test_rows), 374)
            self.assertEqual(len(dev_ids), 80)
            self.assertEqual(len(test_ids), 374)
            self.assertFalse(dev_ids & test_ids)

            refused = self.run_split(output_dir, manifest_path)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("Refusing to overwrite", refused.stderr)

            rebuilt = self.run_split(output_dir, manifest_path, force=True)
            self.assertEqual(rebuilt.returncode, 0, rebuilt.stderr)
            self.assertEqual(
                manifest_path.read_text(encoding="utf-8"), first_manifest_text
            )


if __name__ == "__main__":
    unittest.main()
