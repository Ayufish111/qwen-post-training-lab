#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${T1_PROJECT_DIR:-/root/autodl-tmp/qwen_post_training_lab_t1}"
VENV_DIR="${T1_VENV_DIR:-/root/autodl-tmp/venvs/qwen_rlvr}"
MODEL_PATH="${T1_MODEL_PATH:-/root/autodl-tmp/modelscope/models/Qwen--Qwen3-4B-Base/snapshots/master}"
OUTPUT_DIR="$PROJECT_DIR/outputs/distill/T1_v2"
LOG_DIR="$PROJECT_DIR/reports/distill"
LOG_PATH="$LOG_DIR/T1_v2_train.log"
SCREEN_NAME="t1_v2"

preflight() {
  test -f "$VENV_DIR/bin/activate" || { echo "missing venv: $VENV_DIR"; exit 1; }
  test -f "$MODEL_PATH/config.json" || { echo "missing model: $MODEL_PATH"; exit 1; }
  test -f "$PROJECT_DIR/data/distill/t1_thinking_accepted.jsonl" || { echo "missing frozen teacher data"; exit 1; }
  test -d "$PROJECT_DIR/data/cache/t1_qwen3_4b_1024" || { echo "missing frozen token cache"; exit 1; }

  source "$VENV_DIR/bin/activate"
  cd "$PROJECT_DIR"
  python - <<'PY'
import inspect
import json
from pathlib import Path

import peft
import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig

root = Path.cwd()
config = yaml.safe_load((root / "configs/rlvr.yaml").read_text(encoding="utf-8"))
t1 = config["t1"]
rows = [json.loads(line) for line in (root / t1["input"]).read_text(encoding="utf-8").splitlines() if line.strip()]
cache = load_from_disk(str(root / t1["cache"]))
assert len(rows) == 1461, len(rows)
assert set(cache) == {"t1_train", "t1_validation"}, set(cache)
assert "trainable_token_indices" in inspect.signature(LoraConfig).parameters
assert "lm_head" not in t1["qlora"]["target_modules"]
assert t1["qlora"]["structure_token_strings"] == ["<think>", "</think>", "<|im_end|>"]
assert t1["qlora"]["structure_token_loss_multiplier"] == 4.0
assert t1["generation_gate"]["status"] == "pending"
assert config["rlvr"]["blocked_until_t1_v2_generation_gate"] is True
print("preflight: OK")
print("torch:", torch.__version__)
print("peft:", peft.__version__)
print("teacher rows:", len(rows))
print("cache rows:", {name: len(split) for name, split in cache.items()})
PY
  # 增量修复包不携带完整 SFT/eval 语料；配置门禁已在上面的断言中核验。
  python -m unittest tests.test_train_distill -v
}

start_training() {
  preflight
  mkdir -p "$LOG_DIR" "$PROJECT_DIR/outputs/distill"
  if screen -list | grep -q "[.]$SCREEN_NAME"; then
    echo "screen already running: $SCREEN_NAME"
    exit 1
  fi
  if [ -d "$OUTPUT_DIR" ] && [ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    stamp="$(date +%Y%m%d_%H%M%S)"
    archive="${OUTPUT_DIR}_before_selective_${stamp}"
    mv -- "$OUTPUT_DIR" "$archive"
    echo "archived old output: $archive"
  fi

  screen -dmS "$SCREEN_NAME" bash -lc "
set -o pipefail
source '$VENV_DIR/bin/activate'
cd '$PROJECT_DIR'
export OMP_NUM_THREADS=8
python -u src/train_distill.py \\
  --experiment T1_v2 \\
  --config configs/rlvr.yaml \\
  --model '$MODEL_PATH' \\
  --local-files-only \\
  2>&1 | tee '$LOG_PATH'
"
  echo "started screen: $SCREEN_NAME"
  echo "log: $LOG_PATH"
}

status_training() {
  screen -list || true
  if [ -f "$LOG_PATH" ]; then
    tail -n 40 "$LOG_PATH"
  else
    echo "log not created yet: $LOG_PATH"
  fi
}

follow_training() {
  test -f "$LOG_PATH" || { echo "log not created yet: $LOG_PATH"; exit 1; }
  tail -f "$LOG_PATH"
}

case "${1:-}" in
  preflight) preflight ;;
  start) start_training ;;
  status) status_training ;;
  follow) follow_training ;;
  *) echo "usage: bash scripts/run_autodl_t1_v2_formal.sh {preflight|start|status|follow}"; exit 2 ;;
esac
