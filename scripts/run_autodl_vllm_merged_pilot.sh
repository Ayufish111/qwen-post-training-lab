#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/root/autodl-tmp/qwen_post_training_lab_t1"
BASE_MODEL="/root/autodl-tmp/modelscope/models/Qwen--Qwen3-4B-Base/snapshots/master"
RUN_ID="${RUN_ID:-R0}"
PILOT_LIMIT="${PILOT_LIMIT:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
EVAL_MODE="${EVAL_MODE:-native}"
MERGED_DIR="/root/autodl-tmp/merged_models/${RUN_ID}"
ADAPTER_DIR="$PROJECT_DIR/outputs/rlvr/${RUN_ID}/final_adapter"
MERGE_PY="/root/autodl-tmp/venvs/qwen_rlvr/bin/python"
VLLM_PY="/root/autodl-tmp/venvs/qwen_vllm/bin/python"
OUTPUT_DIR="$PROJECT_DIR/reports/eval_rlvr"
case "$EVAL_MODE" in
  native)
    MODE_SUFFIX="native"
    FALLBACK_ARG=()
    ;;
  fallback)
    MODE_SUFFIX="fallback"
    FALLBACK_ARG=(--answer-only-fallback)
    ;;
  *)
    echo "EVAL_MODE must be native or fallback, got: $EVAL_MODE" >&2
    exit 2
    ;;
esac
EXPERIMENT_ID="${RUN_ID}-vllm-merged-${MODE_SUFFIX}${PILOT_LIMIT}-${MAX_NEW_TOKENS}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR" "$(dirname "$MERGED_DIR")"
export OMP_NUM_THREADS=8
export NLTK_DATA="$PROJECT_DIR/nltk_data"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]]; then
  echo "Missing adapter for RUN_ID=$RUN_ID: $ADAPTER_DIR" >&2
  exit 1
fi
if pgrep -af 'evaluate_multi_if_vllm_final.py|EngineCore' >/dev/null; then
  echo "Refusing to start while another vLLM evaluation/EngineCore process exists." >&2
  pgrep -af 'evaluate_multi_if_vllm_final.py|EngineCore' >&2
  exit 1
fi

if [[ ! -f "$MERGED_DIR/merge_manifest.json" ]]; then
  available_kb=$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')
  required_kb=$((12 * 1024 * 1024))
  if (( available_kb < required_kb )); then
    echo "At least 12 GiB free is required to merge $RUN_ID." >&2
    df -h /root/autodl-tmp >&2
    exit 1
  fi
  "$MERGE_PY" -c "import peft, torch, transformers; print('merge env:', 'peft', peft.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
  "$MERGE_PY" -u src/merge_peft_adapter.py \
    --model "$BASE_MODEL" \
    --adapter "$ADAPTER_DIR" \
    --output "$MERGED_DIR" \
    2>&1 | tee "$OUTPUT_DIR/${RUN_ID}_merge_for_vllm.log"
else
  echo "reusing merged model: $MERGED_DIR"
  cat "$MERGED_DIR/merge_manifest.json"
fi

"$VLLM_PY" -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('${MERGED_DIR}', local_files_only=True); assert hasattr(t, 'all_special_tokens_extended'); print('merged tokenizer: OK', type(t).__name__)"
"$VLLM_PY" -m unittest tests.test_evaluate_multi_if_vllm tests.test_vllm_eos_config -v
"$VLLM_PY" -c "import transformers, vllm; assert transformers.__version__ == '4.55.2'; print('eval env:', 'vllm', vllm.__version__, 'transformers', transformers.__version__)"

"$VLLM_PY" -u src/evaluate_multi_if_vllm_final.py \
  --experiment-id "$EXPERIMENT_ID" \
  --model "$MERGED_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --limit "$PILOT_LIMIT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --gpu-memory-utilization 0.85 \
  "${FALLBACK_ARG[@]}" \
  2>&1 | tee "$OUTPUT_DIR/${EXPERIMENT_ID}.log"