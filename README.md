# Qwen3 Post-Training Lab

A reproducible post-training study for `Qwen3-4B-Base` focused on Chinese instruction following. The repository combines controlled SFT ablations, teacher-guided reasoning cold start, constraint-based RLVR, merged-model vLLM deployment, and auditable Multi-IF evaluation.

## Scope

The project evaluates three parts of the post-training pipeline:

- SFT data quality, dataset size, and LoRA target modules.
- Teacher distillation for stabilizing Qwen3 thinking/answer structure before RLVR.
- GRPO, DAPO-style, and constraint-aware DAPO (CA-DAPO) under a fixed RLVR protocol.

The primary benchmark is the 454-row Multi-IF Chinese multi-turn instruction-following suite. All formal R0/R1/R2 evaluations use the same data hash and deterministic native vLLM decoding.

## Results

### SFT ablation

| Run | Configuration | Turn 1 | Turn 2 | Turn 3 | Mean |
|---|---|---:|---:|---:|---:|
| B0 | Qwen3-4B-Base | 0.404528 | 0.204584 | 0.160707 | 0.256606 |
| S1 | Clean 2k, attention LoRA | 0.469344 | 0.340690 | 0.231513 | 0.347182 |
| S2 | Clean 2k, all-linear LoRA | 0.501450 | 0.348407 | 0.235692 | **0.361849** |
| S3 | Raw 2k, all-linear LoRA | 0.453707 | 0.306988 | 0.203443 | 0.321379 |
| S4 | Clean 10k, all-linear LoRA | 0.449160 | 0.324287 | 0.229305 | 0.334251 |

The controlled comparison favors all-linear LoRA and clean data in this setup. The 10k run does not outperform the matched clean 2k run, so the results do not support a simple “more data is always better” claim.

### Formal RLVR comparison

| Run | Algorithm | Turn 1 | Turn 2 | Turn 3 | Mean |
|---|---|---:|---:|---:|---:|
| R0 | GRPO | 0.478461 | 0.416477 | 0.290902 | 0.395280 |
| R1 | DAPO-style | **0.483289** | **0.448247** | 0.288128 | **0.406555** |
| R2 | CA-DAPO | 0.473928 | 0.437439 | **0.304875** | 0.405414 |

R1 has the highest three-turn mean, while R2 has the strongest third-turn score. The gain over GRPO is approximately one percentage point and should be treated as a small single-seed result rather than a general algorithmic claim.

## Pipeline

1. Build and audit SFT and RLVR datasets.
2. Run B0/S1-S4 controlled SFT experiments.
3. Generate and filter teacher reasoning trajectories.
4. Train the `T1_v2` reasoning cold-start adapter and verify its generation gate.
5. Train R0/R1/R2 from the same cold-start initialization.
6. Merge adapters and run deterministic native vLLM evaluation.
7. Audit JSONL integrity, strict/loose checker results, clipping, empty outputs, and thinking structure.

## Repository Layout

```text
configs/                     Experiment and RLVR configuration
data/                       Frozen datasets and audited training data
outputs/sft/                 SFT metrics and Git-LFS adapters
outputs/rlvr/*/manifest.json RLVR provenance manifests
reports/eval/                B0/S1-S4 evaluation results
reports/eval_rlvr/           Formal R0/R1/R2 JSONL and summaries
reports/distill/             Distillation audits and generation gates
reports/rlvr/                RLVR data audits and validation reports
scripts/                     Dataset, training, audit, and deployment entry points
src/                         Training, reward, sampling, merge, and evaluation code
tests/                       Unit and contract tests
```

Large local RLVR/distillation checkpoints, generated logs, smoke runs, and deployment bundles are intentionally excluded from the public source tree. Evaluation summaries and provenance manifests are included so the reported results remain auditable.

## Reproduction

The commands below assume a local model snapshot and the dependencies required by the selected backend (`transformers`, `peft`, `trl`, `datasets`, `bitsandbytes`, and optionally `vllm`). GPU training and vLLM evaluation are recommended.

### SFT baseline and ablations

```bash
python src/build_sft_datasets.py
python scripts/validate_dataset.py
python scripts/audit_sft.py
python src/preprocess_sft.py
python src/train_sft.py --experiment S1
python src/train_sft.py --experiment S2
python src/train_sft.py --experiment S3
python src/train_sft.py --experiment S4
```

Run the HF Multi-IF evaluator after preparing the frozen benchmark:

```bash
python scripts/prepare_multi_if.py
python src/evaluate_multi_if.py --experiment-id B0
```

### Distillation cold start

```bash
python src/build_distill_dataset.py --local-files-only
python src/train_distill.py --experiment T1_v2 --config configs/rlvr.yaml --local-files-only
```

The formal cold-start wrapper is available at `scripts/run_autodl_t1_v2_formal.sh`.

### RLVR training

R0, R1, and R2 share the same configuration and initialization contract. A preflight check can be run without starting training:

```bash
python src/train_rlvr.py --experiment R0 --config configs/rlvr.yaml --preflight-only
python src/train_rlvr.py --experiment R1 --config configs/rlvr.yaml --preflight-only
python src/train_rlvr.py --experiment R2 --config configs/rlvr.yaml --preflight-only
```

Full training requires a compatible GPU environment and the verified `T1_v2` adapter path.

### Native vLLM evaluation

On the Linux/AutoDL environment used for the formal comparison:

```bash
RUN_ID=R0 PILOT_LIMIT=454 MAX_NEW_TOKENS=2048 EVAL_MODE=native \
  bash scripts/run_autodl_vllm_merged_pilot.sh
```

Repeat with `RUN_ID=R1` and `RUN_ID=R2`. The script writes one JSONL answer file and one summary JSON per run under `reports/eval_rlvr/`.

## Limitations

- The study uses one model size, one primary benchmark, and one seed per formal RL run.
- The official checker has known edge cases for some Chinese punctuation, word-boundary, and sentence-count rules.
- Long reasoning generations still show substantial clipping, empty answers, repetition, and unclosed thinking structures.
- HF SFT scores and native vLLM RLVR scores are different evaluation groups and should not be compared as direct absolute improvements.
- The results establish an auditable engineering baseline, not a universal ranking of RL algorithms.

## License

The source code is released under the [MIT License](LICENSE). Model weights and upstream datasets remain subject to their original licenses and terms.

