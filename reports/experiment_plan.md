# SFT Experiment Protocol

## Research questions

1. 在相同 2,000 条 clean 数据与训练预算下，`q/k/v/o` 和
   `all-linear` 哪种 LoRA target 更好？
2. 固定胜出的 target 和全部超参数后，clean 2,000 是否优于匹配的
   raw 2,000？
3. 胜出配置扩大到 full clean 10,000 后，是否稳定优于 Base 模型？

## Frozen matrix

| ID | Data | Target modules | Role |
|---|---|---|---|
| B0 | none | none | evaluation-only Base baseline |
| S1 | ablation clean 2k | attention | target ablation |
| S2 | same clean 2k | all-linear | target ablation |
| S3 | matched raw 2k | S1/S2 winner | data-quality ablation |
| S4 | full clean 10k | S1/S2 winner | final SFT |

S1/S2/S3 use the same optimizer-step budget and all common hyperparameters.
Clean/raw prompts are disjoint and their task/length strata are identical.
S4 starts again from `Qwen/Qwen3-4B-Base`, never from a pilot checkpoint.

## Gates

- G1 raw data: exact counts, valid schema, zero exact prompt duplicates,
  clean subset relation, raw disjointness, matched ablation strata.
- G2 tokenized data: sequence length <= 1024, three tensor lengths equal,
  prompt labels are `-100`, active labels equal response token IDs.
- G3 baseline: freeze generation settings and save B0 raw generations.
- G4 pilots: use instruction-following evaluation first, pairwise quality second,
  validation loss only as a diagnostic.
- G5 final: compare B0 and S4 with the same frozen evaluation and report
  regressions as well as gains.

## Reporting boundary

Automatic rules and reward-based risk scores are reproducible cleaning signals,
not factual correctness judges. Sampled human review is an audit. It must never
be described as if every row was manually cleaned.
