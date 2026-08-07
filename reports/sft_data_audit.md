# SFT Data Audit

Audit date: 2026-07-31

Status: approved for the frozen SFT experiments after automatic validation and a 100-row stratified manual audit. Semantic factual correctness of every row is not claimed.

## Frozen outputs

| File | Rows | SHA256 | Purpose |
|---|---:|---|---|
| `full_clean_10000.jsonl` | 10000 | `0df5919f5ade41dffd5e19f6c7c69cf12d82b95bba21f9c3cdeec6e8b7fff03d` | E4 final SFT |
| `ablation_clean_2000.jsonl` | 2000 | `24209a4315c73653476334c2a9cc22c5b25e698c91879bfa4fabbb151fc9f4f6` | E1/E2 target-module comparison |
| `ablation_raw_2000.jsonl` | 2000 | `2f96d1d955d593e7125a748428b6f8f988f5727059bf6f911ebdd33d11dc406b` | E3 data-quality ablation |

The clean ablation is a subset of the full clean set. The raw ablation is
disjoint from both clean files. Clean and raw have identical task-bucket and
character-length-bucket counts.

## Cleaning operations

- This rebuild repartitioned the frozen 12,000-row experimental universe. It came from the original 20,000-row candidate pass, where 11 unresolved placeholders and 1 duplicate question were removed before freezing.
- Hard rules removed only observable structural failures.
- Reward and visible risk signals were converted to a documented quality score.
- The 10,000-row set was selected across task buckets, then 2,000 clean/raw
  pairs were matched by task bucket and length bucket.

Hard-rule removals:

- None

## Distribution

- `code`: full=1907, clean=380, raw=380
- `creative_writing`: full=936, clean=186, raw=186
- `dialogue_roleplay`: full=469, clean=94, raw=94
- `extraction_classification`: full=572, clean=114, raw=114
- `general_instruction`: full=1827, clean=365, raw=365
- `knowledge_qa`: full=1817, clean=364, raw=364
- `math_reasoning`: full=1855, clean=372, raw=372
- `rewrite_summary`: full=437, clean=87, raw=87
- `translation`: full=180, clean=38, raw=38

- Full clean: quality mean/median=45.83/52.18, reward mean/median=0.66/1.19
- Ablation clean: quality mean/median=59.04/60.00, reward mean/median=7.49/6.38
- Ablation raw: quality mean/median=28.14/28.00, reward mean/median=-9.68/-8.88

## Interpretation boundary

This pipeline demonstrates schema validation, exact deduplication, transparent
rule-based cleaning, risk scoring, balanced sampling, and controlled ablation
construction. It does not prove every answer is factually correct. Formal model
claims require the frozen external evaluation suite and error analysis after
training; a sampled review is evidence for audit only, not a cleaning step.

<!-- MANUAL_AUDIT_START -->
## Manual stratified audit

- Method: seed 42; one guaranteed sample per task bucket, then seeded random fill.
- Reviewed: 100 rows from `full_clean_10000.jsonl`.
- Task coverage: code=21, creative_writing=12, dialogue_roleplay=3, extraction_classification=7, general_instruction=12, knowledge_qa=22, math_reasoning=16, rewrite_summary=4, translation=3.
- Result: pass=99, minor=0, major=1.
- Gate decision: passed; the observed major-error rate was 1/100 (1%).

Observed non-pass cases:

- #4 `major` (code): The sentence is active voice, but the answer treats it as a passive-voice task; the SQL also extracts `nourishment` as the verb and therefore does not satisfy the request; prompt: 在我的SQL数据库的表“sentences”中，有一个名为“sentence”的列，其中包含句子“Lana provided nourishment to the brown chickens.” 你能提供一个SQL查询来识别句子中使用的被

This is a sampled quality audit, not a claim that all 10,000 rows were manually
cleaned or fact-checked.
<!-- MANUAL_AUDIT_END -->
