# Qwen3 中文后训练升级实验方案 v5.3

> 主题：从可审计 SFT 消融，扩展到 GRPO、DAPO 与 CA-DAPO 的受控 RLVR 对比  
> 更新时间：2026-08-08
> 项目定位：实习项目与工程研究，不宣称达到论文级通用算法创新  
> 预算上限：S4-HF 完成后的新增 AutoDL 费用不超过人民币 50 元  
> 本文状态：P0/S4-HF 已闭合；已修正 DA-CoTD 命名并加入固定质量护栏；组 2 新增脚本尚未创建

---

## 0. 执行摘要

SFT 实验 B0/S1/S2/S3/S4 及其同协议 HF 评测均已完成，不修改、不重跑。P0 已闭合，当前进入
组 2 的实现前冻结与数据边界阶段。

组 2 不再继续旧版 T2 二次蒸馏、SimPO 或多条并行扩展，而是固定为一条精简的 RLVR
主线：

```text
Qwen3 Base
   -> T1：2k DA-CoTD-inspired 难度感知 Thinking cold-start
       |-> R0：GRPO
       |-> R1：单卡 DAPO-style reproduction
       `-> R2：CA-DAPO（P2，约束感知采样变体）
```

执行优先级固定为：P0 已完成；P1 跑通 T1、R0、R1；P2 只在 P1 技术稳定、test 尚未
打开且预算满足客观门禁时运行 R2。若进入 P2，最终 test 评测 R0/R1/R2 三次；若未进入 P2，
最终 test 只评测 R0/R1 两次。不能根据 dev/test 分数决定是否运行或隐藏 R2。

本项目的核心问题是：

> 在相同初始化、训练数据、生成预算和训练步数下，DAPO 是否优于 GRPO；在 DAPO 的其余
> 机制完全不变时，根据约束类型历史失败率调整训练 prompt 采样，是否能进一步改善中文
> 多轮指令遵循。

---

## 1. 当前冻结状态

### 1.1 组 1：SFT 实验矩阵

| ID | 数据 | LoRA target | Multi-IF-zh HF 结果（turn1/2/3） | 状态 |
|---|---|---|---|---|
| B0 | 无训练 | 无 | 0.404528 / 0.204584 / 0.160707 | 完成，454/454 |
| S1 | clean 2k | attention | 0.469344 / 0.340690 / 0.231513 | 完成，454/454 |
| S2 | clean 2k | all-linear | 0.501450 / 0.348407 / 0.235692 | 完成，454/454 |
| S3 | raw 2k | all-linear | 0.453707 / 0.306988 / 0.203443 | 完成，454/454 |
| S4 | full clean 10k | all-linear | 0.449160 / 0.324287 / 0.229305 | 完成，454/454 |

已成立结论：

- S2 三轮均优于 S1：在当前配置下 all-linear 优于 attention target。
- S2 三轮均优于 S3：在相同规模与训练预算下 clean 数据优于 raw 数据。
- S4 三轮均低于 S2：在当前固定训练预算和评测协议下，clean 从 2k 增加到 10k 未带来提升。
- S4 三轮均高于 B0：完整 SFT 相比 Base 有收益。

组 1 的五份评测使用同一数据文件，SHA256 为：

```text
ce43827ef4b43c6fde34038180ed2deb4795a7aef331e187de5fd0a5da15601c
```

### 1.2 S4-HF 验收结果

验收条件 G-S4 已全部通过：

```text
reports/eval/S4_multi_if_zh.jsonl 行数 = 454
唯一 id 数 = 454
summary.rows = 454
summary.data_sha256 = 上述冻结 SHA256
summary.decoding.max_new_tokens = 512
summary.decoding.do_sample = false
summary.decoding.enable_thinking = false
```

额外审计发现：S4 的 1362 次 turn 生成全部达到 512 token 上限；S2/S3 同样全部触顶，S1 为
1361/1362，B0 为 946/1362。组 1 的同协议相对比较仍可使用，但不能宣称自然对话质量或长度控制
良好。详细结果见 `reports/final_comparison.md`。

下一步唯一实现任务是冻结 Multi-IF dev/test 数据边界；在该切分与 manifest 验收前，不构造 T1
数据、不训练模型，也不实现 RL trainer。

---

## 2. 项目目标、主张边界与预注册假设

### 2.1 项目目标

1. 保留组 1 的数据治理、LoRA target、数据质量和数据规模证据链。
2. 用 DA-CoTD-inspired 难度感知 reasoning compression 构造共同 T1 cold-start。
3. 用 TRL 原生 GRPO 建立基线，并实现一个明确标注一致性边界的单卡 DAPO-style 对比。
4. 在 DAPO 上只增加一个可归因的机制：Constraint-Aware Dynamic Sampling。
5. 使用程序化约束验证器构造 RLVR 奖励，避免依赖付费模型裁判。
6. 报告效果、生成长度、吞吐、显存和费用，不只报告最终分数。

### 2.2 不做的主张

- 不宣称 CA-DAPO 是通用的新强化学习算法。
- 不宣称单 seed、单 benchmark 的结果具有统计显著性。
- 不把单卡工程近似描述为官方 DAPO 大规模系统的完整复现。
- 不把 HF 与 vLLM 的绝对分数跨框架直接比较。
- 不因 CA-DAPO 结果不理想而删除该组或临时更换主比较。

### 2.3 预注册假设

定义组 2 主指标：

```text
primary_score = mean(
    turn1 official_overall_average,
    turn2 official_overall_average,
    turn3 official_overall_average
)
```

假设 H1：相同 T1 初始化与预算下，DAPO 的 `primary_score` 高于 GRPO。  
假设 H2：CA-DAPO 的 `primary_score` 高于 DAPO，且至少两个 turn 不低于 DAPO。  
假设 H3：CA-DAPO 对训练期间低通过率约束类别的提升，不以明显增加平均输出长度为代价。

这些是假设，不是承诺。若不成立，仍完整报告负结果和训练曲线。

---

## 3. 最小且可归因的实验矩阵

### 3.1 共同初始化 T1

T1 是 R0/R1/R2 的共同 DA-CoTD-inspired cold-start checkpoint，不进入最终 test 三组主表。T1 只在开发集上
验收，以确认模型能够稳定产生可解析的 thinking 和最终回答。

### 3.2 P1/P2 实验

| ID | 初始化 | 算法 | 与上一组的唯一主要差异 | 最终 test |
|---|---|---|---|---|
| R0 | T1 | GRPO | 标准组内相对优势与对称裁剪 | 必跑 |
| R1 | T1 | DAPO-style reproduction | 在统一 trainer 中近似实现 DAPO 四项机制 | P1 必跑 |
| R2 | T1 | CA-DAPO | 在 R1 上只增加约束感知 prompt 采样 | P2 条件执行 |

固定不变项：

- 基础模型和 T1 adapter。
- RLVR 训练 prompt 集及其 SHA256。
- LoRA target、rank、alpha、dropout。
- reward checker 和基础约束奖励。
- 随机 seed、训练步数、每个 prompt 的生成数、最大生成长度。
- 最终评测数据、vLLM 版本、生成参数和评分器版本。

### 3.3 结果报告规则

- P2 是否执行必须在打开 untouched test 前，按训练稳定性、剩余预算和实现完整性决定。
- 若 R2 已启动或完成，最终报告固定包含三组；不能执行“CA-DAPO 输了就只报告 DAPO vs GRPO”。
- 若按客观门禁未进入 P2，报告必须说明缺少 R2 的原因，不把 P1 结果包装成 CA-DAPO 验证。
- 如果某组训练因技术失败未完成，报告失败阶段、日志和原因，不能用另一组结果代替。
- 一次 full run 默认 seed=42。由于预算限制不做三 seed，因此只做描述性结论。

---

## 4. 数据治理与防泄漏协议

### 4.1 Multi-IF 冻结切分

对 `data/eval/multi_if_zh.csv` 的 454 个唯一 key 排序后，用 `random.Random(42)` 固定打乱：

```text
failure-dev：80 条
untouched test：374 条
```

生成以下只读产物：

```text
data/eval/multi_if_dev.csv
data/eval/multi_if_test.csv
data/eval/multi_if_dev_ids.txt
data/eval/multi_if_test_ids.txt
reports/manifests/multi_if_split_manifest.json
```

manifest 必须包含：

- 输入数据 SHA256。
- seed、切分算法描述和脚本版本。
- dev/test 行数、唯一 ID 数和各自 SHA256。
- dev/test ID 交集计数，必须为 0。
- 归一化 prompt 重合检查结果。

test 在 R0/R1/R2 模型、配置、脚本和报告模板全部冻结前不得生成答案或查看分数。

### 4.2 RLVR 训练数据不能来自评测原题

Multi-IF dev/test 的 prompt、回答和 ID 均不得进入 T1 或 RL 训练。训练数据单独构造：

```text
data/rlvr/constraint_train_2000.jsonl
data/rlvr/constraint_validation_100.jsonl
reports/rlvr/constraint_data_audit.md
```

构造原则：

1. 只复用官方约束检查器支持的约束类型和参数 schema，不复制评测 prompt 文本。
2. 主题从 full_clean 数据的宽泛任务类型抽取，但不得复制完整 prompt。
3. 每条样本保存 `messages`、`instruction_ids`、`kwargs`、`constraint_categories` 和来源元数据。
4. 训练集包含单轮及带历史上下文的当前轮样本；每次 RL completion 只奖励当前 assistant 回答。
5. 对训练集与 Multi-IF dev/test 做 NFKC、空白折叠后的精确重合检查，计数必须为 0。
6. 人工抽查 100 条，检查 prompt 自然性、约束可满足性、标注一致性和答案可验证性。

建议数据构成：

```text
单轮上下文：70%
二轮上下文：20%
三轮上下文：10%
```

上述比例在首次数据构建前写入配置并冻结，不根据 test 结果调整。

### 4.3 数据验收 G-DATA

```text
train = 2000 条，validation = 100 条
所有 key 唯一
所有 instruction_id 可被官方 checker 加载
dev/test/train 三者归一化 prompt 无精确重合
人工抽查 100 条并记录结论
数据文件和 manifest 均有 SHA256
```

---

## 5. T1：DA-CoTD-inspired Thinking cold-start

### 5.1 定位

T1 不是最终创新结论，而是三个 RL 算法的共同初始化。它使用少量、高通过率且经过难度自适应
压缩的教师推理轨迹，
避免直接让基础模型进行高方差在线探索。

T1 与 RL 使用完全相同的 2k 训练 prompt ID。T1 先在这批 prompt 上学习教师轨迹，R0/R1/R2
再在同一 prompt 池上进行 on-policy rollout，这一设定明确记为 **warmed-up on-policy**。
独立的 100 条 synthetic validation 和 Multi-IF failure-dev 只用于验收，不进入 T1/RL 权重更新。

### 5.2 教师生成

首选教师：`Qwen/Qwen3-4B-Instruct`。若模型标识、thinking API 或输出解析在一条样本上失败，
立即停止，不批量生成。

生成顺序：

```text
1 条模板与字段验证
-> 100 条速度/质量 pilot
-> 2k 批量生成
```

生成约束：

- `enable_thinking=True`。
- 使用 tokenizer 的 `apply_chat_template`，禁止手拼特殊 token。
- assistant 消息使用 `reasoning_content` 和 `content` 两个字段。
- 对最终 `content` 运行约束 checker；thinking 不参与格式约束计数。
- 首次生成失败时最多重采样 2 次，仍失败则记录并丢弃。
- 记录 reasoning tokens、answer tokens、通过率、重采样率、截断率和生成耗时。

目标产物：

```text
data/distill/t1_thinking_accepted.jsonl
data/distill/t1_thinking_rejected.jsonl
reports/distill/t1_generation_audit.json
```

### 5.3 DA-CoTD-inspired 难度感知压缩

本项目参考 `DA-CoTD: Efficient Chain-of-Thought Reasoning with Difficulty-Aware CoT-Distillation`
（NeurIPS 2025 Workshop）的难度感知思想，将其作为 T1 数据预处理，而不是新增一个独立模型分支。
它只改变 T1 的
`reasoning_content` 长度，不改最终 `content`，也不改变 R0/R1/R2 的共同初始化关系。

难度只能使用训练数据本身的可见信息计算，不能读取 Multi-IF dev/test：

```text
约束数量、约束类别数量、多轮上下文深度、教师首次验证是否通过、重采样次数
```

预注册三档 reasoning budget：

```text
easy：最多 128 reasoning tokens
medium：最多 256 reasoning tokens
hard：最多 512 reasoning tokens
```

压缩规则：

1. 先保存教师原始 reasoning，原始文件只用于审计，不直接作为 T1 训练输入。
2. 只压缩 `reasoning_content`，最终 `content` 原样保留。
3. 保留任务拆解、关键约束判断和最终格式检查，删除重复自检和无效复述。
4. 压缩后再次用 tokenizer 统计长度，并验证最终 answer 的约束结果没有变化。
5. 记录原始 token 数、压缩 token 数、压缩比例、answer 通过率、截断率和每个难度档的覆盖率。
6. 超过 hard budget 的样本记录为 overflow；不通过验证的样本不能仅靠压缩强行保留。

目标产物：

```text
data/distill/t1_thinking_raw.jsonl
data/distill/t1_thinking_compressed.jsonl
reports/distill/da_cotd_compression_audit.json
```

实现入口：`scripts/compress_thinking_da_cotd.py`。本项目没有复现论文的完整教师、难度估计器和训练
设置，只借鉴“按样本难度分配 reasoning 预算”的思想。文件、日志和报告统一使用
`DA-CoTD-inspired difficulty-aware CoT distillation`，不得写成“完整复现 DA-CoTD”。easy/medium/hard
的 128/256/512 token 预算是本项目预注册的工程设置，不宣称来自原论文默认配置。

### 5.4 T1 训练

主尝试使用 Qwen3-4B-Base。若 4090D 24GB 上的 RL smoke test 无法满足显存和吞吐门禁，
则在任何 R0/R1/R2 full run 前，将 T1 及三组 RL 全部统一降为 Qwen3-1.7B-Base。

初始训练配置：

```yaml
max_seq_length: 1024
load_in_4bit: true
quant_type: nf4
double_quant: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target: all-linear
learning_rate: 2.0e-4
effective_batch_size: 16
max_steps: 200
seed: 42
```

T1 使用 `t1_thinking_compressed.jsonl` 训练；原始长 reasoning 不进入 loss。T1 只评测
failure-dev 和合成 validation，不进入 untouched test。

验收 G-T1：

- final adapter、trainer state、训练/验证 loss 和配置快照齐全。
- 生成格式可被 Qwen3 reasoning parser 稳定拆分。
- failure-dev 无缺行，约束评分器无异常。
- 不要求 T1 一定优于 S4，因为它不是本组最终报告对象。
- 不根据 T1 的 dev 分数修改 easy/medium/hard budget；预算在批量生成前冻结。

---

## 6. RLVR 奖励与算法定义

### 6.1 通用约束奖励

对生成结果先拆分 `reasoning_content` 与最终 `content`。只对最终 `content` 检查格式约束，
thinking tokens 计入生成成本和超长统计，但不计入句数、段落数、关键词等最终答案约束。

基础奖励：

```text
r_instruction = 通过的严格约束数 / 约束总数
r_prompt      = 1，若全部严格约束通过；否则 0
r_core        = 0.7 * r_instruction + 0.3 * r_prompt
```

同一实验内部不使用 LLM-as-a-judge。非法输出、无法解析输出和没有最终 answer 的输出记为 0，
并单独计数。

### 6.2 R0：GRPO

R0 使用标准 group-relative advantage：

- 每个 prompt 生成 4 个 completion。
- 组内对 reward 标准化得到 advantage。
- 对称 clipping，初始 `epsilon=0.2`。
- 不启用 CA prompt 重采样。
- 不启用 DAPO Clip-Higher、Token-Level Loss 和 Soft Overlong Punishment。

### 6.3 R1：单卡 DAPO-style reproduction

R1 以 TRL 原生 GRPO trainer 为共同基座，在与 R0 相同初始化、数据和 rollout 预算下工程化
实现 DAPO 的四个核心机制：

1. **Clip-Higher**：下界 `epsilon_low=0.2`，上界初始 `epsilon_high=0.28`。
2. **Dynamic Sampling**：丢弃组内 reward 全相同、无法产生有效相对 advantage 的 prompt 组，
   从同一冻结训练池补采样。
3. **Token-Level Policy Gradient Loss**：按有效 completion token 聚合损失，避免样本级长度偏置。
4. **Soft Overlong Punishment**：在最大生成长度前设置缓冲区并线性惩罚过长 completion，避免
   硬截断产生噪声奖励。

具体公式必须按 DAPO 论文和锁定实现逐项写入代码注释与单元测试。只有 import 成功或源码出现
`dapo` 字符串，不能算复现完成。报告必须逐项标注“公式一致 / 工程近似 / 未激活”，因此本组
在验证完成前统一称为 `DAPO-style reproduction`，不简称为“原始 DAPO”。

适用边界必须量化：记录进入 overlong buffer 的 completion 占比、completion 长度分布，以及
token-level 与 sequence-level loss 的数值差异。若绝大多数输出远短于 512 tokens，Soft
Overlong Punishment 和 Token-Level Loss 可能基本不激活，这应作为结果而不是被隐去。

### 6.4 R2：CA-DAPO

R2 完整继承 R1 的模型、reward、DAPO 四项机制和训练预算，只改变训练 prompt 的采样概率。

CA sampler 的作用位置固定如下，禁止实现时临时改变：

```text
完整冻结训练池
  -> R1：均匀抽取候选 prompt
  -> R2：按 CA 权重抽取候选 prompt
  -> 为每个 prompt 生成 group completions
  -> DAPO Dynamic Sampling 丢弃零方差组
  -> R1 从均匀分布补采样；R2 从同一个 CA 权重分布补采样
  -> 形成有效训练 batch
```

因此 R1/R2 的 Dynamic Sampling 判定和补齐目标 batch 大小完全相同，唯一差异是候选 prompt
的 proposal distribution。

对每个约束类别 `c` 维护训练 rollout 上的通过率指数移动平均：

```text
pass_ema_c(t) = beta * pass_ema_c(t-1) + (1-beta) * batch_pass_c(t)
difficulty_c  = clip(1 - pass_ema_c, 0.1, 0.9)
```

包含多个约束类别的 prompt `i`，难度为其类别 difficulty 的均值。最终采样权重使用均匀分布
与困难度分布混合：

```text
w_i = (1-lambda) * 1 + lambda * normalized_difficulty_i
w_i = clip(w_i, 0.5, 2.0)
```

预注册初值：

```yaml
ema_beta: 0.9
uniform_mixture_lambda: 0.5
min_sampling_weight: 0.5
max_sampling_weight: 2.0
sampling_weight_update_steps: 20
```

设计约束：

- EMA 只由训练集 rollout 更新，不读取 Multi-IF dev/test 结果。
- 初始所有类别均匀采样。
- 权重上下限防止训练分布坍缩到少数困难类别。
- CA 权重每 20 optimizer steps 更新一次，而不是每步追逐噪声。
- 每个 50-step 窗口记录类别覆盖率；若任一有数据的类别覆盖率为 0，触发告警但不读取 test 调参。
- DAPO 与 CA-DAPO 使用完全相同的基础奖励；不得同时修改 reward 和 sampler。
- 保存每个 step 的类别通过率、EMA、采样权重和实际采样频次，供审计。

2k pool 与 200 steps 仍可能导致困难类别重复采样和冷门类别覆盖不足。因此 R2 定位为单卡
可行性验证；均匀混合、权重裁剪和 20-step 更新只能降低风险，不能证明不会发生采样坍缩。

### 6.5 三组统一训练预算

初始冻结配置如下，只有 smoke test 发现不可运行时才允许统一调整，并在三组 full run 前重新冻结：

```yaml
seed: 42
num_generations: 4
max_prompt_length: 512
max_completion_length: 512
temperature: 0.8
top_p: 0.95
max_steps: 200
learning_rate: 1.0e-6
gradient_checkpointing: true
peft: true
lora_r: 16
lora_alpha: 32
target: all-linear
```

三组必须使用相同的有效 prompt 数、最大 rollout token 预算和 checkpoint 保存间隔。若 Dynamic
Sampling 导致补采样，报告实际生成 token 总量；不能只用 optimizer step 掩盖额外计算成本。

---

## 7. vLLM 与评测协议

### 7.1 vLLM 的定位

vLLM 用于：

- 4090D 上的批量教师生成。
- R0/R1/R2 的同框架最终评测。

在线 RL rollout 默认使用 TRL/HF 原生生成，减少单卡同时维护训练模型和 vLLM engine 的环境、
显存与调试风险。只有 HF rollout 的 5-step 实测表明预算无法满足，且独立 vLLM rollout smoke
已在 4 小时环境预算内通过时，才允许统一切换 R0/R1/R2 的 rollout backend。

本机 RTX 3070 Ti Laptop 8GB 不作为 vLLM 正式环境：vLLM 不支持 Windows 原生部署，WSL2
也不会增加显存。正式生成和评测均在 AutoDL Linux 上完成，本机只用于保存、检查和汇总 JSONL。

### 7.2 环境隔离

不得在已经能运行 S4-HF 的环境中直接升级 Torch。S4-HF 完成并备份结果后，先记录：

```bash
nvidia-smi
python -V
pip show torch transformers peft bitsandbytes
```

随后创建独立 RLVR/vLLM 环境，锁定并记录：

```text
Python
CUDA runtime
torch
transformers
peft
trl 或统一 trainer 基础库
vllm
flash-attn（若使用）
```

版本只有在 2 条数据的前向、rollout、反向和 adapter 保存/重载全部通过后才能冻结。

### 7.3 vLLM pilot

先在 failure-dev 固定前 20 条上运行 S4 或 T1 adapter，验收：

- 24GB 显存内成功加载 base + LoRA。
- Qwen3 reasoning parser 能稳定返回 `reasoning_content` 和 `content`。
- 三轮历史拼接正确，历史只保留 assistant 最终 content，不重复注入隐藏 reasoning。
- `temperature=0`、`top_p=1`、`max_new_tokens=512`。
- 输出无缺行，评分器无异常。
- 记录 tokens/s、峰值显存和预计 374 条总时长。

不要求 vLLM 与 HF 逐 token 完全一致；组 2 的三组最终模型全部使用同一 vLLM 环境即可。

### 7.4 最终 test：运行两次或三次

P1 至少评测 R0/R1；若 P2 门禁通过并完成 R2，则增加第三次。所有将参加最终 test 的模型及其
配置必须先冻结，再依次运行：

```text
R0-GRPO
R1-DAPO
R2-CA-DAPO
```

统一生成设置：

```yaml
backend: vllm
enable_thinking: true
reasoning_parser: qwen3
temperature: 0
top_p: 1
max_new_tokens: 512
multi_turn_history: final_content_only
```

评分器只检查最终 content。每组保存：

```text
reports/eval_rlvr/{ID}_multi_if_test_vllm.jsonl
reports/eval_rlvr/{ID}_multi_if_test_vllm_summary.json
```

summary 至少包含：

- test 数据 SHA256、行数和唯一 ID 数。
- turn1/2/3 四个官方子指标及 official overall average。
- `primary_score`。
- reasoning、answer、total token 均值与分位数。
- 平均/总生成时间、tokens/s、峰值显存。
- 空回答、解析失败、截断和约束 checker 异常计数。

### 7.5 固定 50 条中文质量护栏

Multi-IF 的程序化 checker 适合衡量可验证约束，但无法完整衡量回答是否流畅、相关、简洁，也无法
阻止模型通过重复关键词等方式获得较高约束分。因此，在 R0 开始前额外冻结一份 50 条中文通用
instruction 质量护栏：

```text
data/eval/chinese_quality_guardrail_50.jsonl
reports/manifests/chinese_quality_guardrail_manifest.json
```

这 50 条不得来自 Multi-IF dev/test，也不得与 T1/RLVR 训练 prompt 精确重合。它不是新的主 benchmark，
不用于调参、决定 P2 或替代 untouched Multi-IF test；它只负责暴露 checker 过拟合和可读性退化。

R0/R1/R2 的最终评测作业在同一冻结模型、同一 vLLM 环境中顺带生成这 50 条，不额外启动模型评测
作业。所有模型输出完成后再统一审阅，记录：

- 空回答、解析失败、达到 token 上限和明显重复的比例。
- reasoning、answer 和总 token 长度分布。
- 固定人工 rubric 下的相关性、流畅性、完整性和无冗余情况。
- 相比模型之间是否出现明显帮助性或可读性回退。

质量护栏结果只作为安全解释和退化警报，不能与 Multi-IF 主分数加权成新的复合主指标，也不能因
某组护栏表现不理想而隐藏该组。

组 1 的 HF 表和组 2 的 vLLM 表分开报告，不做跨表绝对分数推断。

---

## 8. 4090D 24GB 与人民币 50 元预算

### 8.1 已核实租卡信息

用户提供的 AutoDL 页面显示：

```text
GPU：RTX 4090D / 24GB
价格：人民币 1.88 元/小时
CPU：18 核 AMD EPYC 9754
内存：60GB
驱动：560.35.03
CUDA：<= 12.6
```

人民币 50 元最多约为：

```text
50 / 1.88 = 26.6 GPU 小时
```

预算适用于 S4-HF 完成后的新增算法阶段。设置 24 小时软上限，保留约 2.6 小时故障余量。

### 8.2 预算分配

| 阶段 | 目标上限 | 预计费用上限 |
|---|---:|---:|
| 环境、TRL/HF、vLLM 与 smoke | 4.0 h | 7.52 元 |
| 2k 教师生成与审计 | 2.5 h | 4.70 元 |
| T1 cold-start | 1.0 h | 1.88 元 |
| R0 GRPO（P1） | 3.5 h | 6.58 元 |
| R1 DAPO-style（P1） | 4.0 h | 7.52 元 |
| R2 CA-DAPO（P2） | 4.0 h | 7.52 元 |
| 两至三组 vLLM 最终评测 | 2.5 h | 4.70 元 |
| 下载、检查和故障余量 | 2.5 h | 4.70 元 |
| 合计计划 | 24.0 h | 45.12 元 |

以上是估计，不是保证。每个 full run 前必须用 5-step smoke 的实测 tokens/s 推算时长。

### 8.3 硬停止条件

- 环境安装与兼容问题达到 4 小时仍未跑通最小前向/反向：停止租卡，先离线修代码。
- Qwen3-4B 的 5-step RL smoke OOM，或推算单组超过 5 小时：三组统一改为 1.7B。
- 不能只让某一组降模型、减步数或缩短生成长度。
- 租卡累计达到 24 小时：停止新的 full run，先下载全部已有产物。
- 任何脚本未通过 resume、保存与重载测试：不得开始 full run。
- P2 门禁：R0/R1 均无 NaN/OOM、可恢复训练、DAPO 机制遥测完整、test 尚未打开，且剩余预算
  至少 6.5 小时；门禁不读取 R0/R1 的 test 分数。

---

## 9. 计划新增文件与统一接口

以下文件目前尚不存在。专家审核通过后再创建：

| 文件 | 作用 |
|---|---|
| `configs/rlvr.yaml` | 冻结数据、模型、reward、算法和预算参数 |
| `scripts/split_multi_if_dev.py` | 固定拆分 dev/test 并生成 manifest |
| `scripts/build_constraint_rlvr_data.py` | 构造独立可验证约束训练集 |
| `scripts/sample_thinking_data.py` | 生成并过滤 T1 thinking 轨迹 |
| `scripts/compress_thinking_da_cotd.py` | 按训练样本难度压缩 T1 reasoning_content |
| `src/build_distill_dataset.py` | 用 Qwen3 template 构建 T1 cache |
| `src/train_distill.py` | 训练 T1 cold-start adapter |
| `src/rlvr_rewards.py` | reasoning 解析、约束 reward 与长度统计 |
| `src/constraint_sampler.py` | CA-DAPO EMA 与采样权重实现 |
| `src/train_rlvr.py` | 统一的 GRPO/DAPO/CA-DAPO 训练入口 |
| `src/evaluate_multi_if_vllm.py` | vLLM 多轮生成、resume 与官方评分 |
| `tests/test_rlvr_algorithms.py` | GRPO/DAPO 公式和 CA sampler 单元测试 |
| `data/eval/chinese_quality_guardrail_50.jsonl` | 固定中文质量护栏 prompt |
| `reports/final_comparison.md` | 组 1 对比已完成，后续追加组 2 分表与最终结论 |

计划统一 CLI（实现前接口，可由专家审核；当前不可运行）：

```bash
python scripts/split_multi_if_dev.py \
  --input data/eval/multi_if_zh.csv \
  --seed 42 \
  --dev-size 80

python scripts/build_constraint_rlvr_data.py \
  --size 2100 \
  --seed 42

python scripts/sample_thinking_data.py \
  --teacher Qwen/Qwen3-4B-Instruct \
  --input data/rlvr/constraint_train_2000.jsonl \
  --output data/distill/t1_thinking_accepted.jsonl \
  --resume

python src/train_distill.py \
  --experiment T1 \
  --config configs/rlvr.yaml \
  --resume

python src/train_rlvr.py --algorithm grpo --experiment R0 --config configs/rlvr.yaml --resume
python src/train_rlvr.py --algorithm dapo --experiment R1 --config configs/rlvr.yaml --resume
python src/train_rlvr.py --algorithm ca-dapo --experiment R2 --config configs/rlvr.yaml --resume

python src/evaluate_multi_if_vllm.py \
  --experiment-id R0-GRPO \
  --data data/eval/multi_if_test.csv \
  --adapter outputs/rlvr/R0/final_adapter \
  --resume
```

R1/R2 使用相同评测命令，只替换实验 ID 与 adapter 路径。

---

## 10. 分阶段执行与验收门禁

### 阶段 A：闭合组 1

1. S4-HF 跑完。已完成。
2. 验证 454 行、454 唯一 ID、冻结 SHA256 和 decoding 参数。已完成。
3. 下载 S4 JSONL、summary、adapter 训练结果。评测结果已在本机；adapter 已按 Git LFS 策略保存。
4. 填写 B0/S1/S2/S3/S4 的 HF 表。已完成。

产物：`reports/final_comparison.md` 表 A。  
门禁：G-S4 已通过。

### 阶段 B：专家审核与实现前冻结

1. 专家审核本文的数据泄漏、算法归因、算力和评测方案。第一轮已完成。
2. 修订为 v5.3；下一步先实现并验收固定 dev/test 切分，再记录方案与 manifest SHA256。
3. 数据边界通过后才构造训练数据；不提前实现 RL trainer。

门禁：G-REVIEW。未通过不租 4090D。

### 阶段 C：数据与单元测试

1. 实现 dev/test 切分和 manifest。
2. 构造 2.1k 可验证约束数据。
3. 完成重合检查和人工抽查。
4. 为 reward、DAPO loss、动态采样与 CA 权重写单元测试。

门禁：G-DATA + 所有 CPU 单元测试通过。

### 阶段 D：4090D 环境和最小 smoke

1. 建独立环境并锁定版本。
2. 2 条数据跑通 teacher generation。
3. 2 条数据跑通 T1 前向/反向和 adapter 重载。
4. R0/R1 各跑 2-5 step；R2 只跑 sampler 单测和 2-step 集成测试。
5. vLLM 跑 20 条 failure-dev pilot。

门禁：G-ENV。累计不超过 4 小时。

### 阶段 E：T1

1. 100 条教师 pilot，记录速度、接受率和截断率。
2. 估算 2k 总时长；预计超过 3 小时则降低重采样次数，不降低审计标准。
3. 批量生成 2k accepted 原始 thinking 数据。
4. 运行 DA-CoTD-inspired 压缩并通过压缩审计门禁。
5. 训练 T1，评测 synthetic validation 和 failure-dev。

门禁：G-T1。

### 阶段 F：P1/P2 RL full run

先完成 P1 的 R0 -> R1。R1 完成后，在不打开 test 的前提下检查 P2 客观门禁；通过才运行 R2，
否则停止在 P1。每组开始前记录累计租卡时长；每组完成立即下载：

```text
final_adapter/
trainer_state.json
all_results.json
resolved_config.yaml
reward_curve.jsonl
sampling_stats.jsonl（R2 必须）
运行日志
```

三组任何超参数变化必须在 R0 full run 前统一冻结。R0 开始后不得只为表现差的一组调参。

门禁：G-RL。P1 必须完成；P2 是否完成按预注册门禁记录。所有待评测模型和配置全部冻结。

### 阶段 G：两至三次最终 test

1. 首次打开 untouched test。
2. 使用同一 vLLM 配置评测 R0/R1；若 P2 已完成，再评测 R2。
3. 每组结果完成即校验行数、唯一 ID、SHA256 和错误计数。
4. 不根据前一组 test 分数修改后一组模型或生成参数。

门禁：G-TEST。

### 阶段 H：汇总

表 A：B0/S1/S2/S3/S4，全部 HF。  
表 B：R0-GRPO/R1-DAPO-style，以及门禁允许时的 R2-CA-DAPO，全部 vLLM。  
表 C：各组训练时间、rollout tokens、峰值显存、评测吞吐和费用。

必须同时报告：

- 成功假设和失败假设。
- 每种约束类别的通过率变化。
- thinking/answer token 长度变化。
- DAPO Dynamic Sampling 丢弃率与 CA-DAPO 实际采样分布。
- 预算使用和异常恢复记录。

---

## 11. 自检与已知局限

### 11.1 当前方案已修正的问题

- test 不再在 T1 后提前查看；P1/P2 决策和所有待评测模型冻结后才一次性评测。
- 不再执行“CA-DAPO 输了就换报告主线”的结果驱动选择。
- DAPO 与 CA-DAPO 只改变 sampler，基础 reward 与其余算法机制一致。
- 训练数据不使用 Multi-IF dev/test 原题，只使用约束 schema。
- thinking 与最终 content 分开，约束 checker 只评分最终 answer。
- vLLM 只在组 2 内比较，组 1 HF 结果保持独立。
- 4B OOM 时三组统一降为 1.7B，不允许混用模型规模。
- 预算有明确分配和 24 小时停止线。
- T1 与 RL 明确使用同一 2k prompt 池，属于 warmed-up on-policy 设定。
- HF 是默认 RL rollout backend，vLLM 只承担教师批量生成和最终评测。
- DA-CoTD-inspired 压缩只作为共同 T1 数据预处理，不单独增加第四个最终模型。

### 11.2 尚未被证明的事项

1. 4090D 24GB 是否能同时容纳 Qwen3-4B QLoRA actor、rollout 和所需缓存，必须 smoke test。
2. 当前环境中 TRL/verl/vLLM 的可用版本和 adapter 接口尚未锁定。
3. 单卡实现是否与 DAPO 论文公式完全一致，必须靠单元测试和配置审计确认。
4. CA-DAPO 的约束失败率采样可能导致困难类别过拟合或简单类别遗忘。
5. 只有一个 benchmark 和一个 seed，不能形成统计显著或通用算法结论。
6. 时间估计来自硬件和任务规模推算，实际以 5-step smoke 为准。
7. T1 不跑最终 test，节省一次评测，但因此不能直接量化 RL 相对 cold-start 的 test 增益。
8. DAPO 的 Overlong Shaping 与 Token-Level Loss 在当前短输出任务上可能很少激活。
9. R2 是 2k prompt/200-step 预算下的可行性验证，不能排除类别重复采样与覆盖不足。
10. 没有未压缩长 CoT 的独立 T1 对照，不能声称 DA-CoTD-inspired 优于 long-CoT；只能报告压缩
    比例、训练成本和共同 T1 的 dev 行为。

### 11.3 项目可以诚实声称的贡献

若完整执行，可以声称：

- 建立了一套可审计的中文 SFT 数据治理和受控消融流程。
- 在相同初始化与预算下完成 GRPO 与单卡 DAPO-style 对比，并在门禁允许时验证 CA-DAPO。
- 使用 DA-CoTD-inspired 难度感知 reasoning compression 降低 T1 冷启动轨迹长度；不声称其
  单独优于未压缩 long-CoT。
- 将程序化多约束验证器用于中文指令遵循 RLVR。
- 提出并实现一个约束类别失败率驱动的 DAPO 采样变体，并如实验证其正负结果。
- 建立 vLLM 多轮评测、断点续跑、结果 manifest 和费用审计链路。

不应声称：

- CA-DAPO 在通用任务上优于 DAPO。
- 已达到论文级算法创新或统计证明。
- vLLM 与 HF 分数可以直接横比。

---

## 12. 第一轮专家审核意见与处理

| 专家意见 | 处理决定 |
|---|---|
| 单卡手写 DAPO 工程量被低估 | 接受。R0 使用 TRL 原生；R1 改称 DAPO-style，并逐项标注公式一致性 |
| CA sampler 作用位置不明确 | 接受。固定作用于 rollout 前 proposal distribution，补采样沿用同一分布 |
| 2k pool/200 steps 有采样坍缩风险 | 接受风险但不视为必然；增加均匀混合、权重裁剪、20-step 更新与覆盖遥测，R2 降为 P2 |
| 两项 DAPO 机制在短输出上可能空转 | 接受。新增机制激活率与长度分布报告 |
| 环境 2h 太乐观、vLLM rollout 风险高 | 接受。环境预算改为 4h，RL 默认 HF rollout，vLLM 主要用于最终评测 |
| T1 与 RL prompt 池不明确 | 接受。明确同一 2k prompt 池的 warmed-up on-policy 设定 |
| Thinking cold-start 可能无差别模仿冗余长 CoT | 接受。T1 增加 DA-CoTD-inspired 难度感知压缩；不额外增加最终模型分支 |

执行优先级正式冻结为：

```text
P0：S4-HF 闭合组 1（已完成）
P1：T1 + R0(GRPO) + R1(DAPO-style)
P2：R2(CA-DAPO)，只在 P1 技术稳定、test 未打开且预算门禁通过后执行
```

---

## 13. 请专家继续审核的问题

1. 用约束类型训练通过率 EMA 调整 prompt 采样，是否与已有 curriculum/hard-example mining 方法
   高度重合，项目中应如何准确命名。
2. R1 对 DAPO 四项机制的公式一致、工程近似和未激活标记是否足够透明。
3. DAPO vs CA-DAPO 是否真正做到只改变 sampler，是否仍有隐藏的计算预算差异。
4. `r_core = 0.7 * instruction + 0.3 * prompt` 是否合理，是否需要在不接触 test 的前提下调整。
5. CA 权重公式、EMA、上下限和均匀混合系数是否会引入明显偏差。
6. 2k 合成约束训练集是否足以支持 200-step RL，类别覆盖是否需要分层采样。
7. thinking 只隐藏评分、但计入 token 成本的协议是否合理。
8. 4B->1.7B 的统一降级门禁是否足够明确。
9. 单 seed、三次最终 test 的结论应该限制到什么程度。
10. 人民币 50 元预算下，哪些实验或指标最应该优先保留。

---

## 14. 最终项目叙事

推荐项目标题：

> **Qwen3 中文后训练与约束感知 RLVR：从可审计 SFT 消融到 GRPO、DAPO 和 CA-DAPO 对比**

面试中的一句话概括：

> 我先用同数据、同预算的实验确定 LoRA target、数据质量和数据规模影响，再使用独立合成的
> 可验证约束数据进行 warmed-up on-policy Thinking cold-start，在完全相同初始化下比较
> GRPO、单卡 DAPO-style 与一个
> 约束失败率驱动的 CA-DAPO 采样变体；最终 test 全程冻结，正负结果、训练成本和推理效率
> 全部可审计。
