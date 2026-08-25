# 第二部分实验代码学习导览

> 目标：只看与第二部分直接相关的代码，并按真实执行顺序理解输入、处理和输出。
> 当前状态：T1 训练产物完整，但自由生成结构门禁失败，只作负结果保留。R0/R1/R2 已改为从同一 `Qwen3-4B + seed=42 新 LoRA` 出发；73 个 RLVR 定向测试与 preflight 通过，新路线 GPU smoke 尚未执行。

## 1. 第一部分与第二部分的边界

第一部分是已经完成的 SFT 消融：

```text
B0 基座
S1 clean-2k + attention LoRA
S2 clean-2k + all-linear LoRA
S3 raw-2k + all-linear LoRA
S4 clean-10k + all-linear LoRA
```

第一部分主要代码是 `src/build_sft_datasets.py`、`src/preprocess_sft.py`、`src/train_sft.py` 和
`src/evaluate_multi_if.py`。学习第二部分时可以先不看这些文件。

第二部分的完整数据流是：

```text
Multi-IF 454 条
  -> 固定切为 dev 80 + untouched test 374

独立 SFT 数据源
  -> 构造 RLVR train 2000 + validation 100
  -> 教师 Qwen3-4B 为 2000 条 prompt 生成 Thinking
  -> 官方 checker 筛出 1461 条 accepted
  -> tokenize 后保留 1340 条
  -> 训练 T1，但自由生成结构门禁失败，不进入 RL
  -> 从同一 Qwen3-4B 和同种子新 LoRA 分别训练 R0 / R1 / R2
  -> 三组全部冻结后统一用 vLLM 评测
```

## 2. 建议阅读顺序

### 第一步：先看 T1，连接你已经会的 QLoRA

#### `scripts/sample_thinking_data.py`

作用：让教师模型回答 2000 条 RLVR prompt，生成“思考过程 + 最终回答”。

核心调用链：

```text
main
  -> build_model：4-bit 加载 Qwen3-4B 教师
  -> generate_once：套 Qwen3 chat template 并调用 model.generate
  -> parse_thinking_continuation：按 </think> 拆出 thinking 和最终回答
  -> check_constraints：只检查最终回答
  -> accepted / rejected / raw 三类 JSONL
```

输入：`data/rlvr/constraint_train_2000.jsonl`。

输出：

- `data/distill/t1_thinking_raw.jsonl`：每次生成的原始审计记录。
- `data/distill/t1_thinking_accepted.jsonl`：最终回答通过所有约束的数据。
- `data/distill/t1_thinking_rejected.jsonl`：重试后仍失败或格式损坏的数据。

这一步不是训练，只是离线生成 T1 的监督答案。

#### `src/build_distill_dataset.py`

作用：把 accepted 教师轨迹变成 Hugging Face Dataset 缓存。

最重要的函数是 `tokenize_row`：

```text
messages
  -> apply_chat_template 得到完整 input_ids
  -> messages[:-1] 得到 prompt token 长度
  -> labels = prompt 部分全 -100 + assistant 部分真实 token id
```

因此 T1 的交叉熵只学习 assistant 的 thinking 和最终回答，不学习 user prompt。完整序列超过
1024 时整条过滤，不从中间截断推理链。

#### `src/train_distill.py`

作用：用你熟悉的 QLoRA + Hugging Face Trainer 训练 T1。

主流程与第一部分很接近：

```text
读取 tokenized cache
  -> 4-bit 加载 Qwen3-4B-Base
  -> prepare_model_for_kbit_training
  -> get_peft_model 创建 all-linear LoRA
  -> TrainingArguments
  -> Trainer
  -> trainer.train / evaluate
  -> 保存 final_adapter
```

这段训练确实跑完，但它只证明教师强制下的 token 预测 loss 下降。后续
`scripts/audit_t1_generation.py` 证明 `Base + T1 LoRA` 不能稳定生成 `<think>`、`</think>`
和 `<|im_end|>`，因此不能作为 RL 起点。这是一个保留的负结果，不是“白跑”，
但也不得包装成成功 cold-start。

### 第二步：理解 R0/R1/R2 共用的数据和奖励

#### `configs/rlvr.yaml`

先看 `rlvr:` 段。它冻结三组共同参数，并用四个算法开关描述差异：

| 实验 | 机制 |
|---|---|
| R0 | 标准 GRPO；三组共用健康门禁只监控、不改样本 |
| R1 | R0 + Clip-Higher + Token-Level Loss + Soft Overlong |
| R2 | R1 + Constraint-Aware sampler |

共同健康门禁不是算法增量：它不重生成、不丢弃回答，也不改变 reward。连续 8 个 generation batch
的 advantage 全为 0 时只写入诊断并停机，防止付费 GPU 长时间空转。

#### `src/rlvr_contract.py`

作用：保护实验归因和数据边界。

关键函数：

- `validate_algorithm_matrix`：禁止 R1/R2 除 sampler 外出现其他差异。
- `validate_training_row`：检查一条 RLVR JSONL 的 messages、约束 ID、kwargs 和类别。
- `to_trl_rows`：转换为 TRL 的 conversational dataset 列。
- `RewardBatchAdapter.__call__`：接收 TRL 生成的一批 completions，返回一批浮点 reward。

它不训练模型，相当于训练器与数据/奖励之间的接口层。

#### `src/rlvr_rewards.py`

作用：实现 R0/R1/R2 的奖励和公式。

一条回答的奖励流：

```text
原始 completion
  -> parse_reasoning_answer 拆开 thinking 与 answer
  -> 官方 checker 逐条检查 answer 的约束
  -> r_instruction = 通过约束数 / 约束总数
  -> r_prompt = 是否全部通过
  -> r_core = 0.7 * r_instruction + 0.3 * r_prompt
  -> R1/R2 再加 soft overlong penalty
```

方法对应关系：

- `group_advantages`：R0 的 GRPO 组内相对优势。
- `clip_higher`：R1/R2 的 DAPO 非对称裁剪。
- `token_level_policy_loss`：R1/R2 按有效生成 token 平均 loss。
- `group_has_zero_reward_variance`：检测同一题的一组回答是否完全没有相对优势。
- `soft_overlong_penalty`：接近 1024 token 上限时线性扣分。

RL 直接使用原生 thinking 结构正常的 `Qwen3-4B`，rollout 启用 thinking，三组共用
1024 token 上限。本机采样审计已观察到 383 token 自然结束；EOS 是 `<|im_end|>`
（151645），pad 是 `<|endoftext|>`（151643）。奖励函数必须从原始 token id 解码，
否则 TRL 的 `skip_special_tokens=True` 会删除 `</think>` 并造成假的解析失败。

当前这些函数已经通过 CPU 单元测试，但仍需接入锁定版本的 TRL GPU 训练循环。

### 第三步：最后看 R2 的创新点

#### `src/constraint_sampler.py`

作用：R2 根据“哪些约束类别最近仍在进步”调整抽题概率。

每个类别维护：

```text
slow EMA：长期通过率
fast EMA：近期通过率
progress = max(0, fast - slow)
frontier = 4 * slow * (1 - slow)
learnability = progress * frontier
```

随后：

```text
类别 learnability
  -> prompt 内多个类别取平均
  -> min-max 归一化
  -> 与 50% 均匀分布混合
  -> 得到每条 prompt 的采样权重
```

关键函数：

- `update_batch_pass_rates`：每批 rollout 更新类别统计。
- `sample_weights`：计算 2000 条 prompt 的候选权重。
- `weights_for_step`：只在每 20 optimizer steps 更新实际权重。
- `state_dict/load_state_dict`：保存和恢复 sampler，支持断点续训。

R1 使用均匀采样，R2 使用这里的权重；其他 reward、loss、数据和预算必须完全相同。

### 第四步：看统一训练入口

#### `src/train_rlvr.py`

它是类似 `train_sft.py` 的唯一入口。目前已完成：

```text
读取配置
  -> 检查 R0/R1/R2 开关
  -> 检查 2000 条训练数据
  -> 检查初始化模式必须是 Qwen3-4B 上的新 LoRA
  -> 检查输出目录和 resume
  -> 输出 preflight manifest
  -> 锁定并核对 TRL 1.9.2 接口
  -> 4-bit 加载 Qwen3-4B
  -> seed=42 创建 all-linear LoRA，记录初始权重哈希
  -> 创建 GRPOConfig / GRPOTrainer
  -> rollout
  -> reward
  -> policy loss 和 backward
  -> R1 的 DAPO 机制绑定
  -> R2 的 sampler 绑定
  -> checkpoint / final_adapter
```

其中 R0 使用 TRL 原生 GRPO；R1 使用 TRL 原生 `loss_type=dapo`、Clip-Higher 和截断 mask，
再叠加项目 Soft Overlong；R2 与 R1 参数完全相同，只替换 `_get_train_sampler`。

R0/R1/R2 都共享同一项运行时健康门禁：每个原始 generation batch 都照单进入 GRPO 计算，不按
reward 结果筛选或重生成。零方差组自然得到 0 advantage；若连续 8 个 batch 全部没有学习信号，
脚本生成 `zero_variance_abort.json` 并停止。这样既不人为制造 reward 差异，也避免 200 步持续空转。
当前没有实现 DAPO 论文的按组丢弃补采样，因此 R1 必须继续称为 `DAPO-style`。

### 第五步：理解 smoke 题目和验收门禁

`configs/rlvr.yaml` 中的 `smoke_prompt_ids` 只在命令带 `--smoke` 时生效。smoke sampler
关闭 shuffle，2-step 运行固定使用前两题，以同时检验 Qwen3-4B 结构与终止、同题四个
回答是否产生 advantage，以及 LoRA 是否真实更新。这组 ID 只是诊断集，实际生成仍需
通过门禁，不能因为题目被选中就预设 smoke 会成功。
本机 Qwen3-4B、seed=42 的四回答探针中，`rlvr-0062` 得到
`[1.0, 0.4667, 0.4667, 0.4667]`，解析率和自然终止率均为 1.0，因此固定为第一道
smoke 题。该选择仅用于证明集成链路可以产生梯度，不改正式训练池或采样分布。

正式 R0/R1/R2 不读取这组 ID，仍从同一份完整 2000 条训练池采样。smoke 结束前还会检查：

```text
checker_errors == 0
至少一个 prompt 组的 reward 方差 > 0
所有回答都能拆成 thinking + answer
不是所有回答都达到 1024 token 上限
LoRA 可训练参数哈希发生变化
```

任意一项失败，脚本直接报错，不会把“跑完两步但梯度为零”当成成功。

## 3. 辅助文件：知道用途即可

- `scripts/split_multi_if_dev.py`：冻结 dev/test，防止用最终 test 调参。
- `scripts/build_constraint_rlvr_data.py`：从独立数据构造 2000+100 条约束 prompt。
- `scripts/probe_rlvr_environment.py`：读取 AutoDL 的 TRL 版本、类签名和方法哈希。
- `tests/test_rlvr_algorithms.py`：验证奖励公式和 sampler 数值。
- `tests/test_rlvr_contract.py`：验证三组归因、数据列、reward adapter 和 preflight。
- `tests/test_rlvr_config.py`：验证 YAML 冻结参数。

## 4. 不属于当前主线

- `scripts/compress_thinking_da_cotd.py`：已经退出正式数据流的压缩原型，不需要学习或运行。
- `tests/test_compress_thinking_da_cotd.py`：只保留原型回归证据。
- `src/train_sft.py` 与 `src/build_sft_datasets.py`：属于第一部分 SFT。
- `reports/eval/B0/S1/S2/S3/S4*`：属于第一部分结果。
- `third_party/Multi-IF/*`：第三方官方 checker，先把它当库使用，不需要逐行学习。

## 5. 现在最适合你的阅读路线

```text
src/train_distill.py
  -> 对照你已经会的 train_sft.py，确认 T1 仍是 QLoRA SFT

src/rlvr_rewards.py
  -> 理解“没有标准答案时，怎么把生成结果变成 reward”

src/rlvr_contract.py
  -> 理解 reward 函数怎样接收 TRL 的一批生成结果

src/constraint_sampler.py
  -> 理解 R2 只改变了抽哪道题

src/train_rlvr.py
  -> 从 build_preflight 看输入门禁
  -> 从 build_grpo_kwargs 对照 R0/R1/R2 参数差异
  -> 从 run_training 看 Qwen3-4B、新 LoRA、reward、trainer 和 final_adapter 的完整串联
```

学习时不要从 `scripts/build_constraint_rlvr_data.py` 的第一行开始硬啃。它是数据工程脚本，代码长，
但不是理解 GRPO/DAPO 的前置条件。
