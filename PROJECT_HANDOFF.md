# Qwen3 中文后训练项目交接说明

> 更新时间：2026-08-08
>
> 用途：新开 Codex 对话时，让新对话先阅读本文件，再继续项目。本文记录已确认事实、当前状态和下一步，不代表所有任务均已完成。

## 1. 项目目标

项目路径：

```text
C:\Users\A\Desktop\ai\TRANSFORMERS\qwen_post_training_lab
```

基于 `Qwen/Qwen3-4B-Base` 完成中文通用指令后训练工程闭环：

1. Infinity-Instruct 数据治理与可审计清洗。
2. 4-bit QLoRA SFT。
3. LoRA target、数据质量、数据规模三组控制变量实验。
4. Multi-IF 中文多轮指令跟随评测与错误分析。
5. SFT 闭环后，使用共同 Thinking cold-start 初始化，做 GRPO、DAPO-style 与条件执行的 CA-DAPO 受控 RLVR 对比。
6. 整理 README、GitHub、简历和面试深挖材料。

项目定位是完整、可复现的后训练工程研究，不包装成论文级通用算法创新。

## 2. 数据设计

数据来源：ModelScope `AI-ModelScope/Infinity-Instruct` 的 `7M` 子集。

冻结的原始训练文件：

```text
data/sft/full_clean_10000.jsonl
data/sft/ablation_clean_2000.jsonl
data/sft/ablation_raw_2000.jsonl
```

已确认的数据关系：

- `clean 2k` 是 `full clean 10k` 的子集。
- `raw 2k` 与 clean/full 的问题无重叠。
- clean/raw 的任务类别和字符长度桶计数一致，但不是同题答案配对。
- 自动治理包括：中文筛选、一问一答结构抽取、schema/非空检查、NFKC 精确去重、客观结构错误过滤、软风险评分和分层采样。
- 不确定的知识、医学、代码等领域不能整类删除，只能标风险或进入抽查。
- 100 条分层人工审核结果为 `99 pass / 1 major`。
- 对外只能写“自动治理 1 万条并人工抽查 100 条”，不能写“人工清洗了 1 万条”。

tokenized cache：

```text
data/cache/sft_qwen3_4b_1024
```

预处理后的规模：

| split | rows |
|---|---:|
| ablation_clean_train | 1815 |
| ablation_clean_validation | 93 |
| ablation_raw_train | 1815 |
| ablation_raw_validation | 93 |
| full_clean_train | 9249 |
| full_clean_validation | 485 |

预处理规则：仅 assistant 回答参与 loss，prompt 标签置为 `-100`。

```python
labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
```

保留 assistant 末尾的 `<|im_end|>`。长度超过 `1024` 的样本直接过滤，不截断；clean/raw 任意一侧超长时成对移除，从而保持两组分层计数一致。

## 3. 冻结实验矩阵

| ID | 数据 | LoRA target | Multi-IF turn 1/2/3 | 实验目的 |
|---|---|---|---|---|
| B0 | 无训练数据 | 无 | 0.404528 / 0.204584 / 0.160707 | Base 基线 |
| S1 | clean 2k | `q/k/v/o` | 0.469344 / 0.340690 / 0.231513 | attention target 方案 |
| S2 | 同一 clean 2k | `all-linear` | 0.501450 / 0.348407 / 0.235692 | target 消融 |
| S3 | raw 2k | `all-linear` | 0.453707 / 0.306988 / 0.203443 | 数据质量消融 |
| S4 | full clean 10k | `all-linear` | 0.449160 / 0.324287 / 0.229305 | 数据规模和最终 SFT |

重要约束：

- S1 和 S2 使用同一数据、step 数和公共超参数，只改变 LoRA target。
- S3 固定 S1/S2 的胜出 target，只改变数据质量。
- S4 必须重新从 Base 开始训练，不能续训 S1/S2。
- S1/S2 的 eval loss 只能作为诊断，最终用 Multi-IF 选胜者。

组 1 已成立结论：

- S2 三轮均高于 S1：当前配置下 all-linear 优于 attention target。
- S2 三轮均高于 S3：相同 2k 规模和预算下 clean 优于 raw。
- S4 三轮均高于 B0：完整 SFT 相比 Base 有收益。
- S4 三轮均低于 S2：clean 从 2k 增加到 10k 没有改善 Multi-IF，这是必须保留的负结果。

## 4. QLoRA 配置

```text
4-bit NF4
double quantization
r=16
alpha=32
dropout=0.05
learning_rate=2e-4
warmup_ratio=0.05
linear scheduler
micro batch=2
gradient_accumulation=8
effective batch=16
paged_adamw_8bit
gradient checkpointing
```

S1/S2 都固定 `max_steps=200`，约等于 `1.76 epoch`。`max_steps` 会覆盖 epoch 设置。

训练结果：

| 实验 | train loss | eval loss | 时间 |
|---|---:|---:|---:|
| S1 | 1.3296 | 1.4947 | 1188.6 秒 |
| S2 | 1.3057 | 1.4935 | 1444.9 秒 |

两者 eval loss 差异极小，不能据此认定 S2 更好。

adapter：

```text
outputs/sft/S1/final_adapter
outputs/sft/S2/final_adapter
```

## 5. Multi-IF 评测协议

固定设置：

```text
dataset: facebook/Multi-IF Chinese
rows: 454
最多 3 轮
max_new_tokens: 512
do_sample: false
enable_thinking: false
```

评测脚本：`src/evaluate_multi_if.py`。

脚本支持 `--resume`：每完成一条就追加到 JSONL，中断后根据样本 ID 跳过已完成结果。

五组均已完成，结果位于：

```text
reports/eval/{B0,S1,S2,S3,S4}_multi_if_zh.jsonl
reports/eval/{B0,S1,S2,S3,S4}_multi_if_zh_summary.json
```

截至 2026-08-08 的本机核验：五组各 `454/454`，各有 454 个唯一 ID，ID 集合完全一致；每行均有
3 个 turn，无空回答。数据 SHA256 均为：

```text
ce43827ef4b43c6fde34038180ed2deb4795a7aef331e187de5fd0a5da15601c
```

重要局限：S4、S2、S3 的 1362 次生成全部达到 512 token 上限，S1 为 1361/1362，B0 为
946/1362。固定协议下的相对比较仍可使用，但不能宣称输出自然、简洁或没有重复。完整审计见
`reports/final_comparison.md`。

## 6. 下一步执行顺序

组 1 的 P0/G-S4 已闭合。组 2 固定主线为：

```text
Qwen3 Base
  -> T1：2k DA-CoTD-inspired 难度感知 Thinking cold-start
      -> R0：GRPO
      -> R1：单卡 DAPO-style reproduction
      -> R2：CA-DAPO（仅在 P2 预算与稳定性门禁通过时）
```

当前唯一实现阶段是先冻结数据边界，不训练 T1，也不编写 RL trainer：

1. 将 454 条 Multi-IF 按 `seed=42` 固定切成 failure-dev 80 条与 untouched test 374 条。
2. 输出两份 CSV、两份 ID 清单和包含 SHA256/交集检查的 manifest。
3. 确认 dev/test ID 无交集，并保证 T1/RL 训练数据不使用任何 Multi-IF 原题或 ID。
4. 完成专家审核与配置冻结后，才构造独立的 2k RLVR 约束训练数据。

详细方案和预算门禁见 `reports/post_sft_upgrade_plan.md`。不得在模型、脚本和配置全部冻结前查看
untouched test 分数。

## 7. 已解决或易复发的故障

- NLTK 需要 `punkt` 和 `punkt_tab`。
- 泰语规则即使评测中文数据也可能被某些约束调用，需要安装 `python-crfsuite`，导入名是 `pycrfsuite`。
- AutoDL 连接 Hugging Face 可能出现 `Cannot assign requested address`；使用已下载的 ModelScope 本地快照并通过 `--model` 传入。
- “模型来自 ModelScope”与“代码使用 Transformers 加载接口”并不冲突，不要因此擅自改回 Hugging Face 下载。
- 本机离线重跑时，远程模型 ID 仍可能触发网络请求；应确认本地 cache 和离线参数真正生效。
- CMD 不能使用 PowerShell 的 `$env:`、`&` 和反引号续行；先看终端提示符再给命令。
- 远程连接断开不会影响 `screen` 内任务，但普通前台进程可能终止。
- 不要因中断删除 JSONL；使用 `--resume` 接着跑。

## 8. 关键文件

新对话开始时至少检查：

```text
configs/project.yaml
reports/experiment_plan.md
reports/sft_data_audit.md
reports/final_comparison.md
reports/post_sft_upgrade_plan.md
reports/eval/*_summary.json
src/train_sft.py
src/evaluate_multi_if.py
```

学习和面试材料：

```text
Qwen3后训练项目_面试深挖与代码阅读手册.docx
```

## 9. 用户履历背景

以下信息来自现有个人简历，仅用于让新对话理解用户基础，不得自行补全或夸大：

- 福州大学（211）本科，2022.09-2026.07，简历所列专业为电子工程信息，GPA `3.19/4.00`。
- 英语：CET-4 `562`，CET-6 `580`。
- 原专业基础偏电子、通信、信号处理和嵌入式，并非计算机科班的大模型方向。
- 已有项目包括 ViT-CNN 手势识别、FPGA 载波同步、MATLAB 图像传输系统，均在简历中写为项目负责人。
- 竞赛包括全国大学生电子设计竞赛省一等奖、美国大学生数学建模竞赛 S 奖；建模成果曾发表于《福建电脑》。
- 履历中的传统工程项目是已有基础，当前新增 Qwen3 后训练项目的目的，是形成与大模型算法岗位直接相关、能被深入追问的核心项目。
- 交接文件不保存电话、邮箱、出生年月等与代码协作无关的个人隐私。

## 10. 已学 Hugging Face 范式与代码习惯

用户此前主要跟随课程学习 Hugging Face，熟悉的是直接、线性的训练脚本。新对话解释本项目时，应优先映射回下面这条主线：

```text
AutoTokenizer.from_pretrained
-> 定义 process_function(example)
-> dataset.map(process_function, batched=True)
-> DataCollator 动态 padding/组 batch
-> 加载 AutoModel
-> TrainingArguments
-> Trainer
-> trainer.train()/evaluate()/predict()
```

用户已经接触并应继续巩固的知识：

- `tokenizer(...)` 把原始文本转成 `input_ids`、`attention_mask` 等模型输入。
- `process_function` 是单条或一批样本的预处理函数，`datasets.map` 才把它应用到整个 Dataset，并返回新的 Dataset。
- NER 课程中使用 `word_ids()` 对齐 token 与标签，特殊 token 和不计算损失的位置用 `-100`。
- `max_length`、`truncation=True` 过去常直接写在 tokenizer 调用里；本项目为了审计超长样本，先完整 tokenization，再在外部 `filter` 掉超过 1024 的样本。
- `DataCollator` 不包含 `map`；它在 DataLoader 取出若干已预处理样本后动态 padding 并拼成 batch。batch 大小由 `per_device_*_batch_size` 等参数决定。
- `apply_chat_template` 是对话模型 tokenizer 提供的模板序列化工具：把 `role/content` 消息转换成模型约定的 user/assistant 控制 token，并不负责真正生成回答。
- `add_generation_prompt=True` 只在 prompt 末尾补 assistant 起始模板，提示模型从 assistant 位置续写；它本身不会执行推理。
- `messages[:-1]` 表示除最后一条 assistant 答案外的 prompt；不能固定写 `[:1]`，因为那会错误丢掉多轮历史或 system 消息。
- SFT 中完整序列用作 `input_ids`，prompt 对应的 labels 置为 `-100`，assistant answer 部分 labels 保持 token id，只对回答计算交叉熵。
- 动态 padding 应由 collator 在 batch 阶段完成，不必在 `map` 阶段把所有样本补到 1024。
- 本机曾安装不同 Python/Transformers 环境；涉及 API 参数名时必须检查当前解释器和函数签名，不能凭另一版本经验直接改名。
- 离线推理应同时从本地 checkpoint/cache 加载 tokenizer 和 model，并在适用时使用 `local_files_only=True`，避免本地已有权重却仍访问网络。

教学和代码组织偏好：

- 初次出现的新 API 要解释输入、输出和它在数据流中的位置。
- 先给出与课程代码的一一对应，再解释工程版为什么多了配置、校验、断点续跑和实验编号。
- 保持脚本执行顺序清晰，避免把 argparse、配置读取、模型、参数和 Trainer 混着讲。
- 不能只说“这部分不用掌握”；应区分面试必须掌握、会使用即可、工程防护三层。
- 核心概念讲明白后，改为紧凑的端到端说明和一行可复制命令，不再无限拆成小步骤。
- 修复代码时优先最小改动并验证，不要为了“工业化”突然大规模重构。

## 11. 求职目标与后续路线

当前目标方向：大模型算法工程师/LLM 后训练相关实习，最终目标是拿到 offer，而不是只把教程跑通。

项目能力目标：

1. 能独立讲清数据格式、清洗、tokenization、label mask、动态 padding 和 DataLoader/Trainer 数据流。
2. 能解释 QLoRA 的 4-bit 量化、LoRA rank/alpha/target modules、显存与可训练参数量。
3. 能用控制变量实验回答“为什么这样配”，而不是只展示一条 loss 曲线。
4. 能解释 Multi-IF 的 prompt-level、instruction-level、strict/loose、多轮退化和评测成本。
5. 能进行失败案例分类、局限性分析和可复现实验记录，支撑至少 30 分钟项目深挖。
6. 能把云端训练、本地推理、断点续跑、环境和产物备份讲成完整工程链路。

学习/项目路线：

```text
从零理解 GPT/nanoGPT
-> Hugging Face 常规微调与 Trainer
-> 当前 Qwen3 QLoRA SFT 消融
-> DA-CoTD-inspired Thinking cold-start
-> GRPO / DAPO-style / CA-DAPO 受控 RLVR 对比
-> 后续再单独设计 Agent 项目
```

原则：当前项目先做完整、结果真实、能深入解释；RLVR 组保持最小可归因矩阵，不堆分布式训练或论文标签。Agent 仍作为后续独立项目。

## 12. 沟通与修改约束

- 用户正在学习。解释代码时需说明输入、输出、作用、为什么这样做和数据流，不能只丢命令。
- 代码改动前先说明计划，避免突然大改破坏学习顺序。
- Windows 命令必须先区分 CMD 和 PowerShell。
- 命令尽量提供可直接复制的一行版本。
- 项目价值判断要诚实，不包装成算法创新。
- 清洗规则必须可见、可审计；不确定领域不能整类删除。
- 抽查不等于逐条人工清洗。
- 修改前读取当前文件，保留用户已有结果，不覆盖长时间评测产生的 JSONL。
- 用户最终需要能够接受至少 30 分钟的项目面试深挖。

## 13. 新对话开场白

在新对话中直接发送：

```text
请先完整阅读 C:\Users\A\Desktop\ai\TRANSFORMERS\qwen_post_training_lab\PROJECT_HANDOFF.md、reports/final_comparison.md 和 reports/post_sft_upgrade_plan.md，并检查 Git 与当前产物。不要重新设计实验或覆盖已有 JSONL。先告诉我当前门禁状态、下一步唯一动作及其原因，再继续带我完成项目；每次代码改动都解释输入、输出、作用、数据流和设计理由。
```
