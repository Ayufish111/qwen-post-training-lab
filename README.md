# Qwen3 中文后训练实验

这个项目研究三个问题：数据清洗是否有效、LoRA 挂载范围如何影响效果、
最终 10,000 条 SFT 数据能把 Base 模型提升到什么程度。

## 实验设计

| 实验 | 是否训练 | 数据 | LoRA target | 目的 |
|---|---|---|---|---|
| B0 | 否 | 无 | 无 | Base 模型基线 |
| S1 | 是 | clean 2,000 | `q/k/v/o` | 注意力投影方案 |
| S2 | 是 | 同一份 clean 2,000 | `all-linear` | 全线性层方案 |
| S3 | 是 | raw 2,000 | S1/S2 胜者 | 数据质量消融 |
| S4 | 是 | full clean 10,000 | S1/S2 胜者 | 最终 SFT |

S1 和 S2 只改变 LoRA target。S3 与胜出的 clean 实验只改变数据质量；
两份 2,000 条数据的问题不重复，但任务类型和长度分布完全匹配。

## 目录

```text
configs/project.yaml          实验参数
data/sft/                     仅保留三份冻结的原始训练数据
data/cache/                   预处理产生的 tokenized 缓存
src/build_sft_datasets.py     清洗并构造三份数据
src/preprocess_sft.py         95/5 切分、套 Qwen 模板、构造 labels
src/train_sft.py              QLoRA 训练 S1-S4
scripts/prepare_multi_if.py   冻结 Multi-IF 中文正式评测集
src/evaluate_multi_if.py      Multi-IF 中文多轮规则评测
src/evaluate_instruction.py   20 条本地冒烟检查（非正式评测）
scripts/validate_dataset.py   数据和 labels 质量门禁
reports/                      审计报告与实验协议
outputs/                      checkpoint、adapter 和训练指标
```

`data/sft/` 只有：

```text
full_clean_10000.jsonl
ablation_clean_2000.jsonl
ablation_raw_2000.jsonl
```

## 执行顺序

本地已经完成第 1-5 步；当前从第 6 步继续。

```powershell
# 1. 构造三份冻结数据；已有最终文件时会确定性重分配
#    删除最终文件且无临时候选缓存时，才会流式读取 ModelScope 7M
python src/build_sft_datasets.py

# 2. 校验行数、schema、重复、集合关系和消融分布
python scripts/validate_dataset.py

# 3. 完成固定 100 条人工抽检，可中途退出后继续
python scripts/audit_sft.py

# 4. 审核通过后进行 token 化
python src/preprocess_sft.py

# 5. 再次校验，包含 input_ids/attention_mask/labels
python scripts/validate_dataset.py
```

两次校验都通过并接受审计边界后，将 `configs/project.yaml` 中的
`current_sft_status` 改成 `approved`。接着冻结正式评测集：

```powershell
# 6. 本地只执行一次：下载并冻结 Multi-IF 中文评测集
python scripts/prepare_multi_if.py
```

在 AutoDL 中克隆 Meta 官方规则代码，并只安装本地评分需要的依赖：

```powershell
git clone https://github.com/facebookresearch/Multi-IF.git third_party/Multi-IF
pip install nltk pythainlp langdetect emoji

# 7. 先用 2 题确认 B0 推理链路，再去掉 --limit 运行完整中文集
python src/evaluate_multi_if.py --experiment-id B0 --limit 2
python src/evaluate_multi_if.py --experiment-id B0
```

B0 完成后才开始付费 GPU 训练：

```powershell
python src/train_sft.py --experiment S1
python src/train_sft.py --experiment S2

# 根据 S1/S2 的正式评测结果选择 attention 或 all-linear
python src/train_sft.py --experiment S3 --target-strategy attention
python src/train_sft.py --experiment S4 --target-strategy attention
```

上例假设 `attention` 获胜；不能只根据训练 loss 选胜者。B0 不运行
`train_sft.py`，它是直接加载 Base 模型后执行同一评测流程。

## 能力边界

数据清洗包含结构校验、精确去重、显式硬规则、风险打分、任务平衡和
匹配消融构造。它不能自动证明每个答案事实正确。最终结论必须来自固定
评测、原始生成结果和错误分析，不能只比较 loss，也不能把抽查 200 条
说成“清洗了 10,000 条”。
