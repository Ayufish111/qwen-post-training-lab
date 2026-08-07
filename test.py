from collections import defaultdict

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen3-4B-Base"
LORA_R = 16

config = AutoConfig.from_pretrained(MODEL_NAME)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

total_params = sum(parameter.numel() for parameter in model.parameters())
module_stats = defaultdict(lambda: {"count": 0, "base": 0, "lora": 0})

for module_name, module in model.named_modules():
    if not isinstance(module, torch.nn.Linear):
        continue

    short_name = module_name.rsplit(".", 1)[-1]
    base_params = module.in_features * module.out_features

    if module.bias is not None:
        base_params += module.out_features

    lora_params = LORA_R * (module.in_features + module.out_features)

    module_stats[short_name]["count"] += 1
    module_stats[short_name]["base"] += base_params
    module_stats[short_name]["lora"] += lora_params

print("模型总参数：", f"{total_params:,}")
print("\n线性模块统计（LoRA r=16）：")

for module_name, stats in module_stats.items():
    print(
        f"{module_name:12} "
        f"层数={stats['count']:2}  "
        f"原参数={stats['base']:>12,}  "
        f"LoRA参数={stats['lora']:>10,}"
    )
all_linear_modules = set(module_stats) - {"lm_head"}

strategies = {
    "q_proj + v_proj": {"q_proj", "v_proj"},
    "全部注意力投影": {"q_proj", "k_proj", "v_proj", "o_proj"},
    "全部Transformer线性层": all_linear_modules,
}

print("\n候选 target_modules 对比：")

for strategy_name, target_modules in strategies.items():
    targeted_base_params = sum(
        module_stats[name]["base"]
        for name in target_modules
        if name in module_stats
    )
    trainable_lora_params = sum(
        module_stats[name]["lora"]
        for name in target_modules
        if name in module_stats
    )
    trainable_ratio = trainable_lora_params / (
        total_params + trainable_lora_params
    )

    print(f"\n{strategy_name}")
    print("target_modules：", sorted(target_modules))
    print("覆盖原模型参数比例：", f"{targeted_base_params / total_params:.2%}")
    print("新增可训练LoRA参数：", f"{trainable_lora_params:,}")
    print("LoRA可训练参数比例：", f"{trainable_ratio:.4%}")
