"""约束奖励、GRPO/DAPO 公式和生成长度统计。

本文件对应实验方案第 6 节，并遵守三个约定：

1. 文件本身只包含 CPU 逻辑，导入时不加载 Multi-IF 的 `ifeval.py`。正式运行时由调用方传入
   官方检查器，单元测试则可以传入轻量假检查器。
2. 约束奖励只检查 assistant 的最终 `content`；thinking 只统计长度和成本，不能参与约束得分。
3. R0 使用标准 GRPO 组内相对优势；R1/R2 额外使用 Clip-Higher、Token-Level Loss 和
   Soft Overlong。零方差检测是三组共用的运行时保护，不属于某一组的算法增量。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence


def parse_reasoning_answer(decoded: str) -> tuple[str, str] | None:
    """把 Qwen3 续写拆成 `(reasoning_content, content)`。

    `enable_thinking=True` 时，Qwen3 模板可能已经把 `<think>` 放进 prompt，因此模型续写通常
    只包含 `</think>`。这里同时兼容“带开头标签”和“不带开头标签”两种结果。若缺少结束标签、
    思考为空或最终回答为空，则返回 `None`，该回答之后会得到 0 分。
    """
    if not isinstance(decoded, str):
        return None
    text = decoded.strip()
    end_markers = ("<|im_end|>", "<|endoftext|>")
    while any(text.endswith(marker) for marker in end_markers):
        for marker in end_markers:
            if text.endswith(marker):
                text = text[: -len(marker)].rstrip()
                break
    if "</think>" not in text:
        return None
    reasoning, answer = text.split("</think>", 1)
    reasoning = reasoning.strip()
    if reasoning.startswith("<think>"):
        reasoning = reasoning[len("<think>") :].strip()
    answer = answer.strip()
    if not reasoning or not answer:
        return None
    return reasoning, answer


@dataclass(frozen=True)
class RewardScore:
    """实验方案 6.1 节定义的一条回答的奖励明细。"""

    r_instruction: float
    r_prompt: float
    r_core: float
    strict_results: tuple[bool, ...]
    instruction_count: int
    passed_count: int


class RewardCalculator:
    """使用 RLVR 官方约束检查器给一条模型续写打分。

    官方 checker 由外部传入：测试时可以注入假 checker；正式训练时则复用
    `scripts/sample_thinking_data.py` 已验证过的官方 checker 加载方式。
    """

    #: r_core 中“逐条约束平均通过率”的权重。
    INSTRUCTION_WEIGHT = 0.7
    #: r_core 中“全部约束同时通过”的权重。
    PROMPT_WEIGHT = 0.3

    def __init__(self, ifeval: Any) -> None:
        self.ifeval = ifeval
        # 记录 checker 自身异常，但不让一条坏参数终止整轮 RL 训练。
        self.checker_errors = 0

    def check_constraints(
        self,
        answer: str,
        instruction_ids: Sequence[str],
        kwargs_list: Sequence[dict[str, Any]],
    ) -> list[bool]:
        """逐条检查最终回答是否满足约束，返回与 instruction_ids 同顺序的布尔值。

        `answer` 已经移除了 thinking；`kwargs_list` 与 `instruction_ids` 一一对应。
        空回答会让所有约束都判为 False。
        """
        results: list[bool] = []
        if len(instruction_ids) != len(kwargs_list):
            raise ValueError(
                "instruction_ids and kwargs_list must have the same length"
            )
        if not answer or not answer.strip():
            return [False] * len(instruction_ids)
        for instruction_id, kwargs in zip(instruction_ids, kwargs_list):
            try:
                instruction_class = self.ifeval.INSTRUCTION_DICT[instruction_id]
                instruction = instruction_class(instruction_id)
                instruction.build_description(**kwargs)
                results.append(bool(instruction.check_following(answer)))
            except Exception:
                # Multi-IF 个别随机参数可能触发正则或 tokenizer 异常；对 RL 来说它只能是失败样本。
                self.checker_errors += 1
                results.append(False)
        return results

    @classmethod
    def score(cls, strict_results: Sequence[bool]) -> RewardScore:
        """根据各约束的布尔结果计算冻结的奖励公式。

        方案 6.1：
            r_instruction = passed_strict / total_strict
            r_prompt      = 1 if all strict passed else 0
            r_core        = 0.7 * r_instruction + 0.3 * r_prompt
        """
        results = tuple(bool(value) for value in strict_results)
        total = len(results)
        passed = sum(results)
        r_instruction = passed / total if total else 0.0
        r_prompt = 1.0 if total and passed == total else 0.0
        r_core = cls.INSTRUCTION_WEIGHT * r_instruction + cls.PROMPT_WEIGHT * r_prompt
        return RewardScore(
            r_instruction=r_instruction,
            r_prompt=r_prompt,
            r_core=r_core,
            strict_results=results,
            instruction_count=total,
            passed_count=passed,
        )

    def reward(
        self,
        row: dict[str, Any],
        completion_text: str,
    ) -> tuple[RewardScore, tuple[str, str] | None]:
        """解析完整续写并只给最终回答打分。

        返回 `(score, parsed)`；`parsed` 是 `(reasoning, content)`。若格式无法解析，
        `parsed=None` 且所有约束按 False 处理，也就是奖励为 0。
        """
        parsed = parse_reasoning_answer(completion_text)
        if parsed is None:
            # 无法拆出有效 thinking 与最终回答时记 0 分，绝不直接检查整段原始续写。
            return self.score([False] * len(row["instruction_ids"])), None
        answer = parsed[1]
        strict_results = self.check_constraints(
            answer, row["instruction_ids"], row["kwargs"]
        )
        return self.score(strict_results), parsed


def group_advantages(rewards: Sequence[float], epsilon: float = 1e-4) -> list[float]:
    """计算 GRPO 的组内相对优势（方案 6.2）。

    对同一个 prompt 生成的一组回答计算：
    `advantage_i = (reward_i - mean) / (std + epsilon)`。
    这里使用样本标准差；AutoDL 集成时还要与锁定版本的 TRL 行为核对。
    """
    if not rewards:
        return []
    if len(rewards) == 1:
        return [0.0]
    mean = sum(rewards) / len(rewards)
    variance = sum((value - mean) ** 2 for value in rewards) / (len(rewards) - 1)
    std = math.sqrt(variance)
    return [(value - mean) / (std + epsilon) for value in rewards]


def clip_higher(
    ratio: float,
    advantage: float,
    epsilon_low: float,
    epsilon_high: float,
) -> float:
    """计算单个 token/样本的 DAPO Clip-Higher loss 项。

    `ratio` 是新策略概率除以旧策略概率，`advantage` 是组内相对优势。DAPO 把对称裁剪改为：
    `clip(ratio, 1-epsilon_low, 1+epsilon_high)`，本项目冻结为下界 0.2、上界 0.28。
    返回值带负号，因为优化器执行的是最小化 loss。
    """
    lower = 1.0 - epsilon_low
    upper = 1.0 + epsilon_high
    unclipped = ratio * advantage
    clipped_ratio = min(max(ratio, lower), upper)
    clipped = clipped_ratio * advantage
    return -min(unclipped, clipped)


def token_level_policy_loss(
    ratios: Sequence[float],
    advantages: Sequence[float],
    weights: Sequence[float],
    epsilon_low: float,
    epsilon_high: float,
) -> float:
    """在所有有效生成 token 上计算 DAPO Token-Level Policy Loss。

    按序列平均会让长短回答的权重不一致；这里把所有有效 completion token 放在一起平均。
    `weights` 是有效 token 掩码，最终除以 `sum(weights)`。
    """
    if not ratios or len(ratios) != len(advantages) or len(ratios) != len(weights):
        raise ValueError("ratios, advantages and weights must be non-empty and same length")
    total_weight = sum(weights)
    if total_weight <= 0.0:
        return 0.0
    total = sum(
        clip_higher(ratio, advantage, epsilon_low, epsilon_high) * weight
        for ratio, advantage, weight in zip(ratios, advantages, weights)
    )
    return total / total_weight


def soft_overlong_threshold(max_completion_length: int, overlong_buffer: int) -> int:
    """计算 Soft Overlong 开始扣分的位置：最大长度减去缓冲区长度。"""
    return max(0, max_completion_length - max(0, overlong_buffer))


def soft_overlong_penalty(
    completion_length: int,
    max_completion_length: int,
    overlong_buffer: int,
) -> float:
    """计算一条回答的线性超长惩罚。

    回答未进入缓冲区时不扣分；从 threshold 到最大长度，惩罚由 0 线性下降到 -1；
    超过最大长度仍保持 -1：`-clip((length-threshold)/buffer, 0, 1)`。
    """
    threshold = soft_overlong_threshold(max_completion_length, overlong_buffer)
    if completion_length <= threshold:
        return 0.0
    return -min(
        (completion_length - threshold) / max(overlong_buffer, 1),
        1.0,
    )


def group_has_zero_reward_variance(rewards: Sequence[float], tolerance: float = 1e-8) -> bool:
    """判断同一 prompt 的一组回答是否没有相对学习信号。

    如果所有回答奖励都一样，组内 advantage 全为 0。max-min 小于 tolerance 时视为相同。
    本函数只负责检测，不决定训练器采用重试、丢弃还是停机策略。
    """
    if not rewards:
        return True
    if len(rewards) < 2:
        return True
    return max(rewards) - min(rewards) < tolerance


def count_text_tokens(text: str, tokenize: Callable[[str], int]) -> int:
    """通过外部传入的 tokenize 函数统计文本 token 数，本函数本身不加载模型。"""
    return int(tokenize(text))
