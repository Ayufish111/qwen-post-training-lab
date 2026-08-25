"""CA-DAPO 的约束感知动态采样权重。

本文件实现方案 6.4 冻结的公式：

    pass_ema_c(t) = beta * pass_ema_c(t-1) + (1-beta) * batch_pass_c(t)
    progress_c = max(0, pass_fast_c - pass_ema_c)
    learnability_c = progress_c * 4 * pass_ema_c * (1 - pass_ema_c)

一个 prompt 含多个约束类别时，取各类别 learnability 的平均值：

    prompt_learnability_i = mean(learnability_c for c in categories_i)

最终采样权重由均匀分布和归一化 learnability 混合：

    w_i = (1-lambda) * 1 + lambda * normalized_learnability_i
    w_i = clip(w_i, min_sampling_weight, max_sampling_weight)

冻结参数（必须与 configs/rlvr.yaml 保持一致）：

    ema_beta                     = 0.9
    uniform_mixture_lambda       = 0.5
    min_sampling_weight          = 0.5
    max_sampling_weight          = 2.0
    sampling_weight_update_steps = 20
    fast_ema_beta              = 0.5
    stagnation_tolerance       = 0.02
    stagnation_patience        = 3
    stagnation_scale           = 0.25

设计约束：EMA 只能由训练 rollout 更新，不能读取 Multi-IF dev/test；第一次更新窗口完成前所有
样本权重均为 1；未被训练 rollout 观察到的类别继续保持均匀权重。本文件不依赖 torch/transformers，
因此可以在本机用 CPU 单元测试锁定采样逻辑。
"""

from __future__ import annotations

import math
from typing import Any, Sequence


FROZEN_EMA_BETA = 0.9
FROZEN_MIXTURE_LAMBDA = 0.5
FROZEN_MIN_WEIGHT = 0.5
FROZEN_MAX_WEIGHT = 2.0
FROZEN_UPDATE_STEPS = 20
FROZEN_FAST_EMA_BETA = 0.5
FROZEN_STAGNATION_TOLERANCE = 0.02
FROZEN_STAGNATION_PATIENCE = 3
FROZEN_STAGNATION_SCALE = 0.25


def category_learnability(pass_ema: float) -> float:
    """由通过率 p 计算学习前沿分数 `4p(1-p)`。

    接近 0% 的类别可能暂时学不会，接近 100% 的类别已经掌握，两者都不应长期占据最高采样权重；
    约 50% 通过率的类别通常能提供更有区分度的 rollout。
    """
    pass_ema = float(pass_ema)
    if not math.isfinite(pass_ema) or not 0.0 <= pass_ema <= 1.0:
        raise ValueError(f"pass_ema must be in [0, 1], got {pass_ema!r}")
    return 4.0 * pass_ema * (1.0 - pass_ema)


def category_progress_signal(
    pass_ema: float,
    fast_pass_ema: float,
    stagnant_updates: int = 0,
    stagnation_patience: int = FROZEN_STAGNATION_PATIENCE,
    stagnation_scale: float = FROZEN_STAGNATION_SCALE,
) -> float:
    """用通过率前沿加权近期正向学习进步；停滞过久时再降低该信号。"""
    if stagnant_updates < 0 or stagnation_patience <= 0:
        raise ValueError("stagnation counters must be non-negative and patience positive")
    if not 0.0 < stagnation_scale <= 1.0:
        raise ValueError("stagnation_scale must be in (0, 1]")
    fast_pass_ema = float(fast_pass_ema)
    if not math.isfinite(fast_pass_ema) or not 0.0 <= fast_pass_ema <= 1.0:
        raise ValueError(f"fast_pass_ema must be in [0, 1], got {fast_pass_ema!r}")
    frontier = category_learnability(pass_ema)
    progress = max(0.0, fast_pass_ema - float(pass_ema))
    if stagnant_updates >= stagnation_patience:
        progress *= stagnation_scale
    return frontier * progress


def normalize_learnabilities(values: Sequence[float]) -> list[float]:
    """把 prompt learnability 做 min-max 归一化到 `[0,1]`。

    只有一个值或所有值相同时统一映射到 0.5，既避免除零，也保持均匀采样。
    """
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high - low < 1e-12:
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]


class ConstraintSampler:
    """维护各约束类别的 EMA 通过率，并计算 CA prompt 采样权重。

    每批 rollout 后调用 `update_batch_pass_rates` 输入各类别通过率；正式训练通过
    `weights_for_step` 获取权重，它会缓存当前权重并强制每 20 step 才更新。
    `sample_weights` 只负责纯公式计算。
    """

    def __init__(
        self,
        *,
        ema_beta: float = FROZEN_EMA_BETA,
        mixture_lambda: float = FROZEN_MIXTURE_LAMBDA,
        min_sampling_weight: float = FROZEN_MIN_WEIGHT,
        max_sampling_weight: float = FROZEN_MAX_WEIGHT,
        update_steps: int = FROZEN_UPDATE_STEPS,
        fast_ema_beta: float = FROZEN_FAST_EMA_BETA,
        stagnation_tolerance: float = FROZEN_STAGNATION_TOLERANCE,
        stagnation_patience: int = FROZEN_STAGNATION_PATIENCE,
        stagnation_scale: float = FROZEN_STAGNATION_SCALE,
    ) -> None:
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError("ema_beta must be in [0, 1)")
        if not 0.0 <= mixture_lambda <= 1.0:
            raise ValueError("mixture_lambda must be in [0, 1]")
        if min_sampling_weight <= 0.0 or max_sampling_weight < min_sampling_weight:
            raise ValueError("sampling weight bounds are invalid")
        if update_steps <= 0:
            raise ValueError("update_steps must be positive")
        if not 0.0 <= fast_ema_beta < 1.0:
            raise ValueError("fast_ema_beta must be in [0, 1)")
        if stagnation_tolerance < 0.0 or stagnation_patience <= 0:
            raise ValueError("stagnation settings are invalid")
        if not 0.0 < stagnation_scale <= 1.0:
            raise ValueError("stagnation_scale must be in (0, 1]")
        self.ema_beta = ema_beta
        self.mixture_lambda = mixture_lambda
        self.min_sampling_weight = min_sampling_weight
        self.max_sampling_weight = max_sampling_weight
        self.update_steps = update_steps
        self.fast_ema_beta = fast_ema_beta
        self.stagnation_tolerance = stagnation_tolerance
        self.stagnation_patience = stagnation_patience
        self.stagnation_scale = stagnation_scale
        #: 约束类别名称 -> 慢速 EMA 通过率。
        self.category_pass_ema: dict[str, float] = {}
        self.category_pass_fast_ema: dict[str, float] = {}
        self.category_last_batch_pass: dict[str, float] = {}
        self.category_stagnant_updates: dict[str, int] = {}
        self._active_row_ids: tuple[str, ...] | None = None
        self._active_weights: list[float] | None = None
        self.last_weight_update_step: int | None = None

    def should_update(self, step: int) -> bool:
        """判断当前 optimizer step 是否应该重新计算 CA 权重。

        冻结规则是每 20 step 更新一次；step 0 仍使用均匀权重。
        """
        return step > 0 and step % self.update_steps == 0

    def update_batch_pass_rates(
        self, batch_pass_rates: dict[str, float]
    ) -> None:
        """用一批训练 rollout 的类别通过率更新快慢 EMA。

        `batch_pass_rates` 的格式是“类别 -> 当前批平均通过率”。禁止传入 dev/test 结果。
        """
        for category, rate in batch_pass_rates.items():
            rate = float(rate)
            if not category or not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
                raise ValueError(
                    f"Invalid category pass rate: {category!r}={rate!r}"
                )
            previous = self.category_pass_ema.get(category)
            if previous is None:
                # 第一次观察到该类别时直接初始化快慢 EMA；实际权重仍受 20-step 更新节奏控制。
                self.category_pass_ema[category] = rate
                self.category_pass_fast_ema[category] = rate
                self.category_last_batch_pass[category] = rate
                self.category_stagnant_updates[category] = 0
            else:
                self.category_pass_ema[category] = (
                    self.ema_beta * previous + (1.0 - self.ema_beta) * rate
                )
                fast_previous = self.category_pass_fast_ema[category]
                self.category_pass_fast_ema[category] = (
                    self.fast_ema_beta * fast_previous
                    + (1.0 - self.fast_ema_beta) * rate
                )
                last_rate = self.category_last_batch_pass[category]
                if abs(rate - last_rate) <= self.stagnation_tolerance:
                    self.category_stagnant_updates[category] += 1
                else:
                    self.category_stagnant_updates[category] = 0
                self.category_last_batch_pass[category] = rate

    def has_data(self, category: str) -> bool:
        """训练 rollout 是否已经观察到这个类别。"""
        return category in self.category_pass_ema

    def learnability(self, category: str) -> float | None:
        """返回某个已观察类别的学习进展分数；未观察类别返回 None，权重保持 1。"""
        pass_ema = self.category_pass_ema.get(category)
        if pass_ema is None:
            return None
        return category_progress_signal(
            pass_ema,
            self.category_pass_fast_ema[category],
            self.category_stagnant_updates[category],
            self.stagnation_patience,
            self.stagnation_scale,
        )

    def prompt_learnability(self, row: dict[str, Any]) -> float | None:
        """计算一条 prompt 中已观察类别的平均 learnability。

        若所有类别都未观察则返回 None，继续使用权重 1；已观察和未观察类别混合时，只平均已观察类别。
        """
        categories = row.get("constraint_categories") or []
        known = [
            self.learnability(category)
            for category in categories
            if self.has_data(category)
        ]
        if not known:
            return None
        return sum(known) / len(known)

    def sample_weights(self, rows: Sequence[dict[str, Any]]) -> list[float]:
        """为冻结的 2k prompt 池计算 CA 采样权重。

        公式是 `w_i=(1-lambda)+lambda*normalized_learnability_i`，随后裁剪到冻结范围。
        R2 按这些权重抽题；R1 始终均匀抽题。
        """
        if not rows:
            return []
        learnabilities = [self.prompt_learnability(row) for row in rows]
        known_indices = [
            index
            for index, value in enumerate(learnabilities)
            if value is not None
        ]
        known_values = [learnabilities[index] for index in known_indices]  # type: ignore[misc]
        if known_values and max(known_values) <= 1e-12:
            return [1.0] * len(rows)
        normalized = normalize_learnabilities(known_values)
        normalized_by_index = dict(zip(known_indices, normalized))
        weights = [
            1.0 if index not in normalized_by_index else (
                (1.0 - self.mixture_lambda)
                + self.mixture_lambda * normalized_by_index[index]
            )
            for index, _ in enumerate(learnabilities)
        ]
        return [
            min(max(value, self.min_sampling_weight), self.max_sampling_weight)
            for value in weights
        ]

    @staticmethod
    def _row_ids(rows: Sequence[dict[str, Any]]) -> tuple[str, ...]:
        ids = tuple(str(row.get("id", "")) for row in rows)
        if any(not row_id for row_id in ids):
            raise ValueError("Every sampler row must have a non-empty id")
        if len(set(ids)) != len(ids):
            raise ValueError("Sampler row ids must be unique")
        return ids

    def weights_for_step(
        self,
        rows: Sequence[dict[str, Any]],
        step: int,
    ) -> list[float]:
        """返回缓存权重，并强制执行每 20 step 更新一次的规则。

        第一次调用会冻结整个训练池的 ID 顺序并返回全 1 权重。EMA 可以每批更新，但抽样权重只有
        到更新 step 才改变；若训练池顺序或 ID 被替换则直接报错，避免权重错配到其他 prompt。
        """
        if step < 0:
            raise ValueError("step cannot be negative")
        row_ids = self._row_ids(rows)
        if self._active_row_ids is None:
            self._active_row_ids = row_ids
            self._active_weights = [1.0] * len(rows)
        elif row_ids != self._active_row_ids:
            raise ValueError("Frozen sampler pool ids or order changed")

        if (
            self.should_update(step)
            and step != self.last_weight_update_step
        ):
            self._active_weights = self.sample_weights(rows)
            self.last_weight_update_step = step
        return list(self._active_weights or [])

    def state_dict(self) -> dict[str, Any]:
        """导出可序列化状态，供断点续训和审计。"""
        return {
            "ema_beta": self.ema_beta,
            "mixture_lambda": self.mixture_lambda,
            "min_sampling_weight": self.min_sampling_weight,
            "max_sampling_weight": self.max_sampling_weight,
            "update_steps": self.update_steps,
            "fast_ema_beta": self.fast_ema_beta,
            "stagnation_tolerance": self.stagnation_tolerance,
            "stagnation_patience": self.stagnation_patience,
            "stagnation_scale": self.stagnation_scale,
            "category_pass_ema": dict(self.category_pass_ema),
            "category_pass_fast_ema": dict(self.category_pass_fast_ema),
            "category_last_batch_pass": dict(self.category_last_batch_pass),
            "category_stagnant_updates": dict(self.category_stagnant_updates),
            "active_row_ids": list(self._active_row_ids or ()),
            "active_weights": list(self._active_weights or []),
            "last_weight_update_step": self.last_weight_update_step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """仅当保存状态与当前冻结参数一致时恢复 sampler。"""
        frozen = {
            "ema_beta": self.ema_beta,
            "mixture_lambda": self.mixture_lambda,
            "min_sampling_weight": self.min_sampling_weight,
            "max_sampling_weight": self.max_sampling_weight,
            "update_steps": self.update_steps,
            "fast_ema_beta": self.fast_ema_beta,
            "stagnation_tolerance": self.stagnation_tolerance,
            "stagnation_patience": self.stagnation_patience,
            "stagnation_scale": self.stagnation_scale,
        }
        for name, expected in frozen.items():
            if state.get(name) != expected:
                raise ValueError(
                    f"Sampler state {name}={state.get(name)!r} does not match "
                    f"the frozen value {expected!r}"
                )
        restored_ema = state.get("category_pass_ema") or {}
        self.category_pass_ema = {}
        self.category_pass_fast_ema = {}
        self.category_last_batch_pass = {}
        self.category_stagnant_updates = {}
        restored_fast = state.get("category_pass_fast_ema") or {}
        restored_last = state.get("category_last_batch_pass") or {}
        restored_stagnant = state.get("category_stagnant_updates") or {}
        self.update_batch_pass_rates(restored_ema)
        self.category_pass_fast_ema.update(
            {str(key): float(value) for key, value in restored_fast.items()}
        )
        self.category_last_batch_pass.update(
            {str(key): float(value) for key, value in restored_last.items()}
        )
        self.category_stagnant_updates.update(
            {str(key): int(value) for key, value in restored_stagnant.items()}
        )
        row_ids = tuple(str(value) for value in state.get("active_row_ids") or [])
        weights = [float(value) for value in state.get("active_weights") or []]
        if len(row_ids) != len(weights):
            raise ValueError("Sampler state row ids and weights differ in length")
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("Sampler state row ids must be unique")
        if any(
            not math.isfinite(value)
            or not self.min_sampling_weight <= value <= self.max_sampling_weight
            for value in weights
        ):
            raise ValueError("Sampler state contains an invalid active weight")
        self._active_row_ids = row_ids or None
        self._active_weights = weights or None
        last_step = state.get("last_weight_update_step")
        if last_step is not None and (not isinstance(last_step, int) or last_step < 0):
            raise ValueError("Sampler state has an invalid update step")
        self.last_weight_update_step = last_step
