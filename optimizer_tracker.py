"""Step-driven CSV collector. Call before/after the same successful optimizer update."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
import torch.distributed as dist

from .gradient_statistics import CsvBuffer, LayerStatistics, local_tensor


class _GroupStep(NamedTuple):
    config: dict
    lr: float | torch.Tensor
    decay: float
    mode: str  # decoupled、coupled、unknown；零衰减无需区分。


def _optimizer_groups(optimizer):
    """每组只固定一次超参数；自定义优化器可显式声明兼容的解耦规则。"""
    groups = {}
    for group in optimizer.param_groups if optimizer else ():
        lr = group["lr"]
        if torch.is_tensor(lr):
            lr = lr.detach().clone()
        if group.get(
            "decoupled_weight_decay", isinstance(optimizer, torch.optim.AdamW)
        ):
            mode = "decoupled"
        elif isinstance(optimizer, torch.optim.Adam):
            mode = "coupled"
        else:
            mode = "unknown"
        settings = _GroupStep(group, lr, float(group.get("weight_decay", 0)), mode)
        groups.update((id(param), settings) for param in group["params"])
    return groups


class _ParameterSnapshot(NamedTuple):
    settings: _GroupStep
    had_grad: bool
    owner: torch.Tensor
    shard: slice | None
    old_weight: torch.Tensor | None


class _PendingStep(NamedTuple):
    step: int
    rows: list
    metadata: dict
    weight_squares: dict


@dataclass
class _LayerUpdate:
    """一层在本 rank 的更新后观测。空分片也保留相同的层顺序。"""

    totals: torch.Tensor | None = None
    missing: int = 0
    unsplittable: int = 0
    exp_avg: list = field(default_factory=list)
    exp_avg_sq: list = field(default_factory=list)
    parameters: list = field(default_factory=list)


def _local_slice(tensor, shard):
    values = local_tensor(tensor).reshape(-1)
    return values if shard is None else values[shard]


def _optimizer_state(optimizer, owner, shard):
    state = optimizer.state.get(owner, {}) if optimizer else {}
    if shard is None:
        return state
    # 切片只属于本层；更新后仍需用 flat 参数作为优化器状态的查找键。
    state = dict(state)
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        if key in state:
            state[key] = _local_slice(state[key], shard)
    return state


def _adam_direction(meta, state):
    """由更新后的矩还原 Adam 下降方向（含学习率，实际步长取负号）。"""
    group = meta.settings.config
    variance_key = "max_exp_avg_sq" if group.get("amsgrad") else "exp_avg_sq"
    if "betas" not in group or not {"exp_avg", "exp_avg_sq", variance_key}.issubset(
        state
    ):
        return None

    first = local_tensor(state["exp_avg"]).float()
    variance = local_tensor(state[variance_key]).float()
    correction1 = correction2 = 1.0
    if group.get("bias_correction", group.get("correct_bias", True)):
        age = state.get("step")
        if age is None:
            return None
        beta1, beta2 = group["betas"]
        correction1, correction2 = 1 - beta1**age, 1 - beta2**age
    update = (
        (meta.settings.lr / correction1)
        * first
        / ((variance / correction2).sqrt() + group["eps"])
    )
    return update.reshape(-1)


class _UpdateSquares:
    """每层先在本地累计，最后与旧权重平方和一起跨卡求和。"""

    def __init__(self, reference):
        self.total = reference.new_zeros(())
        self.adaptive = reference.new_zeros(())
        self.decay = reference.new_zeros(())
        self.missing = 0
        self.unsplittable = 0

    def add(self, meta, state):
        if meta is None:
            self.missing += 1
            return
        if not meta.had_grad:
            return
        settings = meta.settings
        direction = _adam_direction(meta, state)
        if direction is None or (settings.decay and settings.mode == "unknown"):
            self.missing += 1
            return
        direction_sq = direction.dot(direction)
        if settings.decay and settings.mode == "coupled":
            # 矩已经混入 L2 项，仍能计算总步长。
            self.unsplittable += 1
            self.total += direction_sq
            return
        self.adaptive += direction_sq
        if meta.old_weight is None:
            self.total += direction_sq
            return
        decay = meta.old_weight.float() * (settings.lr * settings.decay)
        self.decay += decay.dot(decay)
        combined = decay.add_(direction)
        self.total += combined.dot(combined)


def _norm(square):
    return math.sqrt(square) if math.isfinite(square) and square >= 0 else None


def _ratio(numerator, denominator):
    if numerator is not None and denominator is not None and denominator > 0:
        return numerator / denominator
    return None


def _update_row(step, layer, values):
    weight, total, adaptive, decay, missing, unsplittable = values
    total_norm = None if missing else _norm(total)
    components_known = not (missing or unsplittable)
    grad_norm = _norm(adaptive) if components_known else None
    decay_norm = _norm(decay) if components_known else None
    return {
        "step": step,
        "layer": layer,
        "kind": "update",
        "update_norm": total_norm,
        "update_to_weight_ratio": _ratio(total_norm, _norm(weight)),
        "gradient_update_norm": grad_norm,
        "weight_decay_update_norm": decay_norm,
        "wd_ratio": _ratio(decay_norm, grad_norm),
    }


def _legacy_terms(meta, state, weight_square):
    """旧 TB 的自适应项沿用未校正的矩，确保同历史曲线可比。"""
    zero = weight_square.new_zeros(())
    if "exp_avg" not in state or "exp_avg_sq" not in state:
        return torch.stack([zero] * 4)
    first = local_tensor(state["exp_avg"]).reshape(-1).float()
    second = local_tensor(state["exp_avg_sq"]).reshape(-1).float()
    adaptive = meta.settings.lr * first / (second.sqrt() + meta.settings.config["eps"])
    return torch.stack(
        [first.dot(first), second.dot(second), adaptive.dot(adaptive), zero + 1]
    )


class OptimizationTracker:
    """为一次 optimizer 更新配对采集梯度、参数和 Adam 状态。

    parameters 的键是参数完整路径，同一直接父模块的 weight/bias 归入一层。
    light_interval 和 heavy_interval 以训练步计数；light_flush 和 heavy_flush
    分别以各自通道的采集次数计数。分片训练中，同组所有 rank 必须按同一步号调用。
    """

    def __init__(
        self,
        parameters,
        directory,
        device,
        group=None,
        sharded=False,
        light_interval=20,
        heavy_interval=200,
        light_flush=5,
        heavy_flush=1,
        near_zero=1e-6,
        write=True,
        flat_shards=None,
        replica_names=(),
    ):
        if light_interval < 1 or heavy_interval % light_interval or heavy_interval < 1:
            raise ValueError(
                "heavy_interval must be a positive multiple of light_interval"
            )
        self.light_interval = light_interval
        self.heavy_interval = heavy_interval
        self.stats = LayerStatistics(device, group, sharded, near_zero)
        self.layers = {}
        # 各 rank 按同一层顺序发起通信。即使本 rank 跳过复制参数，也要保留层的空槽。
        for name, param in sorted(parameters.items()):
            if param.requires_grad:
                layer = name.rpartition(".")[0] or "<root>"
                parts = self.layers.setdefault(layer, [])
                if self.stats.rank == 0 or name not in replica_names:
                    parts.append((name, param))
        self.light = CsvBuffer(directory, "light", light_flush)
        self.heavy = CsvBuffer(directory, "heavy", heavy_flush)
        self.write = write
        self.flat_shards = flat_shards or {}
        self.pending = None
        self.last_wd_ratio = None
        self.previous_m = {}
        self.streak = {}

    def active(self, step):
        """第 0 步跳过；重步也是一次轻通道采集。"""
        return step is not None and step > 0 and step % self.light_interval == 0

    @torch.no_grad()
    def before_update(self, step, optimizer, fallback_report=None):
        """采集同步后的梯度，把分层旧权重平方和与本次更新配对。"""
        if not self.active(step):
            return
        groups = _optimizer_groups(optimizer)
        gradients, metadata, weight_squares = {}, {}, {}
        for layer, params in self.layers.items():
            gradients[layer], weight_squares[layer] = self._read_before_layer(
                params, groups, metadata
            )
        if fallback_report is not None:
            fallback_report(gradients)
        rows = self.stats.collect(step, "gradient", gradients)
        self.pending = _PendingStep(step, rows, metadata, weight_squares)

    def _parameter_slice(self, param, groups):
        """例如 flat 中 A 占 3 个数、B 占 2 个数，B 在本 rank 的切片是 [3:5]。"""
        flat = self.flat_shards.get(id(param))
        if flat is None or id(param) in groups or id(flat[0]) not in groups:
            return param, None
        owner, offset, length = flat
        return owner, slice(offset, offset + length)

    def _read_before_layer(self, params, groups, metadata):
        gradients, squares = [], []
        for name, param in params:
            owner, shard = self._parameter_slice(param, groups)
            local_weight = _local_slice(owner, shard)
            weight = local_weight.float()
            squares.append(weight.dot(weight))
            had_grad = owner.grad is not None
            if had_grad:
                gradients.append((name, _local_slice(owner.grad, shard)))
            settings = groups.get(id(owner))
            if settings is not None:
                snapshot = None
                if had_grad and settings.decay and settings.mode == "decoupled":
                    snapshot = local_weight.clone()
                metadata[name] = _ParameterSnapshot(
                    settings, had_grad, owner, shard, snapshot
                )
        total = (
            torch.stack(squares).sum()
            if squares
            else torch.zeros((), device=self.stats.device)
        )
        return gradients, total

    @torch.no_grad()
    def after_update(self, step, optimizer, legacy=None):
        """配对同一步的旧权重与新矩，汇总本次更新并写入对应通道。"""
        pending = self.pending
        if pending is None or pending.step != step:
            return
        self.pending = None
        heavy = step % self.heavy_interval == 0
        legacy_terms = {} if legacy is not None else None
        layers = {}
        for layer, params in self.layers.items():
            layers[layer] = self._read_after_layer(
                params,
                pending.metadata,
                pending.weight_squares[layer],
                optimizer,
                heavy,
                legacy_terms,
            )
        self._append_update_rows(step, pending.rows, layers)
        report = (
            self._legacy_metrics(legacy, legacy_terms) if legacy is not None else None
        )
        if self.write:
            self.light.add(pending.rows)
        if heavy:
            parameters = {name: layer.parameters for name, layer in layers.items()}
            records = self.stats.collect(step, "parameter", parameters, cosine=False)
            if self.write:
                self.heavy.add(records)
        return report

    def _read_after_layer(
        self, params, metadata, weight_square, optimizer, heavy, legacy_terms
    ):
        squares = _UpdateSquares(weight_square)
        layer = _LayerUpdate()
        for name, param in params:
            # 读完这一段即可释放旧权重，后续全层分布统计无需保留快照。
            meta = metadata.pop(name, None)
            owner, shard = (meta.owner, meta.shard) if meta else (param, None)
            weight = _local_slice(owner, shard)
            if heavy:
                layer.parameters.append((name, weight))
            if not weight.numel():
                continue
            state = _optimizer_state(optimizer, owner, shard)
            for kind in ("exp_avg", "exp_avg_sq"):
                if kind in state:
                    getattr(layer, kind).append((name, local_tensor(state[kind])))
            if legacy_terms is not None and meta is not None:
                legacy_terms[name] = _legacy_terms(meta, state, weight_square)
            squares.add(meta, state)
        layer.totals = torch.stack(
            [weight_square, squares.total, squares.adaptive, squares.decay]
        )
        layer.missing = squares.missing
        layer.unsplittable = squares.unsplittable
        return layer

    def _append_update_rows(self, step, rows, layers):
        for kind in ("exp_avg", "exp_avg_sq"):
            moments = {name: getattr(layer, kind) for name, layer in layers.items()}
            rows.extend(self.stats.collect(step, kind, moments))
        self.last_wd_ratio = None
        if not layers:
            return
        # 各卡先加平方和，再算层范数；旧范数 3、4 对应整层范数 5。
        totals = torch.stack([layer.totals for layer in layers.values()])
        # 所有层的有效性标记一次传到设备，和平方和共用一次归约。
        flags = totals.new_tensor(
            [(layer.missing, layer.unsplittable) for layer in layers.values()]
        )
        totals = torch.cat((totals, flags), dim=1)
        if self.stats.world > 1:
            dist.all_reduce(totals, group=self.stats.group)
        if self.write:
            values = totals.tolist()
            rows.extend(
                _update_row(step, layer, row) for layer, row in zip(layers, values)
            )
            # 全局比值用全部受监控参数的平方和计算，层比值不能直接取平均。
            _, _, adaptive, decay, missing, unsplittable = map(sum, zip(*values))
            if not (missing or unsplittable):
                self.last_wd_ratio = _ratio(_norm(decay), _norm(adaptive))

    def _legacy_metrics(self, monitor, terms):
        """按旧 TB 主组汇总标量；stale 名单反映写入 rank 本地一阶矩平方和的变化。"""
        group_count = monitor._num_groups
        sums = torch.zeros((group_count, 4), device=self.stats.device)
        stale = []
        if terms:
            names = list(terms)
            gids = [monitor._param_primary_gid[name] for name in names]
            per_param = torch.stack(list(terms.values()))
            # per_param 每行是一个参数的一阶矩平方和、二阶矩平方和、自适应更新
            # 平方和及状态覆盖标志。gids 将行累加到旧 TB 主组，最后一列给出覆盖数。
            sums.index_add_(0, torch.tensor(gids, device=self.stats.device), per_param)
            if self.write:
                for name, value in zip(names, per_param[:, 0].tolist()):
                    streak = (
                        self.streak.get(name, 0) + 1
                        if self.previous_m.get(name) == value
                        else 0
                    )
                    self.streak[name] = streak
                    self.previous_m[name] = value
                    if streak > monitor._stale_patience:
                        stale.append(name)
        if self.stats.world > 1:
            dist.all_reduce(sums, group=self.stats.group)
        return (sums.tolist(), stale) if self.write else None

    def close(self):
        """写出未满批的记录并释放跨步快照，供训练结束时调用。"""
        if self.write:
            self.light.flush()
            self.heavy.flush()
        self.stats.history.clear()
        self.pending = None
