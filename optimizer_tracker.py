"""Step-driven CSV collector. Call before/after the same successful optimizer update."""

import math

import torch
import torch.distributed as dist

from .gradient_statistics import CsvBuffer, LayerStatistics, local_tensor


class OptimizationTracker:
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
        self.previous_m = {}
        self.streak = {}

    def active(self, step):
        return step is not None and step > 0 and step % self.light_interval == 0

    @torch.no_grad()
    def before_update(self, step, optimizer, fallback_report=None):
        if not self.active(step):
            return
        groups = (
            {id(p): g for g in optimizer.param_groups for p in g["params"]}
            if optimizer
            else {}
        )
        gradients = {}
        metadata = {}
        weight_squares = []
        for layer, params in self.layers.items():
            gradients[layer] = []
            squares = []
            for name, param in params:
                owner = param
                shard = None
                flat = self.flat_shards.get(id(param))
                if (
                    flat is not None
                    and id(param) not in groups
                    and id(flat[0]) in groups
                ):
                    owner, offset, length = flat
                    shard = slice(offset, offset + length)
                weight = local_tensor(owner).reshape(-1)
                grad = (
                    local_tensor(owner.grad).reshape(-1)
                    if owner.grad is not None
                    else None
                )
                if shard is not None:
                    weight = weight[shard]
                    grad = grad[shard] if grad is not None else None
                weight = weight.float()
                squares.append(weight.dot(weight))
                if grad is not None:
                    gradients[layer].append((name, grad))
                group = groups.get(id(owner))
                if group is not None:
                    # Capture the LR actually used, before schedulers can advance it.
                    lr = group["lr"]
                    metadata[name] = (
                        group,
                        lr.detach().clone() if torch.is_tensor(lr) else lr,
                        grad is not None,
                        owner,
                        shard,
                    )
            weight_squares.append(
                torch.stack(squares).sum()
                if squares
                else torch.zeros((), device=self.stats.device)
            )
        if fallback_report is not None:
            fallback_report(gradients)
        rows = self.stats.collect(step, "gradient", gradients)
        self.pending = (step, rows, metadata, weight_squares)

    @torch.no_grad()
    def after_update(self, step, optimizer, legacy=None):
        if self.pending is None or self.pending[0] != step:
            return
        _, rows, metadata, weights = self.pending
        self.pending = None
        moments = {kind: {} for kind in ("exp_avg", "exp_avg_sq")}
        parameters = {}
        totals = []
        legacy_terms, legacy_ids, legacy_names = [], [], []
        heavy = step % self.heavy_interval == 0
        for i, (layer, params) in enumerate(self.layers.items()):
            for dest in moments.values():
                dest[layer] = []
            parameters[layer] = []
            delta_sq = weights[i].new_zeros(())
            missing = weights[i].new_zeros(())
            for name, param in params:
                meta = metadata.get(name)
                owner, shard = meta[3:] if meta is not None else (param, None)
                weight = local_tensor(owner).reshape(-1)
                if shard is not None:
                    weight = weight[shard]
                if heavy:
                    parameters[layer].append((name, weight))
                if not weight.numel():
                    continue
                state = optimizer.state.get(owner, {}) if optimizer else {}
                if shard is not None:
                    state = dict(state)
                    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        if key in state:
                            state[key] = local_tensor(state[key]).reshape(-1)[shard]
                for kind, dest in moments.items():
                    if kind in state:
                        dest[layer].append((name, local_tensor(state[kind])))
                if meta is None:
                    missing += 1
                    continue
                group, lr, updated = meta[:3]
                if legacy is not None:
                    zero = weights[i].new_zeros(())
                    if "exp_avg" in state and "exp_avg_sq" in state:
                        first = local_tensor(state["exp_avg"]).reshape(-1).float()
                        second = local_tensor(state["exp_avg_sq"]).reshape(-1).float()
                        # Existing TB delta_w_rms uses the uncorrected adaptive term.
                        adaptive = lr * first / (second.sqrt() + group["eps"])
                        legacy_terms.append(
                            torch.stack(
                                [
                                    first.dot(first),
                                    second.dot(second),
                                    adaptive.dot(adaptive),
                                    zero + 1,
                                ]
                            )
                        )
                    else:
                        legacy_terms.append(torch.stack([zero] * 4))
                    legacy_ids.append(legacy._param_primary_gid[name])
                    legacy_names.append(name)
                if not updated:
                    continue
                if (
                    not all(k in state for k in ("exp_avg", "exp_avg_sq"))
                    or "betas" not in group
                    or group.get("weight_decay", 0)
                ):
                    missing += 1
                    continue
                m = local_tensor(state["exp_avg"]).float()
                variance_key = (
                    "max_exp_avg_sq" if group.get("amsgrad") else "exp_avg_sq"
                )
                if variance_key not in state:
                    missing += 1
                    continue
                v = local_tensor(state[variance_key]).float()
                beta1, beta2 = group["betas"]
                bc1 = bc2 = 1.0
                if group.get("bias_correction", group.get("correct_bias", True)):
                    age = state.get("step")
                    if age is None:
                        missing += 1
                        continue
                    bc1 = 1 - beta1**age
                    bc2 = 1 - beta2**age
                update = (lr / bc1) * m / ((v / bc2).sqrt() + group["eps"])
                update = update.reshape(-1)
                delta_sq += update.dot(update)
            totals.append(torch.stack([weights[i], delta_sq, missing]))
        for kind, dest in moments.items():
            rows.extend(self.stats.collect(step, kind, dest))
        if totals:
            pack = torch.stack(totals)
            if self.stats.world > 1:
                dist.all_reduce(pack, group=self.stats.group)
            if self.write:
                for layer, (w, d, missing) in zip(self.layers, pack.tolist()):
                    valid = not missing and math.isfinite(d)
                    rows.append(
                        {
                            "step": step,
                            "layer": layer,
                            "kind": "update",
                            "update_norm": math.sqrt(d) if valid else None,
                            "update_to_weight_ratio": math.sqrt(d / w)
                            if valid and w > 0 and math.isfinite(w)
                            else None,
                        }
                    )
        report = (
            self._legacy_metrics(legacy, legacy_terms, legacy_ids, legacy_names)
            if legacy is not None
            else None
        )
        if self.write:
            self.light.add(rows)
        if heavy:
            records = self.stats.collect(step, "parameter", parameters, cosine=False)
            if self.write:
                self.heavy.add(records)

        return report

    def _legacy_metrics(self, monitor, terms, gids, names):
        group_count = monitor._num_groups
        sums = torch.zeros((group_count, 4), device=self.stats.device)
        stale = []
        if terms:
            per_param = torch.stack(terms)
            # Only four scalar contributions per parameter, independent of shard size.
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
        if self.write:
            self.light.flush()
            self.heavy.flush()
        self.stats.history.clear()
        self.pending = None
