"""Register this callback in place of GradientHealthMonitorCallback (v1.1.5)."""

import math
import os

import torch.distributed as dist

from .fallback_reporting import report_fallback
from .gradient_health_monitor import (
    GradientHealthMonitor,
    GradientHealthMonitorCallback,
    _env_int,
    _resolve_optimizer,
)
from .optimizer_tracker import OptimizationTracker


class GradientHealthTrackingCallback(GradientHealthMonitorCallback):
    def __init__(self):
        super().__init__()
        self.tracker = None
        self.light_interval = 20
        self._trainer = None

    def on_init_end(self, args, state, control, **kwargs):
        self.tracker = None
        self._trainer = kwargs.get("trainer")
        super().on_init_end(args, state, control, **kwargs)
        monitor = self._monitor
        if monitor is None or monitor.mode == monitor.MODE_OFF:
            return
        self.light_interval = max(1, _env_int("GRADIENT_HEALTH_LIGHT_K", 20))
        heavy_interval = _env_int("GRADIENT_HEALTH_PARAMETER_K", 200)
        trainer = kwargs["trainer"]
        strategy = getattr(trainer.args, "parallel_strategy", "ddp")
        group = None
        sharded = False
        flat_shards = {}
        replica_names = set()
        if strategy == "fsdp1" or monitor.mode == monitor.MODE_HOOK:
            from .gradient_monitor_fsdp import extract_fsdp1_config

            config = extract_fsdp1_config(trainer)
            group = config.get("process_group")
            sharded = bool(config.get("needs_shard_reduction"))
            for root in config.get("fsdp_modules") or ():
                for module in root.modules():
                    flat = getattr(getattr(module, "_handle", None), "flat_param", None)
                    infos = getattr(flat, "_shard_param_infos", ())
                    for param, info in zip(getattr(flat, "_params", None) or (), infos):
                        flat_shards[id(param)] = (
                            flat,
                            info.offset_in_shard or 0,
                            info.numel_in_shard or 0,
                        )
        elif strategy == "fsdp2":
            from torch.distributed.tensor import Shard

            for name, p in monitor.monitored_params.items():
                mesh = getattr(p, "device_mesh", None)
                if mesh is None:
                    replica_names.add(name)
                    continue
                dims = [
                    i
                    for i, placement in enumerate(p.placements)
                    if isinstance(placement, Shard)
                ]
                if len(dims) == 1:
                    param_group = mesh.get_group(dims[0])
                    if sharded and group is not param_group:
                        raise ValueError(
                            "Monitored FSDP2 parameters span different shard groups"
                        )
                    group = param_group
                    sharded = True
                elif not dims:
                    replica_names.add(name)
                if len(dims) > 1:
                    raise ValueError(
                        "Multiple sharding dimensions need an explicit statistics group"
                    )
        if not sharded and self.rank != 0:
            return
        if sharded and not dist.is_initialized():
            raise RuntimeError(
                "A sharded monitor requires an initialized process group"
            )
        directory = os.getenv("GRADIENT_HEALTH_CSV_DIR") or os.path.join(
            args.output_dir, "gradient_health"
        )
        self.tracker = OptimizationTracker(
            monitor.monitored_params,
            directory,
            monitor._device,
            group,
            sharded,
            self.light_interval,
            heavy_interval,
            _env_int("GRADIENT_HEALTH_LIGHT_FLUSH", 5),
            _env_int("GRADIENT_HEALTH_HEAVY_FLUSH", 1),
            monitor._near_zero,
            write=self.rank == 0,
            flat_shards=flat_shards,
            replica_names=replica_names,
        )

    def _active(self, kwargs):
        step = kwargs.get("cur_it")
        return (
            self._monitor is not None
            and step is not None
            and step > 0
            and step % self.light_interval == 0
        )

    def on_pre_backward(self, args, state, control, **kwargs):
        if self._active(kwargs):
            super().on_pre_backward(args, state, control, **kwargs)
        elif (
            self._monitor is not None
            and self._monitor.mode == GradientHealthMonitor.MODE_HOOK
        ):
            self._monitor._hook._step_active = False

    def on_post_backward(self, args, state, control, **kwargs):
        if self._active(kwargs):
            super().on_post_backward(args, state, control, **kwargs)

    def _optimizer(self, kwargs):
        opt = kwargs.get("optimizer")
        return _resolve_optimizer(
            opt if opt is not None else kwargs.get("trainer", self._trainer)
        )

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self._active(kwargs) and self.tracker is not None:
            step = kwargs["cur_it"]
            fallback = None
            if self._monitor.mode == self._monitor.MODE_FALLBACK:

                def fallback(gradients):
                    report_fallback(
                        self._monitor,
                        self.tracker,
                        gradients,
                        step,
                        self._tb_logger.add_scalar,
                        self._str_logger,
                    )

            self.tracker.before_update(step, self._optimizer(kwargs), fallback)

    def on_optimizer_step(self, args, state, control, **kwargs):
        if self._active(kwargs):
            monitor = self._monitor
            step = kwargs["cur_it"]
            if monitor.mode == monitor.MODE_HOOK:
                monitor.post_optimizer(
                    step, self._tb_logger.add_scalar, self._str_logger
                )
            if self.tracker is not None:
                opt = self._optimizer(kwargs)
                report = self.tracker.after_update(
                    step, opt, legacy=monitor if step % monitor._heavy_k == 0 else None
                )
                if report is not None and opt is not None:
                    self._report_optimizer(step, args, *report)

    def _report_optimizer(self, step, args, sums, stale):
        monitor = self._monitor
        if (
            stale
            and self._str_logger is not None
            and monitor._warn_throttled("opt_stale", step)
        ):
            self._str_logger.info(
                f"[GradientHealthMonitor] step={step} stale params "
                f"count={len(stale)} sample={stale[:8]}"
            )
        if step % monitor._tb_freq:
            return
        tb = self._tb_logger.add_scalar
        prefix = "GradientHealth/optimizer"
        tb(f"{prefix}/grad_clip_enabled", 0, step)
        tb(
            f"{prefix}/grad_accum_steps",
            max(1, getattr(args, "gradient_accumulation_steps", 1)),
            step,
        )
        tb(f"{prefix}/stale_param_count", len(stale), step)
        if not any(row[3] for row in sums):
            return
        numels = monitor._last_numels or monitor._expected_numel
        for i, metric in enumerate(("m_rms", "v_rms", "delta_w_rms")):
            count = sum(numels)
            tb(
                f"{prefix}/{metric}",
                math.sqrt(sum(r[i] for r in sums) / count) if count else 0.0,
                step,
            )
            for name, row, n in zip(monitor._group_names, sums, numels):
                tb(
                    f"GradientHealth/{name}/opt_{metric}",
                    math.sqrt(row[i] / n) if n else 0.0,
                    step,
                )
        tb(f"{prefix}/wd_ratio", 0.0, step)

    def on_train_end(self, args, state, control, **kwargs):
        try:
            if self.tracker is not None:
                self.tracker.close()
        finally:
            self._trainer = None
            super().on_train_end(args, state, control, **kwargs)
