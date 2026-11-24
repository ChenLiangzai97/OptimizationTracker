import csv
import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .gradient_statistics import LayerStatistics
from .optimizer_tracker import OptimizationTracker


def distributed_worker(rank, rendezvous, threaded=False):
    try:
        _run_shard(rank, rendezvous, threaded)
    except BaseException as exc:
        if threaded:
            from torch.testing._internal.distributed.multi_threaded_pg import (
                ProcessLocalGroup,
            )

            ProcessLocalGroup.exception_handle(exc)
        raise


def _run_shard(rank, rendezvous, threaded):
    if threaded:
        dist.init_process_group("threaded", store=rendezvous, rank=rank, world_size=2)
    else:
        dist.init_process_group(
            "gloo", init_method="file://" + rendezvous, rank=rank, world_size=2
        )
    try:
        stats = LayerStatistics("cpu", sharded=True)
        values = (
            torch.tensor([-5.0, 1.0, float("nan")])
            if rank == 0
            else torch.tensor([10.0])
        )
        other = torch.tensor([]) if rank == 0 else torch.tensor([2.0])
        layers = {"a": [("w", values)], "b": [("b", other)]}
        stats.collect(20, "grad", layers)
        records = stats.collect(40, "grad", layers)
        if rank == 0:
            row, single = records
            assert row["p50"] == 1 and abs(row["p99"] - 9.82) < 1e-5
            assert row["nonfinite_numel"] == 1 and row["cosine"] is None
            assert sum(row[f"bin_{i}_count"] for i in range(10)) == 3
            assert single["cosine"] == 1 and single["bin_0_count"] == 1
        _check_sharded_decay(rank)
    finally:
        dist.destroy_process_group()


def _check_sharded_decay(rank):
    # 同一逻辑层的两个分片：衰减与梯度在两张卡上分别同向、反向。
    p = torch.nn.Parameter(torch.tensor([10.0] if rank == 0 else [2.0, 4.0]))
    opt = torch.optim.AdamW([p], lr=0.1, weight_decay=0.1, eps=0.0)
    p.grad = torch.ones_like(p) if rank == 0 else -torch.ones_like(p)
    with tempfile.TemporaryDirectory() as folder:
        tracker = OptimizationTracker(
            {"layer.weight": p}, folder, "cpu", sharded=True, write=rank == 0
        )
        tracker.before_update(20, opt)
        opt.step()
        tracker.after_update(20, opt)
        if rank == 0:
            row = tracker.light.pending[-1]
            decay_norm = (0.1**2 + 0.02**2 + 0.04**2) ** 0.5
            grad_norm = (3 * 0.1**2) ** 0.5
            total_norm = (0.2**2 + 0.08**2 + 0.06**2) ** 0.5
            for key, ref in (
                ("wd_ratio", decay_norm / grad_norm),
                ("update_norm", total_norm),
            ):
                assert abs(row[key] - ref) < 1e-5
            assert abs(tracker.last_wd_ratio - decay_norm / grad_norm) < 1e-5


def _layer_norm(values, names):
    return torch.cat([values[k] for k in names]).norm().item()


class TrackerTest(unittest.TestCase):
    def test_shards(self):
        if os.getenv("TRACKER_TEST_THREADED_PG") == "1":
            from concurrent.futures import ThreadPoolExecutor

            from torch.testing._internal.distributed import (
                multi_threaded_pg as threaded_pg,
            )

            store = dist.HashStore()
            torch._C._distributed_c10d._set_thread_isolation_mode(True)
            threaded_pg._install_threaded_pg()
            try:
                with ThreadPoolExecutor(2) as pool:
                    tasks = [
                        pool.submit(distributed_worker, rank, store, True)
                        for rank in range(2)
                    ]
                    for task in tasks:
                        task.result()
            finally:
                threaded_pg._uninstall_threaded_pg()
                torch._C._distributed_c10d._set_thread_isolation_mode(False)
            return
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(
                distributed_worker, args=(os.path.join(folder, "store"),), nprocs=2
            )

    def test_full_cycle(self):
        torch.manual_seed(4)
        model = torch.nn.Linear(8, 3)
        params = dict(model.named_parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
        with tempfile.TemporaryDirectory() as folder:
            tracker = OptimizationTracker(
                params, folder, "cpu", light_flush=2, heavy_flush=2
            )
            for step in (1, 20, 40, 200):
                before = torch.cat([p.detach().flatten() for p in params.values()])
                for p in params.values():
                    p.grad = torch.ones_like(p)
                tracker.before_update(step, opt)
                opt.step()
                tracker.after_update(step, opt)
                after = torch.cat([p.detach().flatten() for p in params.values()])
                if step == 20:
                    self.assertFalse(list(Path(folder).glob("*.csv")))
                    self.assertAlmostEqual(
                        tracker.light.pending[-1]["update_to_weight_ratio"],
                        float((before - after).norm() / before.norm()),
                        places=6,
                    )
                if step == 40:
                    with next(Path(folder).glob("light*.csv")).open(
                        encoding="utf-8-sig"
                    ) as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(len(rows), 8)
                    self.assertAlmostEqual(float(rows[4]["cosine"]), 1.0, places=6)
            tracker.close()
            self.assertEqual(len(list(Path(folder).glob("*.csv"))), 3)

    def test_flat_optimizer_maps_unaligned_layers(self):
        # Optimizer owns a flat shard; monitored views have no independent state.
        owner = torch.nn.Parameter(torch.tensor([1.0, 2.0, 4.0, 8.0]))
        refs = {n: torch.nn.Parameter(torch.zeros(1)) for n in ("a.weight", "b.bias")}
        offsets = {id(p): (owner, pos, 2) for pos, p in zip((0, 2), refs.values())}
        opt = torch.optim.AdamW([owner], lr=0.1, weight_decay=0)
        with tempfile.TemporaryDirectory() as folder:
            tracker = OptimizationTracker(
                refs, folder, "cpu", flat_shards=offsets, light_flush=5
            )
            owner.grad = torch.tensor([1.0, 3.0, 5.0, 7.0])
            before = owner.detach().clone()
            tracker.before_update(20, opt)
            opt.step()
            tracker.after_update(20, opt)
            rows = {(r["layer"], r["kind"]): r for r in tracker.light.pending}
            self.assertEqual(rows["b", "gradient"]["min"], 5.0)
            self.assertAlmostEqual(rows["b", "exp_avg"]["max"], 0.7, places=6)
            for layer, pos in (("a", 0), ("b", 2)):
                section = slice(pos, pos + 2)
                ratio = (
                    (before - owner.detach())[section].norm() / before[section].norm()
                ).item()
                self.assertAlmostEqual(
                    rows[layer, "update"]["update_to_weight_ratio"], ratio, places=6
                )

    def test_nonfinite_constant_and_missing(self):
        stats = LayerStatistics("cpu")
        layers = {
            "mixed": [("w", torch.tensor([float("inf"), 0.0, float("nan")]))],
            "missing": [],
            "bad": [("w", torch.tensor([float("nan")]))],
        }
        for cosine in (False, True):
            mixed, missing, bad = stats.collect(20, "gradient", layers, cosine)
            self.assertEqual(mixed["near_zero_ratio"], 1.0)
            self.assertEqual(mixed["bin_0_count"], 1)
            self.assertEqual(mixed["nonfinite_numel"], 2)
            self.assertIsNone(missing["rms"])
            self.assertIsNone(bad["max"])
            self.assertEqual(sum(bad[f"bin_{i}_count"] for i in range(10)), 0)

    def test_weight_decay_matches_real_updates(self):
        # 两层、三组，不同学习率；bias 无衰减。用真实权重变化分离自适应分量。
        params = {
            name: torch.nn.Parameter(torch.tensor(values))
            for name, values in (
                ("a.weight", [4.0, -2.0]),
                ("a.bias", [0.5]),
                ("b.weight", [12.0, 3.0]),
            )
        }
        groups = [
            {"params": [p], "lr": lr, "weight_decay": wd}
            for p, lr, wd in zip(params.values(), (0.1, 0.04, 0.02), (0.25, 0.0, 0.5))
        ]
        opt = torch.optim.AdamW(groups, amsgrad=True)
        with tempfile.TemporaryDirectory() as folder:
            tracker = OptimizationTracker(params, folder, "cpu", light_interval=1)
            for step in (1, 2):
                for p in params.values():
                    p.grad = torch.full_like(p, (-1.0) ** step)
                old = {k: p.detach().clone() for k, p in params.items()}
                tracker.before_update(step, opt)
                opt.step()
                decay, grad, delta = {}, {}, {}
                for (k, p), group in zip(params.items(), opt.param_groups):
                    delta[k] = p.detach() - old[k]
                    decay[k] = -group["lr"] * group["weight_decay"] * old[k]
                    grad[k] = delta[k] - decay[k]
                # 调度器可在更新结束后改变数值，统计必须使用本次 step 时的配置。
                opt.param_groups[0]["weight_decay"] *= 0.5
                tracker.after_update(step, opt)
                rows = {
                    r["layer"]: r
                    for r in tracker.light.pending
                    if r["kind"] == "update" and r["step"] == step
                }
                for layer, row in rows.items():
                    names = [k for k in params if k.startswith(layer + ".")]

                    for key, expected in (
                        ("gradient_update_norm", _layer_norm(grad, names)),
                        ("weight_decay_update_norm", _layer_norm(decay, names)),
                        ("update_norm", _layer_norm(delta, names)),
                        (
                            "wd_ratio",
                            _layer_norm(decay, names) / _layer_norm(grad, names),
                        ),
                    ):
                        self.assertAlmostEqual(
                            row[key], expected, delta=2e-4 * max(1, expected)
                        )
                expected = (
                    torch.cat(list(decay.values())).norm()
                    / torch.cat(list(grad.values())).norm()
                )
                self.assertAlmostEqual(
                    tracker.last_wd_ratio,
                    expected.item(),
                    delta=2e-4 * max(1, expected.item()),
                )

    def test_decay_zero_denominator_and_skipped_grad(self):
        params = {
            k: torch.nn.Parameter(torch.tensor([10.0])) for k in ("used.w", "unused.w")
        }
        opt = torch.optim.AdamW(params.values(), lr=0.1, weight_decay=0.1)
        params["used.w"].grad = torch.zeros(1)
        with tempfile.TemporaryDirectory() as folder:
            tracker = OptimizationTracker(params, folder, "cpu")
            tracker.before_update(20, opt)
            opt.step()
            tracker.after_update(20, opt)
            rows = {
                r["layer"]: r for r in tracker.light.pending if r["kind"] == "update"
            }
            self.assertAlmostEqual(
                rows["used"]["weight_decay_update_norm"], 0.1, places=6
            )
            self.assertEqual(rows["unused"]["weight_decay_update_norm"], 0.0)
            self.assertEqual(rows["used"]["gradient_update_norm"], 0.0)
            self.assertIsNone(rows["used"]["wd_ratio"])
            self.assertIsNone(tracker.last_wd_ratio)

    def test_coupled_adam_reports_only_total(self):
        p = torch.nn.Parameter(torch.tensor([10.0]))
        opt = torch.optim.Adam([p], lr=0.1, weight_decay=0.1)
        p.grad = torch.zeros_like(p)
        with tempfile.TemporaryDirectory() as folder:
            tracker = OptimizationTracker({"a.w": p}, folder, "cpu")
            tracker.before_update(20, opt)
            opt.step()
            tracker.after_update(20, opt)
            row = tracker.light.pending[-1]
            self.assertAlmostEqual(row["update_norm"], (10 - p.item()), places=5)
            for key in ("wd_ratio", "weight_decay_update_norm", "gradient_update_norm"):
                self.assertIsNone(row[key])
            self.assertIsNone(tracker.last_wd_ratio)
