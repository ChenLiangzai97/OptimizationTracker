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
    finally:
        dist.destroy_process_group()


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
