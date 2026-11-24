"""Full-population statistics for local tensors and disjoint FSDP shards."""

import csv
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist


class CsvBuffer:
    def __init__(self, directory, channel, interval):
        self.directory = Path(directory)
        self.channel = channel
        self.interval = max(1, int(interval))
        self.pending = []
        self.steps = 0

    def add(self, records):
        self.pending.extend(records)
        self.steps += 1
        if self.steps >= self.interval:
            self.flush()

    def flush(self):
        if self.pending:
            self.directory.mkdir(parents=True, exist_ok=True)
            last = self.pending[-1]["step"]
            path = self.directory / f"{self.channel}_layer_stats_step_{last}.csv"
            temp = path.with_suffix(".csv.tmp")
            columns = list(dict.fromkeys(k for row in self.pending for k in row))
            with temp.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(self.pending)
            os.replace(temp, path)
        self.pending.clear()
        self.steps = 0


def local_tensor(tensor):
    if hasattr(tensor, "to_local"):
        return tensor.to_local().detach()
    return tensor.detach()


def distribution_columns(quantiles):
    names = [
        "finite_numel",
        "nonfinite_numel",
        "norm",
        "rms",
        "min",
        "max",
        "near_zero_ratio",
    ] + [f"bin_{i}_count" for i in range(10)]
    if quantiles:
        names += ["p50", "p90", "p99"]
    return names


def distribution(values, threshold, quantiles):
    """Signed finite-value statistics. Reuse the full sort for CDF counts."""
    names = distribution_columns(quantiles)
    if not values.numel():
        return (
            names,
            values.new_tensor(
                [0, 0]
                + [math.nan] * 5
                + [0] * 10
                + ([math.nan] * 3 if quantiles else []),
                dtype=torch.float32,
            ),
            torch.zeros(12, dtype=torch.int64, device=values.device),
        )
    finite = torch.isfinite(values)
    count = finite.sum()
    safe = torch.where(finite, values, 0.0)
    norm = torch.linalg.vector_norm(safe)
    minimum = torch.where(finite, values, math.inf).amin()
    maximum = torch.where(finite, values, -math.inf).amax()
    nan = values.new_tensor(math.nan)
    lo = torch.where(count > 0, minimum, nan)
    hi = torch.where(count > 0, maximum, nan)
    fractions = torch.arange(1, 10, device=values.device, dtype=torch.float32) / 10
    edges = torch.where(
        (lo < 0) & (hi > 0),
        lo * (1 - fractions) + hi * fractions,
        torch.lerp(lo, hi, fractions),
    )
    if quantiles:
        ordered = torch.where(finite, values, math.inf).sort().values
        cuts = torch.searchsorted(ordered, edges)
    else:
        cuts = torch.stack(
            [(finite & (values < edge.reshape(1))).sum() for edge in edges]
        )
    cuts = torch.where(count == 0, 0, cuts)
    bins = torch.diff(torch.cat([count.new_zeros(1), cuts, count.view(1)]))
    # Constant layers are stored entirely in bin 0.
    constant = torch.zeros_like(bins)
    constant[0] = count
    bins = torch.where(lo == hi, constant, bins)
    result = [
        count,
        values.numel() - count,
        torch.where(count > 0, norm, nan),
        norm / count.sqrt(),
        lo,
        hi,
        (finite & (values.abs() < threshold)).sum() / count,
    ]
    result.extend(bins.unbind())
    if quantiles:
        scaled = (count - 1).clamp_min(0) * count.new_tensor([50, 90, 99])
        left = scaled // 100
        right = (scaled + 99) // 100
        q = torch.lerp(ordered[left], ordered[right], (scaled % 100).float() / 100)
        result.extend(torch.where(count > 0, q, nan).unbind())
    return (
        names,
        torch.stack([x.float() for x in result]),
        torch.cat((count.view(1), (values.numel() - count).view(1), bins)),
    )


class LayerStatistics:
    """All ranks supply the same ordered layer/parameter names, including empty shards."""

    def __init__(self, device, group=None, sharded=False, near_zero=1e-6):
        self.device = torch.device(device)
        self.group = group
        self.world = dist.get_world_size(group) if sharded else 1
        self.rank = dist.get_rank(group) if sharded else 0
        self.threshold = near_zero
        self.history = {}

    def _sizes(self, lengths):
        own = torch.tensor(lengths, device=self.device, dtype=torch.int64)
        if self.world == 1:
            return [lengths]
        sizes = torch.empty(
            self.world * len(lengths), device=self.device, dtype=own.dtype
        )
        dist.all_gather_into_tensor(sizes, own, group=self.group)
        return sizes.view(self.world, -1).tolist()

    def _gather(self, local, sizes, owner):
        if sum(sizes) == sizes[owner]:
            return local
        # Keep the largest shard in place. Only split layers need a collective.
        recv = sizes if self.rank == owner else [0] * self.world
        send = [0] * self.world
        send[owner] = local.numel()
        result = local.new_empty(sum(recv))
        dist.all_to_all_single(result, local, recv, send, group=self.group)
        return result

    @torch.no_grad()
    def collect(self, step, kind, layers, cosine=True):
        """layers maps a module path to ordered (parameter name, local tensor) pairs."""
        lengths = [sum(t.numel() for _, t in parts) for parts in layers.values()]
        sizes = self._sizes(lengths)
        dots, summaries, counts = [], [], []
        for i, (layer, parts) in enumerate(layers.items()):
            local = (
                torch.cat([local_tensor(t).reshape(-1) for _, t in parts])
                if parts
                else torch.empty(0, device=self.device)
            ).float()
            if cosine:
                key = (kind, layer)
                layout = tuple((name, t.numel()) for name, t in parts if t.numel())
                prior = self.history.get(key)
                if prior is not None and layout == prior[0]:
                    old = prior[1].float()
                    dots.append(
                        torch.stack(
                            (
                                old.dot(local),
                                old.dot(old),
                                local.dot(local),
                                local.new_zeros(()),
                            )
                        )
                    )
                else:
                    invalid = bool(layout or (prior and prior[0]))
                    dots.append(local.new_tensor([0, 0, 0, int(invalid)]))
                # cat owns its storage, independent of zero_grad/optimizer in-place writes.
                dtype = (
                    parts[0][1].dtype
                    if parts and all(t.dtype == parts[0][1].dtype for _, t in parts)
                    else torch.float32
                )
                self.history[key] = (layout, local.to(dtype))
            chunks = [row[i] for row in sizes]
            owner = chunks.index(max(chunks))
            full = self._gather(local, chunks, owner)
            if self.rank == owner:
                names, stats, integers = distribution(full, self.threshold, cosine)
                summaries.append(stats)
                counts.append(integers)
            else:
                summaries.append(local.new_zeros(20 if cosine else 17))
                counts.append(local.new_zeros(12, dtype=torch.int64))
            del full
        if not layers:
            return []
        if cosine:
            packed = torch.stack(dots)
            if self.world > 1:
                dist.all_reduce(packed, group=self.group)
        output = torch.stack(summaries)
        integer_counts = torch.stack(counts)
        if self.world > 1:
            # Exactly one rank populated each row. Move only layer summaries.
            dist.all_reduce(output, group=self.group)
            dist.all_reduce(integer_counts, group=self.group)
        if self.rank != 0:
            return []
        names = distribution_columns(cosine)
        if cosine:
            dot, old_sq, new_sq, invalid = packed.unbind(1)
            denom = old_sq.sqrt() * new_sq.sqrt()
            similarity = torch.where(
                (invalid == 0) & (denom > 0), dot / denom, math.nan
            )
            output = torch.cat((output, similarity[:, None]), dim=1)
            names = names + ["cosine"]
        records = []
        count_keys = ["finite_numel", "nonfinite_numel"] + [
            f"bin_{i}_count" for i in range(10)
        ]
        for layer, values, ints in zip(
            layers, output.tolist(), integer_counts.tolist()
        ):
            row = {"step": step, "layer": layer, "kind": kind}
            row.update(
                (k, v if math.isfinite(v) else None) for k, v in zip(names, values)
            )
            row.update(zip(count_keys, ints))
            row["has_nonfinite"] = bool(row["nonfinite_numel"])
            records.append(row)
        return records
