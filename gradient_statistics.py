"""Full-population statistics for local tensors and disjoint FSDP shards."""

import csv
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist


class CsvBuffer:
    """缓存单个通道的 CPU 明细；interval 表示几次采集写一个文件。"""
    def __init__(self, directory, channel, interval):
        self.directory = Path(directory)
        self.channel = channel
        self.interval = max(1, int(interval))
        self.pending = []
        self.steps = 0

    def add(self, records):
        """一次调用对应一个统计步，每条明细保留其原始步号。"""
        self.pending.extend(records)
        self.steps += 1
        if self.steps >= self.interval:
            self.flush()

    def flush(self):
        """按缓存末步命名文件。原子替换成功后才清空缓存，写入失败时可重试。"""
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
    """取脱离计算图的本地视图；DTensor 在此只暴露本 rank 的分片。"""
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
    """在持有整层数据的 rank 上统计有限值，返回列名、浮点摘要和精确整数计数。

    10 个桶按本步带符号的最小值、最大值等宽划分。计数额外保留 int64，
    避免大层的元素数在 FP32 汇总中发生舍入。threshold 是绝对值近零阈值。
    """
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
        # 非有限值放在有序数组末尾。searchsorted 的左插入点给出严格小于
        # 每条边界的数量，相邻累计数之差就是桶计数，末桶包含最大值。
        ordered = torch.where(finite, values, math.inf).sort().values
        cuts = torch.searchsorted(ordered, edges)
    else:
        cuts = torch.stack(
            [(finite & (values < edge.reshape(1))).sum() for edge in edges]
        )
    cuts = torch.where(count == 0, 0, cuts)
    bins = torch.diff(torch.cat([count.new_zeros(1), cuts, count.view(1)]))
    # 常量层的九条边界完全重合，统一把有限元素放在首桶，便于从 CSV 识别这种退化分布。
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
        # 排名只覆盖有限值所在的前缀。保留整数商和余数，最后再线性插值。
        scaled = (count - 1).clamp_min(0) * count.new_tensor([50, 90, 99])
        # 有限值若为 [-5, 1, 10]，p90 落在从 0 开始的第 1.8 位。
        # 左右端点是 1 和 10，插值得 8.2。末尾为非有限值预留的位置不参与排名。
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
        # 同步取回的是 rank × 层的长度表；后面的变长通信需要主机侧 split sizes。
        return sizes.view(self.world, -1).tolist()

    def _gather(self, local, sizes, owner):
        if sum(sizes) == sizes[owner]:
            return local
        # owner 是分片最大的 rank。每个 rank 只向它发送本地段，其他 rank
        # 的接收长度全为 0；临时接收缓冲只需容纳当前层的这一种统计对象。
        # 例：rank 0/1 分别持有 2/5 个元素，owner=1。两卡的 send 分别是
        # [0, 2]、[0, 5]；rank 1 的 recv=[2, 5]，得到完整 7 个值后才能排全局分位数。
        recv = sizes if self.rank == owner else [0] * self.world
        send = [0] * self.world
        send[owner] = local.numel()
        result = local.new_empty(sum(recv))
        dist.all_to_all_single(result, local, recv, send, group=self.group)
        return result

    @torch.no_grad()
    def collect(self, step, kind, layers, cosine=True):
        """采集一个 step、一个对象类型的逐层统计，返回本进程组 rank 0 的明细。

        layers 按各 rank 约定的顺序给出层名及本地参数段。分位数需要全层数据；
        cosine 只归约各分片的点积和平方和，历史快照始终按分片保存。
        """
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
                # 即使两次扁平向量等长，参数也可能缺梯度或参与次序不同。
                # 先核对名称和段长，避免对不对应的坐标计算方向相似度。
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
                # local 来自本轮 cat，拥有独立存储；即使 to(dtype) 直接返回原张量，
                # 下一次 zero_grad 或优化器原地修改 exp_avg，也不会改写这份历史。
                # 以默认间隔为例，第 200 步先与 180 步比较，再留下供 220 步使用的快照。
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
            # 每一层只有 owner 填写真实统计行，其他 rank 填零。例如本层 owner=1，
            # 求和会把 rank 1 的整行传到输出端；此处传输的数据量仅取决于层数和列数。
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
        # 到这里才把整批逐层摘要搬回主机；整数桶计数覆盖浮点摘要中的计数列。
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
