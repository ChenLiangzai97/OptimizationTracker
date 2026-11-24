"""Build the existing v1.1.5 fallback TB record from local gradient views."""

import torch
import torch.distributed as dist


@torch.no_grad()
def report_fallback(monitor, tracker, gradients, step, tb, logger):
    group_count = monitor._num_groups
    gids = monitor._param_primary_gid
    device = tracker.stats.device
    heavy = step % monitor._heavy_k == 0
    groups = [[] for _ in range(group_count)]
    seen = set()
    for parts in gradients.values():
        for name, tensor in parts:
            seen.add(name)
            if tensor.numel():
                groups[gids[name]].append(tensor.reshape(-1))
    # 每行对应一个旧 TB 主组。前两列是扫描数和非有限数；重步再追加
    # 13 个绝对值对数桶、近零数、极值数。元素计数在跨卡求和前保持 int64。
    counts = torch.zeros(
        (group_count, 17 if heavy else 2), device=device, dtype=torch.int64
    )
    squares = torch.zeros(group_count, device=device)
    maxima = torch.zeros_like(squares)
    for i, parts in enumerate(groups):
        if not parts:
            continue
        grad = torch.cat(parts).float()
        finite = torch.isfinite(grad)
        absolute = torch.where(finite, grad.abs(), 0.0)
        counts[i, 0] = grad.numel()
        counts[i, 1] = (~finite).sum()
        squares[i] = absolute.dot(absolute)
        maxima[i] = absolute.max()
        if heavy:
            logs = absolute.clamp_min(1e-38).log10()
            cdf = torch.stack([(logs <= edge).sum() for edge in range(-8, 4)])
            counts[i, 2:15] = torch.diff(
                torch.cat((cdf.new_zeros(1), cdf, counts[i, :1]))
            )
            # 求范数时已把 NaN/Inf 置零，它们也因此进入首桶。扣掉这部分，
            # 让直方图计数和旧监控器传给 _attach_heavy 的有限元素分母一致。
            counts[i, 2] -= counts[i, 1]
            counts[i, 15] = (finite & (absolute < monitor._near_zero)).sum()
            counts[i, 16] = (finite & (absolute > monitor._extreme)).sum()
    if tracker.stats.world > 1:
        works = [
            dist.all_reduce(x, op=op, group=tracker.stats.group, async_op=True)
            for x, op in (
                (counts, dist.ReduceOp.SUM),
                (squares, dist.ReduceOp.SUM),
                (maxima, dist.ReduceOp.MAX),
            )
        ]
        for work in works:
            work.wait()
    if not tracker.write:
        return
    totals = counts.tolist()
    numels = [row[0] for row in totals]
    nf = [row[1] for row in totals]
    sq, mx = torch.stack([squares, maxima]).tolist()
    missing = sum(
        p.requires_grad and name not in seen
        for name, p in monitor.monitored_params.items()
    )
    monitor._last_numels = numels
    rec = monitor._build_result(step, sq, nf, mx, -1, -1, missing, heavy, numels)
    if heavy:
        packed = (
            [0] * group_count
            + [x for row in totals for x in row[2:15]]
            + [row[15] for row in totals]
            + [row[16] for row in totals]
        )
        monitor._attach_heavy(
            rec, packed, sq, nf, [n - f for n, f in zip(numels, nf)], False
        )
    monitor._report(step, rec, tb, logger)
