# OptimizationTracker 代码文档

当前版本 **v1.0.2**。

## 项目背景与版本

OptimizationTracker 的初始版本为 **v1.0.0**。它为训练仓已有的 GradientHealthMonitor v1.1.5 增加逐层 CSV 记录，沿用原有 TensorBoard 指标与展示形式。训练仓的 UTP 和旧回调由接入方提供。统计核心可以单独用 PyTorch 运行。

## 架构与代码结构

| 文件和入口 | 职责 | 这样划分的原因 |
|---|---|---|
| `gradient_health_tracking_callback.py` · `GradientHealthTrackingCallback` | 接入训练钩子，读取选参配置、设备、分片组和优化器，派发采集、写 TB | 将 UTP 与旧组件的依赖限制在适配层 |
| `optimizer_tracker.py` · `OptimizationTracker` | 管理采集步、模块分组、更新前后配对和两个 CSV 缓存 | 梯度、学习率、更新前权重和更新后状态分属不同时间点 |
| `gradient_statistics.py` · `LayerStatistics`、`CsvBuffer` | 处理分片、分布、历史快照，缓存和写出 CSV | DDP 和 FSDP 共用统计口径；写盘频率可独立于采集频率 |
| `fallback_reporting.py` · `report_fallback` | 从本地梯度视图生成旧 TB 所需的组统计 | CSV 逐层使用有符号的 10 桶，旧 TB 使用梯度绝对值的 13 个对数桶 |
| `test_tracker.py`、`benchmark.py` | 验证数值、分片和文件行为；对比合成训练开销 | 正确性和性能分别验收 |

## 核心逻辑

1. **选层与门控。** 用每个可训练参数的直接父模块完整路径分组，同层 weight 和 bias 共享一条统计记录。默认每 20 个 optimizer step 激活一次，第 200 步同时开启参数通道。相邻两次轻通道可跨过重步，例如第 220 步比较第 200 步。
2. **更新前。** `before_update` 读取已同步的梯度、当步学习率和各层权重平方和。保存权重平方和就足以计算相对更新量的分母。flat 参数通过原参数对应的偏移和长度切片，避免混入同一 flat buffer 中的其他层。
3. **更新后。** `after_update` 配对同一步，读取一阶矩和二阶矩，用各参数组学习率、偏差修正和 epsilon 推算更新，再将各段更新平方和合并为层范数。相对更新量为该范数除以更新前权重范数，假定 weight decay 为零。分母为零时比例为空。重步另采集更新后的参数。
4. **分布和历史。** 每层对有限值计算有符号 min/max、RMS 和 10 桶分布，NaN/Inf 单独计数。梯度和矩的分位数对全量值排序并插值，复用排序结果计算桶计数。参数通道无需分位数，直接计算各阈值的累计数量。cosine 在各 rank 保存上一统计步的本地快照，归约点积与平方和；首次采集或零范数留空。
5. **跨卡与落盘。** 单卡持有整层时就地计算。层跨分片时，将本层当前统计对象集中到持有最多元素的 rank，以获得统一的桶边界和全量分位数。各层的结果打包归约，rank 0 按通道缓存 CPU 记录，到期按缓存末步命名文件，`close()` 刷出尾批。全量快照占用常驻显存，排序工作区带来统计步峰值。

## 基础 API 与场景样例

从仓库的上一级目录运行下面的 Python 示例。它将间隔缩短到 2/4，便于用四次真实更新观察两种 CSV。

```python
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from OptimizationTracker.optimizer_tracker import OptimizationTracker

model = torch.nn.Sequential(torch.nn.Linear(2, 1))
opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)

with TemporaryDirectory() as output:
    tracker = OptimizationTracker(
        dict(model.named_parameters()), output, "cpu",
        light_interval=2, heavy_interval=4, light_flush=2, heavy_flush=1,
    )
    for step in range(1, 5):
        opt.zero_grad(set_to_none=True)
        model(torch.ones(1, 2)).sum().backward()
        tracker.before_update(step, opt)
        opt.step()
        tracker.after_update(step, opt)
    tracker.close()

    for path in sorted(Path(output).glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        print(path.name, len(rows), sorted({r["step"] for r in rows}))
```

预期输出：

```text
heavy_layer_stats_step_4.csv 1 ['4']
light_layer_stats_step_4.csv 8 ['2', '4']
```

轻通道在第 2、4 步各记录 gradient、exp_avg、exp_avg_sq、update 四行。重通道第 4 步为模块 `0` 记录一行 parameter。生产使用默认 20/200 间隔，注册方式见[接入说明](../../../../README.md)。

## 测试用例与验收

| 测试项 | 测试内容 | 期望输出 |
|---|---|---|
| `test_full_cycle` | 对照真实 AdamW 更新前后的权重，同时检查首次缓存、批量写盘及结束刷盘 | 相对更新量与参考值一致到小数点后六位；第 40 步写 8 行，相同梯度 cosine 约为 1；关闭后共 3 个文件 |
| `test_flat_optimizer_maps_unaligned_layers` | 两个逻辑层共用 flat 参数及优化器状态 | 第二层梯度最小值为 5，一阶矩最大值约为 0.7；两层更新比例分别匹配各自切片 |
| `test_shards` | 两 rank 分布不均匀，包含 NaN，另有一层仅在单 rank 上非空 | 全局 p50 为 1、p99 约为 9.82、桶计数合计 3；另一层常量进首桶 |
| `test_nonfinite_constant_and_missing` | 含 Inf/NaN 的常量层、空层和全 NaN 层 | 仅有限零值进入首桶，非有限计数为 2；空层 RMS 与全 NaN 层 max 均为空 |

从仓库上一级运行 `python -m unittest -v OptimizationTracker.test_tracker`。当前环境使用 `TRACKER_TEST_THREADED_PG=1` 运行双 rank 用例。生产 UTP、真实多进程 HCCL 和自定义 BF16AdamW 仍需在训练环境验证。

## 版本记录

| 版本 | 修改时间（北京时间） | 代码 commit | 修改内容、demo 与验证 |
|---|---|---|---|
| v1.0.0 | 2026-11-24 15:17 | `c7a0cf77fca128a339b45f719d8fad967a4a28aa` | 初版实现逐层采集、分片统计、更新量推算、CSV 缓存和 TB 接入。上方四步 AdamW demo 演示两类落盘；本版包含上表四项 CPU 测试。 |
| v1.0.1 | 2026-11-24 15:38 | `e9f476e9d0f2deade72e5173715ebb499d8a271f` | 补齐更新时序、flat 参数切片、分片通信与分位数、快照生命周期、旧 TB 分母及 CSV 写入的源码注释；上方四步 AdamW demo 输出与预期一致，四项回归测试全部通过。 |
| v1.0.2 | 2026-11-24 15:42 | `2c97cf25182559cc19d01e4b27c5a986aea50280` | 在采集路径旁补充分片切片、跨卡收集、全量分位数插值和同层更新比例的手算示例，标明第 200 步回调与快照时序。上方四步 AdamW demo 输出符合预期，4 项存量测试全部通过。 |
