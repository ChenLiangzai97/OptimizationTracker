# OptimizationTracker

在 optimizer step 20、40、60…采集梯度和 Adam 状态，每 200 步追加更新后的参数统计。重步的轻通道只执行一次。普通步跳过本组件的张量运算和通信。

## 接入训练仓

将 `gradient_health_tracking_callback.py`、`optimizer_tracker.py`、`gradient_statistics.py`、`fallback_reporting.py` 放在现有 `gradient_health_monitor.py` 旁。将已注册的 `GradientHealthMonitorCallback` 替换为 `GradientHealthTrackingCallback`，保留原先的注册位置。该适配器依赖训练仓现有的 v1.1.5 组件及 UTP。独立统计器和基准只依赖 PyTorch。

沿用 `GRADIENT_HEALTH_MONITOR=1` 和原 `logger.grad` 选参配置。`cur_it` 采用各回调一致的 optimizer 步号，0 跳过。训练端应在同步梯度可读时触发 pre-optimizer，在该次更新成功后触发 optimizer-step。独立调用方式如下。

```python
tracker.before_update(step, optimizer)
optimizer.step()
tracker.after_update(step, optimizer)
# 训练结束时
tracker.close()
```

| 环境变量 | 默认 | 含义 |
|---|---:|---|
| GRADIENT_HEALTH_LIGHT_K | 20 | 梯度、优化器状态和更新量的统计间隔 |
| GRADIENT_HEALTH_PARAMETER_K | 200 | 参数统计间隔，取轻步间隔的正整数倍 |
| GRADIENT_HEALTH_LIGHT_FLUSH | 5 | 每累计几次轻通道记录写一个文件 |
| GRADIENT_HEALTH_HEAVY_FLUSH | 1 | 每累计几次参数统计写一个文件 |
| GRADIENT_HEALTH_CSV_DIR | output_dir/gradient_health | CSV 输出目录 |
| GRADIENT_HEALTH_NEAR_ZERO | 1e-6 | 近零判断的绝对值阈值 |

沿用 v1.1.5 的 TB 标签、旧组名、对数分桶和 TB 写入频率门控。在新统计步上应用旧门控，出点位于两者的交集。FSDP1 Hook 继续异步归约并在 optimizer 后消费已有 TB 指标。CSV 的全量梯度统计在 optimizer 前完成，分布式通信和排序可能延长统计步。原先单独注册的 per-param GradientMonitor 仍由训练仓控制。

## CSV 口径

rank 0 用 UTF-8 BOM 写入 `light_layer_stats_step_<末步>.csv` 和 `heavy_layer_stats_step_<末步>.csv`。每行保留实际采集步、模块完整路径和对象类型。积累多个统计步后批量落盘，正常结束刷出尾批。文件使用同目录临时文件和原子替换。新训练任务宜使用独立目录，同目录重跑相同末步会替换同名文件。

- 层包含直接注册在该模块上的所有受监控参数，包括 weight 和 bias。梯度取 pre-optimizer 实际可读的值，exp_avg、exp_avg_sq 和重步参数取更新后的值。
- 范数、RMS、最值、p50/p90/p99、直方图使用有限元素。记录有限及非有限数量，非有限值单独标记。近零比例按绝对值严格小于阈值计算。空集合指标留空。CSV 分桶及分位数使用带符号的数值。
- 每层每步用全局 min/max 等宽分 10 桶，左闭右开，末桶包含最大值。常量层放入首桶。桶计数为整数。分位数对全量 FP32 统计值排序并线性插值，复用排序结果进行 CDF 分桶。参数通道只做阈值计数，无需排序。
- cosine 比较相邻两次轻通道，按本地分片保存全部元素，快照保留源张量精度（混合精度层统一用 FP32），点积和范数用 FP32。首次采集、参与的参数段变化、零范数或存在非有限值时留空。重启后的首个统计步重新建立快照。
- 完整模块的更新前参数范数作为相对更新量分母，更新向量按参数组学习率和更新后的 Adam 一二阶矩推算。支持标准 Adam/AdamW 的 epsilon 放置、偏差修正和 AMSGrad。假设 weight decay 为零。未收到梯度的已注册参数贡献零更新，缺失所需优化器状态或更新前范数为零时相应字段留空。CSV 的推算量是优化器算术更新；低精度权重写回舍入会影响实测差值。自定义 BF16AdamW 需核实其 state 字段和更新规则。
- 历史 TB 的 opt_delta_w_rms 沿用原先未做偏差修正的自适应项；CSV 使用上述更新口径。CSV 记录每层，TB 延续原有主组聚合。

## 分片与开销

DDP 仅 rank 0 运行新增 CSV 采集。FSDP 在 shard group 内合并，全层位于单卡时就地统计。跨卡层由持有最多元素的 rank 临时收齐该层当前对象的分片。各层的结果打包归约，原始梯度和状态不传到 CPU。该路径依赖后端支持不等长、含零长度分片的 all_to_all_single。HSDP 使用 shard group，FSDP2 支持一个分片维度及复制维度。各 rank 需要一致的模块名称及顺序。FSDP1 支持原参数及能通过现有 handle 元数据关联的 flat optimizer state。

显存预算包含完整 g/m/v 本地历史、单层 gather 和排序工作区，以及已有 TB Hook 保留的梯度快照。在三个对象均为 FP32 的情况下，CSV 历史每个本地元素约 12 字节。实际设备峰值还受最大单层和 allocator 影响。

## 验证

从仓库的上一级目录运行：

```bash
python -m unittest -v OptimizationTracker.test_tracker
python -m OptimizationTracker.benchmark --mode baseline --device npu
python -m OptimizationTracker.benchmark --mode full --device npu
```

基准以两个独立进程对比相同的合成训练循环，包含预热，测量 220 步，输出普通、轻、重步耗时以及对应的设备 allocated 峰值。baseline 关闭统计，full 启用新增 CSV 统计；生产 TB、FSDP 通信和 UTP 调度的开销需要在实际任务中用关闭、原 v1.1.5、新回调三组对照。比较完整 200 步周期时分别计入 190 个普通步、9 个轻步、1 个重步。

CPU 测试覆盖真实 AdamW 更新、flat state 分片映射、两 rank 不均匀分片、NaN/Inf、空层、常量层及批量落盘。缺少 TCP socket 权限的环境可用 `TRACKER_TEST_THREADED_PG=1` 选择 PyTorch CPU 测试进程组。
