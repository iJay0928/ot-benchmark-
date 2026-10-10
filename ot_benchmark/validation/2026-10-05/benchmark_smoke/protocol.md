# OTBench Benchmark v1.0 — 实验协议 (修订版)

## 1. 数据产生
根据 `configs/benchmark_v1.yaml` 中的 4 个场景配置生成数据集。
相同场景、相同 dataset_seed 必定生成完全一致的数据。
- `dimension` 表示分布坐标/质量向量长度；`geometry_dimension` 表示 OT 支撑点的几何维度。
- 合成高维场景必须显式设置 `geometry_dim > 1`；未设置时为兼容旧结果使用 1D index geometry。

## 2. 种子体系
- **dataset_seed**: 决定数据集内容。每场景至少 3 个 seed（quick 模式用 1 个）。
- **algorithm_seed**: 仅供算法内部随机过程使用。确定性算法为 ""。
- **repetition_index**: 同一 (场景, seed, 算法) 组合的第几次重复运行。

## 3. 实验结构
场景定义
  ├─ dataset_seed A
  │   ├─ 算法 1, 重复 1～N
  │   ├─ 算法 2, 重复 1～N
  │   └─ ...
  ├─ dataset_seed B
  │   └─ ...
  └─ ...

## 4. Ground Truth / Reference / Consensus

| 类型 | 含义 | 何时可用 |
|------|------|---------|
| ground_truth_distance | EMD 精确解 (数学真值) | 小场景 (有效维度 ≤ 300) |
| reference_distance | Sinkhorn(reg=0.01) 高精度参考解 | 中大规模场景 |
| consensus_deviation | 算法结果偏离候选算法中位数的程度 | 始终可用 (辅助诊断) |

- ground_truth_error: 相对 ground_truth_distance 的误差。无精确真值时为 ""。
- reference_error: 相对 reference_distance 的误差。无参考解时为 ""。
- consensus_deviation: 辅助诊断指标，标题必须写 "共识偏差"。
- 有 exact 时排行榜优先使用 ground_truth_error。
- 无 exact/ reference 时不显示 "真实误差"。
- 综合评分受参考类型和权重影响，不代表算法绝对优劣。

## 5. 时间测量
- 使用 `time.perf_counter()` 高分辨率单调时钟。
- GPU 运算通过 PyTorch CUDA Events 同步。
- runtime_seconds 包含算法执行的全部时间（归一化、迭代、结果提取）。
- 不包含数据生成和代价矩阵构建时间。

## 6. 内存测量
- CPU 内存: `psutil.Process().memory_info().rss` 进程 RSS 增量。
- GPU 显存: `torch.cuda.memory_allocated()` 额外记录（如有 CUDA）。
- peak_memory_mb = max(before_rss, after_rss)，每 2ms 采样。
- **当前 peak_memory_mb 仅反映 CPU 进程 RSS 增量，不含 GPU 显存。**

## 7. 成功 / 失败判定
| status | 含义 |
|--------|------|
| success | 算法正常返回有限值 |
| timeout | 运行时间超限 |
| out_of_memory | 内存不足 |
| numerical_error | 数值错误 (NaN/Inf/singular) |
| dependency_error | 依赖缺失 |
| not_applicable | 算法不适用该场景 |
| unknown_error | 未分类错误 |

## 8. 统计规则
### 同一 seed 内统计
- 时间、内存、距离的均值/标准差仅基于成功运行。
- 成功样本 < 2 时标准差为空。
- P95 运行时间对成功运行取 95 分位。

### 跨 seed 场景统计
- 先计算每 seed 的均值，再跨 seed 计算均值和标准差。
- 区分同一数据上的运行波动 (within-seed) 与不同数据间的差异 (cross-seed)。

## 9. 算法参数
所有算法使用 registry 注册的默认参数。参数详情见 `algorithms/registry.py`。

---

*本文档由 run_benchmark_v1.py 自动生成 · 修订版*
