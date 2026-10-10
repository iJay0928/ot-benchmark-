# 第一轮数据修正与验证

本轮属于工程和数据语义修正，不代表已经完成论文主实验或证明专利创新性。

## 检查结果

- 5 项新增数据合同回归检查与 11 项原有科学正确性检查通过，共 16 项。
- 五个 v0.2.0 内部候选数据子集通过发布校验。
- 全量 pytest 尚未运行：本机 Python 环境没有 pytest，尝试安装时网络代理连接失败。使用标准库 unittest 执行上述不依赖 pytest fixture 的检查，未伪造或跳过失败断言。
- 快速评测完成 4 个场景、14 个算法配置、每场景 1 个数据种子、2 次重复，共 112 条记录，均为 success；它用于验证入口和输出，不用于论文结论。success 仅表示正常返回有限值。

## 复现命令

在 `ot_benchmark` 工作目录运行：

```powershell
python tools/run_contract_checks.py
python tools/build_dataset_release.py --output dataset_release/hd_sparse_ot_v0.2.0
python tools/verify_dataset_release.py dataset_release/hd_sparse_ot_v0.2.0
python run_benchmark_v1.py --quick --no-sweep --output validation/2026-10-05/benchmark_smoke_rerun
```

新增测试验证了：CSR 完整往返、结构变化而权重集合不变、小尺寸和极端稀疏率、块连续性、输入比例语义和几何维度。v0.2.0 的结构是支撑索引布局，尚未证明对应真实应用中的几何结构。

## 限制

CSR 无损转换不保证节省内存；稠密代价矩阵转 CSR 可能占更多空间。现有稀疏求解器仍从完整矩阵构造截断核，不能据此声称整个流程为线性内存。

分布生成仍把长尾形状参数与稀疏率耦合，且噪声可增加非零元素。因此元数据明确记录实际稀疏率，后续控制变量实验还需拆分这些因素。

现有参考值校验可拦截缺少 OT 依赖导致的 L1 fallback，但尚未全面检验参考解收敛残差。完整算法正确性与统计验证留待下一轮。
