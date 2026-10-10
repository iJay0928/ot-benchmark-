# 模块 A：不同数据条件下的成熟算法对照

独立目录、独立种子，比较 POT 原生 `sinkhorn_knopp`、`sinkhorn_stabilized`、`sinkhorn_epsilon_scaling`。不修改主项目算法，不训练推荐器。

## 安装与完整复现

在仓库根目录执行 PowerShell，使用 **Python 3.12**（本轮锁定的 SciPy 1.18.0 要求 Python≥3.12）。任务书建议 3.11，但此模块实际使用并验证的是 3.12，不能将这一依赖锁直接用于 3.11。

```powershell
python -m venv .venv_module_a
.\.venv_module_a\Scripts\python.exe -m pip install -r ot_benchmark/member_modules/module_a/requirements.txt
$py = '.\.venv_module_a\Scripts\python.exe'
$mod = 'ot_benchmark/member_modules/module_a'
$out = "$mod/experiment_reproduction"
& $py -m pytest "$mod/test_module_a.py" "$mod/test_reporting.py" -q
& $py "$mod/generate_inputs.py" --output $out
& $py "$mod/run_baselines.py" --output $out
& $py "$mod/audit_results.py" --output $out
& $py "$mod/summarize_results.py" --output $out
& $py "$mod/plot_results.py" --output $out
& $py "$mod/package_evidence.py" --output $out
```

输出目录必须新建，禁止覆盖原始实验。中断后可用相同 runner 命令继续，已完成日志不会重复求解；代码或协议改变后必须新冻结目录。正式计时串行，不要同时另跑求解器。最坏求解预算 432×30 秒，加启动/热身和写盘开销；实际时间见原始记录。

`--limit` 只用于开发/断点检查，正式交付必须有 432 条。正式种子 210101/210102/210103；开发种子 210001。结果由脚本生成，不提供“必定加速”的结论。

## 文件与证据

- `experiment_20261010/protocol.json`、`task_order.json`：正式求解前的配置与随机顺序。
- `report.md`：完整条件对照、达标分母、同质量配对耗时与限制。
- `experiment_20261010/results/summary.csv`：每次调用一行；`conditions.csv`：三次中位数、最小/最大耗时与不稳定状态。
- `environment.json`：实际 Python/NumPy/SciPy/POT 版本、硬件、POT 源码 SHA256。
- `results/audit.json`：24 输入逐元素重建、432 任务身份及计划/对偶复算。
- `results/figures/`：达标矩阵、共同达标配对时间、失败类型分解；SVG/PDF/300DPI PNG。
- `experiment_20261010_evidence.zip`：输入、全部原生轨迹、每次原始日志和返回计划；由 package_evidence.py 生成，原始数组与长日志不加入 Git。

与公共诊断模块的唯一联系是按文件路径加载 `transport_diagnostics.py`，不要求安装整个项目，也不修改公共函数。

## 质量和计时

固定最终 ε 的熵正则目标；共同数值 gap∈[−1e-10,1e-8]，舍入边缘 L1≤1e-12。两种返回（原生计划与最终势重建）均检查有限、非负及重建一致，禁止裁剪无效势。

计时含求解与共同外部验收，不含进程启动、导入、热身、输入生成或写盘；后者单列。超时可强制终止，记为截尾而不是 30 秒完成时间。延续实际内层迭代数未知时为 null。审计通过不等于所有算法成功。

本子实验仅代表一维、正质量、双簇的平衡 OT 数据族。它不证明高维/稀疏/真实应用结论，不是新的求解算法。
