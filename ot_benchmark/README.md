# OTBench

OTBench 是一个面向高维、稀疏数据的最优传输（Optimal Transport, OT）算法评测平台。它使用统一接口评估算法的距离误差、运行时间、CPU 内存与成功率，并可生成可复现实验记录和报告。

## 功能

- 合成随机、块状、组状、层级稀疏与协变量偏移数据集
- MovieLens 100K、UCI Bag-of-Words 等真实数据加载器
- 内置 EMD、Sinkhorn、Greenkhorn、Sliced OT、Unbalanced OT、Low-Rank OT、Gaussian OT、Factored OT、Sparse Sinkhorn 和 GPU Sinkhorn
- 多随机种子、多次重复运行、HTML 报告、失效边界与敏感性分析
- 外部算法插件加载、决策树算法推荐和 KNN 下游验证

## 安装

需要 Python 3.10 或更高版本。

```powershell
cd ot_benchmark
python -m pip install -e .
```

GPU 后端为可选依赖：

```powershell
python -m pip install -e ".[gpu]"
```

## 快速开始

从本仓库的 `ot_benchmark` 目录运行：

```powershell
# 查看交互式评测器参数
python -m ot_benchmark.main --help

# 快速运行两组小型数据（不生成图表）
python -m ot_benchmark.main --quick --no-ui --no-viz

# 固定清单的可复现实验入口
python run_benchmark_v1.py --quick --no-sweep
```

安装后也可以使用命令行入口：

```powershell
ot-benchmark --quick --no-ui --no-viz
```

完整固定场景与结果字段见 `configs/benchmark_v1.yaml`，实验运行器的参数见：

```powershell
python run_benchmark_v1.py --help
```

## 项目结构

```text
ot_benchmark/
├── ot_benchmark/            # 可安装 Python 包
│   ├── algorithms/          # 内置算法、注册表、插件加载器
│   ├── backend/             # NumPy、PyTorch、CuPy 后端
│   ├── core/                # 配置、类型、日志与并行基础设施
│   ├── data_generator.py    # 数据集与支撑点几何生成
│   ├── benchmark_engine.py  # 计时、内存、评分与统计
│   ├── real_data.py         # 真实数据加载器
│   └── main.py              # 交互式评测命令行
├── configs/                 # 固定场景和预设配置
├── external_algorithms/     # 第三方算法插件目录
├── tests/                   # 单元测试
└── run_benchmark_v1.py      # 可复现实验批处理入口
```

## 输出与解释

`run_benchmark_v1.py` 的输出目录包含原始运行记录、按 seed 汇总和跨 seed 汇总、环境信息、实验协议以及 HTML 报告。EMD 精确解仅在可承受的小场景中作为数学真值；较大场景的 Sinkhorn 结果是参考值，不应标注为精确真值。

## 外部插件

将插件放在 `external_algorithms/` 下即可被主程序发现。单文件插件需要提供：

```python
def run_ot(source, target, cost_matrix, **kwargs) -> float:
    ...
```

详情见 `external_algorithms/README.md`。
