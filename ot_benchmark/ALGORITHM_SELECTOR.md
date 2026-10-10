# OT 算法选择决策树

该模块先在一组覆盖不同维度、稀疏度、噪声和分布形态的 OT 数据集上运行候选算法，按现有综合评分选出每个数据集的冠军，再用数据集特征训练 CART 决策树。

## 图形界面（推荐）

直接双击项目根目录下的 `打开算法选择决策树.bat`。界面可以输入数据参数并一键推荐算法、查看完整决策树图形，以及重新训练模型，全程不需要输入终端命令。

“一键推荐”采用高维轻量模式：只生成长度为 `n` 的源/目标分布，代价矩阵统计量通过固定数量的位置对抽样估算，不会创建 `n × n` 稠密矩阵。预测阶段内存由原来的 `O(n²)` 降为 `O(n)`，并在后台线程运行，窗口不会因高维计算停止响应。完整算法评测仍会按算法需要使用真实代价信息。

## 1. 训练模型

在项目根目录执行：

```powershell
python train_algorithm_selector.py --samples 60 --backend numpy
```

输出位于 `selector_results`：

- `ot_algorithm_selector.joblib`：可直接加载的模型
- `ot_algorithm_selector.json`：模型版本、特征和训练配置
- `selector_training_records.csv`：每个训练数据集的特征与冠军标签
- `selector_evaluation.json`：准确率、混淆矩阵和特征重要性
- `decision_tree_rules.txt`：可直接阅读、可放进论文的决策规则

正式实验建议把样本数提高到至少 100，并更换多个随机种子重复训练。例如：

```powershell
python train_algorithm_selector.py --samples 100 --max-depth 5 --min-leaf 3 --random-state 42
```

## 2. 在主程序中查看推荐

```powershell
python ot_benchmark\main.py --quick --no-ui --selector-model selector_results\ot_algorithm_selector.joblib
```

程序会先打印每个数据集的推荐算法和置信度，随后仍按正常方式完成多算法评测。

## 3. 自动只运行推荐算法

```powershell
python ot_benchmark\main.py --quick --no-ui --no-viz --selector-model selector_results\ot_algorithm_selector.joblib --selector-only
```

这个模式会针对每个数据集只运行决策树推荐的算法，用来减少完整遍历所有算法的耗时。此时单算法得分只是执行摘要，不能当作算法之间的横向排名。

## KNN 下游任务验证

图形界面的“KNN 下游验证”页面会生成带标签的源域与目标域数据，并人为施加旋转、缩放、平移和噪声形成协变量偏移。各 OT 方法先在不使用目标标签的情况下把源域映射到目标域，再使用源域标签训练 KNN，最后仅用目标标签计算 Accuracy 和 Macro-F1。

页面同时报告无 OT 基线、准确率提升、OT+KNN 耗时和峰值内存。结果自动保存到 `selector_results/knn_downstream_results.csv`、`selector_results/knn_downstream_results.json`，并生成本地图表 `selector_results/knn_downstream_comparison.png`。当前适配 EMD、两种 Sinkhorn、Greenkhorn、Low-Rank OT、Sparse Sinkhorn 和 Sliced OT；EMD 默认不勾选，以避免样本较多时精确耦合耗时过长。

当前训练场景覆盖普通一维、极端稀疏、块/分组/层级结构，以及 2D、5D、10D 点云几何。每类场景会轮换使用综合平衡、精度优先、速度优先和内存优先四种优化目标。界面中的几何维度、稀疏结构和优化目标都会直接参与决策树判断。质量不守恒任务由前置分支推荐 Unbalanced OT，避免把不同数学目标的距离错误混排。

## 当前特征

决策树使用 21 个无需先运行 OT 算法即可取得的特征，包括四项优化目标权重、维度、支撑点几何维度、实际稀疏度、噪声水平、源/目标活跃支撑比例、支撑重叠率、连续结构强度、归一化熵、Gini 系数、L1 分布偏移、JS 散度，以及代价矩阵的均值、标准差和局部代价比例。
