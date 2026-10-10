"""Generate benchmark labels and train the OT algorithm decision tree."""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ot_benchmark"))

from algorithm_selector import (
    collect_training_records,
    generate_training_datasets,
    save_selector_model,
    save_training_records,
    select_training_algorithms,
    selector_weight_profiles,
    train_decision_tree,
)
from algorithms.registry import ALGORITHMS
from backend.base import get_backend


def main():
    parser = argparse.ArgumentParser(description="训练 OT 算法选择决策树")
    parser.add_argument("--samples", type=int, default=60, help="训练数据集数量（默认 60）")
    parser.add_argument("--max-depth", type=int, default=5, help="决策树最大深度")
    parser.add_argument("--min-leaf", type=int, default=2, help="叶节点最少样本数")
    parser.add_argument("--test-size", type=float, default=0.25, help="测试集比例")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--backend", choices=["numpy", "torch", "cupy", "auto"], default="numpy")
    parser.add_argument("--output-dir", default="selector_results", help="模型和报告输出目录")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    algorithms = select_training_algorithms(ALGORITHMS)
    print("参与生成标签的算法：")
    for name in algorithms:
        print(f"  - {name}")

    datasets = generate_training_datasets(args.samples, args.random_state)
    records = collect_training_records(
        datasets, algorithms, backend=get_backend(args.backend), progress=True,
        weight_profiles=selector_weight_profiles(),
        evidence_dir=output / "case_evidence",
    )
    records_path = save_training_records(output / "selector_training_records.csv", records)
    model, evaluation = train_decision_tree(
        records,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_leaf,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    config = vars(args) | {"algorithms": list(algorithms), "valid_records": len(records)}
    model_path = save_selector_model(
        output / "ot_algorithm_selector.joblib", model, evaluation, config
    )
    report_path = output / "selector_evaluation.json"
    report_path.write_text(
        json.dumps(asdict(evaluation), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rules_path = output / "decision_tree_rules.txt"
    rules_path.write_text(evaluation.rules, encoding="utf-8")

    print("\n训练完成")
    print(f"  有效记录: {len(records)}")
    print(f"  训练/测试: {evaluation.train_size}/{evaluation.test_size}")
    print(f"  测试准确率: {evaluation.accuracy:.2%}")
    print(f"  平衡准确率: {evaluation.balanced_accuracy:.2%}")
    print("  标签分布:")
    for name, count in evaluation.class_counts.items():
        print(f"    {name}: {count}")
    print("  最重要特征:")
    for name, importance in evaluation.feature_importance[:5]:
        print(f"    {name}: {importance:.3f}")
    print(f"  模型: {model_path}")
    print(f"  训练记录: {records_path}")
    print(f"  评估报告: {report_path}")
    print(f"  可读规则: {rules_path}")


if __name__ == "__main__":
    main()
