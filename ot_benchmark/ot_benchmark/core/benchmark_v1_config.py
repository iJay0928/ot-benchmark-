"""Benchmark v1.0 数据集清单加载与校验."""

from __future__ import annotations

import os
import yaml
from typing import Optional


# 场景必需字段
_REQUIRED_FIELDS = ["id", "name", "n_source", "dimension", "sparsity", "seeds"]

# 合法字段值
_VALID_SOURCE_TYPES = {"uniform", "zipf", "pareto", "gaussian_mixture", "real"}
_VALID_STRUCTURES = {"random", "block", "group", "hierarchical"}
_VALID_GROUND_TRUTH_MODES = {"exact", "sinkhorn_ref", "l1_fallback", "unavailable"}
_VALID_SPARSITY_SEMANTICS = {"zero_ratio", "nonzero_ratio"}


class ConfigValidationError(ValueError):
    """配置文件校验失败异常。"""


def validate_manifest(scenarios: list[dict]) -> list[str]:
    """校验 benchmark_v1.yaml 的场景列表。返回错误消息列表，空列表表示通过。

    校验规则:
      - 必填字段存在性检查
      - sparsity 范围 [0, 1]
      - sparsity_semantics 枚举检查
      - source_type / target_type 枚举检查
      - structure 枚举检查
      - ground_truth_mode 枚举检查
      - id 唯一性检查
      - n_source > 0, dimension > 0
      - seed 类型检查
    """
    errors: list[str] = []
    seen_ids: set[str] = set()

    if not isinstance(scenarios, list):
        return ["scenarios 必须是一个列表"]

    for i, sc in enumerate(scenarios):
        prefix = f"场景[{i}] (id={sc.get('id', '缺失')})"

        # --- 必填字段 ---
        for field in _REQUIRED_FIELDS:
            if field not in sc or sc[field] is None:
                errors.append(f"{prefix}: 缺少必需字段 '{field}'")

        # --- sparsity 范围 ---
        if "sparsity" in sc and isinstance(sc["sparsity"], (int, float)):
            if not (0.0 <= sc["sparsity"] <= 1.0):
                errors.append(
                    f"{prefix}: sparsity 必须在 [0, 1] 之间, 当前值 {sc['sparsity']}"
                )

        # --- sparsity_semantics ---
        if "sparsity_semantics" in sc and sc["sparsity_semantics"] is not None:
            if str(sc["sparsity_semantics"]) not in _VALID_SPARSITY_SEMANTICS:
                errors.append(
                    f"{prefix}: sparsity_semantics 必须是 "
                    f"{_VALID_SPARSITY_SEMANTICS}, 当前值 '{sc['sparsity_semantics']}'"
                )

        # --- source_type ---
        if sc.get("source_type") and str(sc["source_type"]) not in _VALID_SOURCE_TYPES:
            errors.append(
                f"{prefix}: source_type 必须是 {_VALID_SOURCE_TYPES}, "
                f"当前值 '{sc['source_type']}'"
            )

        # --- target_type ---
        if sc.get("target_type") and str(sc["target_type"]) not in _VALID_SOURCE_TYPES:
            errors.append(
                f"{prefix}: target_type 必须是 {_VALID_SOURCE_TYPES}, "
                f"当前值 '{sc['target_type']}'"
            )

        # --- structure ---
        if sc.get("structure") and str(sc["structure"]) not in _VALID_STRUCTURES:
            errors.append(
                f"{prefix}: structure 必须是 {_VALID_STRUCTURES}, "
                f"当前值 '{sc['structure']}'"
            )

        # --- ground_truth_mode ---
        if sc.get("ground_truth_mode") and str(sc["ground_truth_mode"]) not in _VALID_GROUND_TRUTH_MODES:
            errors.append(
                f"{prefix}: ground_truth_mode 必须是 "
                f"{_VALID_GROUND_TRUTH_MODES}, 当前值 '{sc['ground_truth_mode']}'"
            )

        # --- seeds 检查 ---
        seeds = sc.get("seeds")
        if seeds is not None:
            if not isinstance(seeds, list) or len(seeds) == 0:
                errors.append(f"{prefix}: seeds 必须是一个非空的整数列表")
            else:
                for j, s in enumerate(seeds):
                    if not isinstance(s, int):
                        errors.append(f"{prefix}: seeds[{j}] 必须是整数, 当前值 {s}")

        # --- id 唯一性 ---
        sid = sc.get("id")
        if sid is not None and str(sid) in seen_ids:
            errors.append(f"{prefix}: id '{sid}' 重复")
        if sid is not None:
            seen_ids.add(str(sid))

        # --- n_source > 0 ---
        if "n_source" in sc and isinstance(sc["n_source"], (int, float)):
            if sc["n_source"] <= 0:
                errors.append(
                    f"{prefix}: n_source 必须 > 0, 当前值 {sc['n_source']}"
                )

        # --- dimension > 0 ---
        if "dimension" in sc and isinstance(sc["dimension"], (int, float)):
            if sc["dimension"] <= 0:
                errors.append(
                    f"{prefix}: dimension 必须 > 0, 当前值 {sc['dimension']}"
                )

        # --- noise_level 范围 ---
        if "noise_level" in sc and isinstance(sc["noise_level"], (int, float)):
            if not (0.0 <= sc["noise_level"] <= 1.0):
                errors.append(
                    f"{prefix}: noise_level 必须在 [0, 1] 之间, "
                    f"当前值 {sc['noise_level']}"
                )

    return errors


def load_benchmark_v1_manifest(
    config_path: Optional[str] = None,
) -> list[dict]:
    """加载并校验 benchmark_v1.yaml。

    Args:
        config_path: YAML 文件路径。None 时使用项目默认路径。

    Returns:
        校验通过的场景列表（list of dict）。

    Raises:
        FileNotFoundError: 配置文件不存在。
        ConfigValidationError: 配置校验失败（包含中文错误信息）。
    """
    if config_path is None:
        # 默认路径: 项目根目录的 configs/benchmark_v1.yaml
        this_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(this_dir))  # ot_benchmark/
        config_path = os.path.join(project_root, "configs", "benchmark_v1.yaml")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ConfigValidationError("配置文件为空或格式错误")

    scenarios = data.get("scenarios", [])
    if not isinstance(scenarios, list) or len(scenarios) == 0:
        raise ConfigValidationError("scenarios 必须是一个非空列表")

    # 校验
    errors = validate_manifest(scenarios)
    if errors:
        msg = "\n".join(errors)
        raise ConfigValidationError(
            f"配置校验失败 ({len(errors)} 个错误):\n{msg}"
        )

    return scenarios


def get_enabled_scenarios(scenarios: list[dict]) -> list[dict]:
    """返回 enabled=True 的场景列表。"""
    return [s for s in scenarios if s.get("enabled", True)]


def get_quick_scenarios(scenarios: list[dict]) -> list[dict]:
    """快速模式：返回前 4 个维度 <= 500 的小场景。"""
    small = [s for s in scenarios if s.get("dimension", 9999) <= 500]
    return small[:4]
