"""Discovery and loading of third-party OT algorithm plugins."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


@dataclass
class PluginLoadReport:
    """Summary returned by :func:`load_algorithm_plugins`."""

    loaded: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)


def _plugin_files(plugin_dir: Path) -> list[Path]:
    if not plugin_dir.is_dir():
        return []

    files = [
        path for path in plugin_dir.glob("*.py")
        if not path.name.startswith("_") and path.name != "plugin_template.py"
        and not (path.parent / "ot_benchmark_plugin.json").exists()
    ]
    files.extend(
        path for path in plugin_dir.glob("*/ot_plugin.py")
        if not (path.parent / "ot_benchmark_plugin.json").exists()
    )
    return sorted(set(files), key=lambda path: str(path).lower())


def _manifest_files(plugin_dir: Path) -> list[Path]:
    if not plugin_dir.is_dir():
        return []
    return sorted(plugin_dir.rglob("ot_benchmark_plugin.json"), key=lambda p: str(p).lower())


def _import_plugin(path: Path) -> ModuleType:
    # When the selected file lives inside a real Python package, import it by
    # its dotted name. Loading it as an anonymous file would break statements
    # such as ``from ..utils import ...`` (e.g. POT's Screenkhorn module).
    package_parts: list[str] = []
    package_dir = path.parent
    while (package_dir / "__init__.py").is_file():
        package_parts.insert(0, package_dir.name)
        package_dir = package_dir.parent
    if package_parts:
        module_name = ".".join([*package_parts, path.stem])
        import_root = str(package_dir)
        if import_root not in sys.path:
            sys.path.insert(0, import_root)
        return importlib.import_module(module_name)

    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    module_name = f"ot_benchmark_external_{path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为 {path} 创建模块")

    module = importlib.util.module_from_spec(spec)
    # The repository root and adapter directory are temporarily importable so an
    # ot_plugin.py adapter can import the GitHub project placed beside it.
    additions = [str(path.parent), str(path.parent.parent)]
    old_path = list(sys.path)
    try:
        for entry in reversed(additions):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.path[:] = old_path
    return module


def _conventional_spec(module: ModuleType, path: Path) -> dict[str, Any] | None:
    func = getattr(module, "run_ot", None)
    if not callable(func):
        return None
    return {
        "name": getattr(module, "ALGORITHM_NAME", path.stem.replace("_", " ").title()),
        "func": func,
        "description": getattr(module, "DESCRIPTION", f"External plugin: {path.name}"),
        "complexity_class": getattr(module, "COMPLEXITY", "unknown"),
        "supports_sparse": bool(getattr(module, "SUPPORTS_SPARSE", False)),
        "supports_gpu": bool(getattr(module, "SUPPORTS_GPU", False)),
        "uses_support_points": bool(getattr(module, "USES_SUPPORT_POINTS", False)),
        "uses_sparse_cost": bool(getattr(module, "USES_SPARSE_COST", False)),
        "max_dim": getattr(module, "MAX_DIM", float("inf")),
        "max_active_dim": getattr(module, "MAX_ACTIVE_DIM", float("inf")),
    }


def _extract_specs(module: ModuleType, path: Path) -> list[dict[str, Any]]:
    raw = getattr(module, "ALGORITHMS", None)
    if raw is None:
        raw = getattr(module, "ALGORITHM", None)
    if raw is None:
        conventional = _conventional_spec(module, path)
        if conventional is None:
            raise ValueError("未找到 ALGORITHM、ALGORITHMS 或 run_ot")
        return [conventional]
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, (list, tuple)) and all(isinstance(item, dict) for item in raw):
        return list(raw)
    raise TypeError("ALGORITHM 必须是字典，ALGORITHMS 必须是字典列表")


def _validate_spec(spec: dict[str, Any]) -> None:
    if not isinstance(spec.get("name"), str) or not spec["name"].strip():
        raise ValueError("算法配置缺少非空 name")
    func = spec.get("func")
    if not callable(func):
        raise ValueError(f"{spec['name']} 的 func 不可调用")
    try:
        inspect.signature(func).bind(None, None, None)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{spec['name']} 必须能接收 source, target, cost_matrix 三个位置参数"
        ) from exc


def _convert_input(value: Any, backend: str) -> Any:
    if backend in {"auto", "numpy"}:
        return value
    if backend == "torch":
        import torch
        return torch.as_tensor(value, dtype=torch.float64)
    if backend == "cupy":
        import cupy as cp
        return cp.asarray(value, dtype=cp.float64)
    if backend == "jax":
        import jax.numpy as jnp
        return jnp.asarray(value)
    raise ValueError(f"不支持的 input_backend: {backend}")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            return np.asarray(value.numpy())
        except Exception:
            pass
    return np.asarray(value)


def _result_to_distance(result: Any, cost_matrix: np.ndarray, output: str) -> float:
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, dict):
        for key in ("distance", "cost", "ot_cost", "value", "loss"):
            if key in result:
                return float(_to_numpy(result[key]).reshape(-1)[0])
        for key in ("transport_plan", "plan", "gamma", "coupling", "P"):
            if key in result:
                result = result[key]
                break
        else:
            raise ValueError("返回字典中没有可识别的距离或运输矩阵字段")

    array = _to_numpy(result)
    if output == "distance" or (output == "auto" and array.size == 1):
        return float(array.reshape(-1)[0])
    if output in {"plan", "auto"} and array.ndim == 2:
        if array.shape != cost_matrix.shape:
            raise ValueError(
                f"运输矩阵形状 {array.shape} 与代价矩阵 {cost_matrix.shape} 不一致"
            )
        return float(np.sum(array * cost_matrix))
    raise ValueError(f"无法把形状 {array.shape} 的返回值解释为 OT 距离")


def _manifest_callable(manifest_path: Path, config: dict[str, Any]):
    entry_file = (manifest_path.parent / config["entry_file"]).resolve()
    function_name = config["function"]
    backend = config.get("input_backend", "numpy")
    output = config.get("output", "auto")
    fixed_kwargs = config.get("kwargs", {})
    cached: dict[str, Any] = {}

    def run(source, target, cost_matrix, **benchmark_kwargs):
        if "function" not in cached:
            module = _import_plugin(entry_file)
            function = getattr(module, function_name, None)
            if not callable(function):
                raise AttributeError(f"{entry_file} 中找不到函数 {function_name}")
            cached["function"] = function
        function = cached["function"]
        call_kwargs = dict(fixed_kwargs)
        # Only pass optional benchmark data when the target function accepts it.
        signature = inspect.signature(function)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
        )
        for key in ("support_points", "cost_matrix_csr", "backend"):
            if key in benchmark_kwargs and (accepts_kwargs or key in signature.parameters):
                call_kwargs[key] = benchmark_kwargs[key]
        converted_cost = _convert_input(cost_matrix, backend)
        result = function(
            _convert_input(source, backend),
            _convert_input(target, backend),
            converted_cost,
            **call_kwargs,
        )
        return _result_to_distance(result, cost_matrix, output)

    return run


def _load_manifest_specs(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_algorithms = data.get("algorithms")
    if raw_algorithms is None:
        raw_algorithms = [data.get("algorithm", data)]
    if not isinstance(raw_algorithms, list):
        raise TypeError("manifest 的 algorithms 必须是列表")
    specs = []
    for config in raw_algorithms:
        if not isinstance(config, dict):
            raise TypeError("manifest 中的算法配置必须是字典")
        for required in ("name", "entry_file", "function"):
            if not config.get(required):
                raise ValueError(f"manifest 算法配置缺少 {required}")
        spec = dict(config)
        spec["func"] = _manifest_callable(path, config)
        specs.append(spec)
    return specs


def load_algorithm_plugins(
    plugin_dir: str | Path,
    registry: dict[str, dict[str, Any]],
) -> PluginLoadReport:
    """Load plugins into ``registry`` without allowing name collisions."""
    report = PluginLoadReport()
    allowed = {
        "description", "max_dim", "supports_sparse", "supports_gpu",
        "complexity_class", "paper_reference", "hyperparameters",
        "uses_support_points", "uses_sparse_cost", "ranking_group",
        "max_active_dim",
    }

    plugin_root = Path(plugin_dir)
    sources: list[tuple[Path, str]] = [
        (path, "manifest") for path in _manifest_files(plugin_root)
    ] + [
        (path, "python") for path in _plugin_files(plugin_root)
    ]

    for path, source_type in sources:
        try:
            if source_type == "manifest":
                specs = _load_manifest_specs(path)
            else:
                module = _import_plugin(path)
                specs = _extract_specs(module, path)
            for spec in specs:
                _validate_spec(spec)
                name = spec["name"].strip()
                if name in registry:
                    raise ValueError(f"算法名称与现有算法冲突: {name}")
                metadata = {key: value for key, value in spec.items() if key in allowed}
                registry[name] = {"func": spec["func"], **metadata}
                # Fill fields consumed directly by the benchmark/UI.
                registry[name].setdefault("description", f"External plugin: {path.name}")
                registry[name].setdefault("max_dim", float("inf"))
                registry[name].setdefault("max_active_dim", float("inf"))
                registry[name].setdefault("supports_sparse", False)
                registry[name].setdefault("supports_gpu", False)
                registry[name].setdefault("complexity_class", "unknown")
                registry[name].setdefault("ranking_group", "balanced")
                registry[name].setdefault("uses_support_points", False)
                registry[name].setdefault("uses_sparse_cost", False)
                report.loaded.append(name)
        except Exception as exc:
            report.errors.append((str(path), f"{type(exc).__name__}: {exc}"))
    return report
