"""Static, no-code import wizard for third-party OT repositories."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path


SOURCE_NAMES = {"a", "source", "source_dist", "mu", "p", "weights_a"}
TARGET_NAMES = {"b", "target", "target_dist", "nu", "q", "weights_b"}
COST_NAMES = {"m", "cost", "cost_matrix", "c", "ground_cost", "distance_matrix"}
FUNCTION_HINTS = ("transport", "sinkhorn", "wasserstein", "ot", "solve", "emd")
SKIP_DIRS = {".git", ".venv", "venv", "tests", "test", "docs", "examples", "__pycache__"}


@dataclass
class Candidate:
    file: Path
    function: str
    parameters: list[str]
    required_extra: list[str] = field(default_factory=list)
    score: int = 0

    @property
    def display(self) -> str:
        return f"{self.file} :: {self.function}({', '.join(self.parameters)})"


def _python_files(repository: Path):
    for path in repository.rglob("*.py"):
        if not any(part.lower() in SKIP_DIRS for part in path.relative_to(repository).parts):
            if path.name not in {"ot_plugin.py"}:
                yield path


def _is_ot_signature(parameters: list[str]) -> bool:
    if len(parameters) < 3:
        return False
    first = [name.lower() for name in parameters[:3]]
    return first[0] in SOURCE_NAMES and first[1] in TARGET_NAMES and first[2] in COST_NAMES


def scan_repository(repository: str | Path) -> list[Candidate]:
    """Find functions whose first three arguments look like distributions and cost."""
    root = Path(repository).resolve()
    candidates: list[Candidate] = []
    for path in _python_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            positional = list(node.args.posonlyargs) + list(node.args.args)
            names = [arg.arg for arg in positional]
            if names and names[0] in {"self", "cls"}:
                continue
            if not _is_ot_signature(names):
                continue
            default_count = len(node.args.defaults)
            required_count = len(positional) - default_count
            required_extra = names[3:required_count]
            score = 10
            lowered = node.name.lower()
            score += sum(3 for hint in FUNCTION_HINTS if hint in lowered)
            score -= len(required_extra) * 2
            candidates.append(Candidate(
                file=path.relative_to(root), function=node.name,
                parameters=names, required_extra=required_extra, score=score,
            ))
    return sorted(candidates, key=lambda item: (-item.score, str(item.file), item.function))


def _ask_choice(prompt: str, count: int, default: int = 1) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default - 1
        if raw.isdigit() and 1 <= int(raw) <= count:
            return int(raw) - 1
        print(f"请输入 1 到 {count}。")


def _parse_value(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def run_import_wizard(repository: str | Path, auto: bool = False) -> Path:
    root = Path(repository).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"仓库目录不存在: {root}")
    candidates = scan_repository(root)
    if not candidates:
        raise RuntimeError(
            "没有找到兼容函数。函数的前三个参数需要类似 "
            "(a, b, M) 或 (source, target, cost_matrix)。"
        )

    if auto:
        compatible = [candidate for candidate in candidates if not candidate.required_extra]
        if not compatible:
            raise RuntimeError(
                "自动模式找到的候选函数都有额外必需参数，请使用交互向导配置。"
            )
        chosen = compatible[0]
        name = chosen.function.replace("_", " ").title()
        output = "auto"
        backend = "numpy"
        kwargs = {}
    else:
        print("\n发现以下兼容 OT 函数：")
        for index, candidate in enumerate(candidates, 1):
            suffix = (
                f"  [还需配置: {', '.join(candidate.required_extra)}]"
                if candidate.required_extra else ""
            )
            print(f"  [{index}] {candidate.display}{suffix}")
        chosen = candidates[_ask_choice("选择入口函数", len(candidates))]
        default_name = chosen.function.replace("_", " ").title()
        name = input(f"算法显示名称 [{default_name}]: ").strip() or default_name

        print("\n返回类型: [1] 自动判断  [2] 距离标量  [3] 运输矩阵")
        output = ("auto", "distance", "plan")[_ask_choice("选择返回类型", 3)]
        print("输入后端: [1] NumPy  [2] PyTorch  [3] CuPy  [4] JAX")
        backend = ("numpy", "torch", "cupy", "jax")[_ask_choice("选择输入后端", 4)]
        kwargs = {}
        for parameter in chosen.required_extra:
            raw = input(f"必需参数 {parameter} 的值（支持数字/JSON）: ").strip()
            if not raw:
                raise ValueError(f"必需参数 {parameter} 未提供")
            kwargs[parameter] = _parse_value(raw)

    manifest = {
        "version": 1,
        "algorithm": {
            "name": name,
            "entry_file": chosen.file.as_posix(),
            "function": chosen.function,
            "output": output,
            "input_backend": backend,
            "kwargs": kwargs,
            "description": "由 OT Benchmark 无代码导入向导生成",
            "supports_gpu": backend in {"torch", "cupy", "jax"},
            "supports_sparse": False,
            "complexity_class": "unknown",
        },
    }
    destination = root / "ot_benchmark_plugin.json"
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n导入配置已生成: {destination}")
    print("以后启动评测程序时会自动加载这个算法。")
    return destination
