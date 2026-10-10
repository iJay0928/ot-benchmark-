"""KNN downstream validation for OT-based unsupervised domain adaptation."""

from __future__ import annotations

import time
import tracemalloc
import csv
import json
from dataclasses import asdict, dataclass
from typing import Callable
from pathlib import Path

import numpy as np
import ot
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class DomainAdaptationDataset:
    source_x: np.ndarray
    source_y: np.ndarray
    target_x: np.ndarray
    target_y: np.ndarray
    description: str


@dataclass
class KNNValidationResult:
    algorithm: str
    accuracy: float
    macro_f1: float
    accuracy_gain: float
    runtime_seconds: float
    peak_memory_mb: float
    success: bool
    error: str = ""


def generate_domain_shift_dataset(
    n_samples: int = 240,
    n_features: int = 10,
    n_classes: int = 3,
    shift_intensity: float = 1.0,
    noise: float = 0.35,
    random_state: int = 42,
) -> DomainAdaptationDataset:
    """Create labeled source/target domains with a reproducible covariate shift."""
    if n_samples < n_classes * 10:
        raise ValueError("每个类别至少需要约 10 个样本")
    if n_features < 2:
        raise ValueError("特征维度至少为 2")
    rng = np.random.RandomState(random_state)
    centers = rng.normal(0.0, 2.2, size=(n_classes, n_features))
    source_y = np.arange(n_samples) % n_classes
    target_y = np.arange(n_samples) % n_classes
    rng.shuffle(source_y)
    rng.shuffle(target_y)
    source_x = centers[source_y] + rng.normal(0.0, noise, size=(n_samples, n_features))
    latent_target = centers[target_y] + rng.normal(0.0, noise * 1.15, size=(n_samples, n_features))

    raw = rng.normal(size=(n_features, n_features))
    rotation, _ = np.linalg.qr(raw)
    blend = min(1.0, max(0.0, shift_intensity / 2.0))
    transform = (1.0 - blend) * np.eye(n_features) + blend * rotation
    scales = np.linspace(1.0, 1.0 + 0.35 * shift_intensity, n_features)
    translation = rng.normal(0.0, 0.9 * shift_intensity, size=n_features)
    target_x = (latent_target @ transform) * scales + translation
    return DomainAdaptationDataset(
        source_x=source_x.astype(np.float64), source_y=source_y,
        target_x=target_x.astype(np.float64), target_y=target_y,
        description=(f"{n_classes} classes, {n_features} features, "
                     f"covariate shift={shift_intensity:.2f}, noise={noise:.2f}"),
    )


def _normalized_cost(source_x: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    cost = cdist(source_x, target_x, metric="sqeuclidean")
    positive = cost[cost > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    return cost / max(scale, 1e-12)


def _barycentric_map(coupling: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    row_mass = coupling.sum(axis=1, keepdims=True)
    return coupling @ target_x / np.maximum(row_mass, 1e-15)


def _emd_map(source_x, target_x, cost):
    a = np.full(len(source_x), 1.0 / len(source_x))
    b = np.full(len(target_x), 1.0 / len(target_x))
    return _barycentric_map(ot.emd(a, b, cost, numItermax=300_000), target_x)


def _sinkhorn_map(source_x, target_x, cost, reg=0.1):
    a = np.full(len(source_x), 1.0 / len(source_x))
    b = np.full(len(target_x), 1.0 / len(target_x))
    coupling = ot.sinkhorn(a, b, cost, reg=reg, method="sinkhorn_log",
                           numItermax=3000, stopThr=1e-7, warn=False)
    return _barycentric_map(coupling, target_x)


def _greenkhorn_map(source_x, target_x, cost):
    a = np.full(len(source_x), 1.0 / len(source_x))
    b = np.full(len(target_x), 1.0 / len(target_x))
    coupling = ot.bregman.greenkhorn(
        a, b, cost, reg=0.1, numItermax=30_000, stopThr=1e-7, warn=False
    )
    return _barycentric_map(coupling, target_x)


def _lowrank_map(source_x, target_x, _cost):
    rank = max(2, min(20, len(source_x) // 8, len(target_x) // 8))
    q, r, g = ot.lowrank_sinkhorn(
        source_x, target_x, reg=0.1, rank=rank, numItermax=1500,
        stopThr=1e-7, warn=False, seed_init=42,
    )
    coupling = q @ np.diag(1.0 / np.maximum(g, 1e-15)) @ r.T
    return _barycentric_map(coupling, target_x)


def _sparse_sinkhorn_map(source_x, target_x, cost):
    n, m = cost.shape
    k = max(5, min(m, int(np.sqrt(m)) + 2))
    mask = np.zeros((n, m), dtype=bool)
    nearest = np.argpartition(cost, kth=k - 1, axis=1)[:, :k]
    mask[np.arange(n)[:, None], nearest] = True
    nearest_source = np.argmin(cost, axis=0)
    mask[nearest_source, np.arange(m)] = True
    kernel = np.exp(-cost / 0.1) * mask
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    u = np.ones(n)
    v = np.ones(m)
    for _ in range(1500):
        old_u = u
        u = a / np.maximum(kernel @ v, 1e-300)
        v = b / np.maximum(kernel.T @ u, 1e-300)
        if np.max(np.abs(u - old_u)) < 1e-8:
            break
    coupling = (u[:, None] * kernel) * v[None, :]
    return _barycentric_map(coupling, target_x)


def _sliced_map(source_x, target_x, _cost, n_projections=80):
    """Iterative random-projection transport map for equal empirical sizes."""
    if len(source_x) != len(target_x):
        raise ValueError("Sliced mapping currently requires equal source/target sizes")
    rng = np.random.RandomState(42)
    mapped = source_x.copy()
    for _ in range(n_projections):
        direction = rng.normal(size=mapped.shape[1])
        direction /= np.linalg.norm(direction) + 1e-15
        source_projection = mapped @ direction
        target_projection = target_x @ direction
        source_order = np.argsort(source_projection)
        target_order = np.argsort(target_projection)
        delta = target_projection[target_order] - source_projection[source_order]
        mapped[source_order] += 0.7 * delta[:, None] * direction[None, :]
    return mapped


KNN_OT_ADAPTERS: dict[str, Callable] = {
    "EMD exact": _emd_map,
    "Sinkhorn (reg=0.1)": lambda x, y, c: _sinkhorn_map(x, y, c, 0.1),
    "Sinkhorn (reg=0.01)": lambda x, y, c: _sinkhorn_map(x, y, c, 0.01),
    "Greenkhorn": _greenkhorn_map,
    "Low-Rank OT": _lowrank_map,
    "Sparse Sinkhorn": _sparse_sinkhorn_map,
    "Sliced OT (80)": _sliced_map,
}


def _knn_scores(train_x, train_y, target_x, target_y, neighbors):
    model = KNeighborsClassifier(n_neighbors=neighbors, weights="distance")
    model.fit(train_x, train_y)
    prediction = model.predict(target_x)
    return float(accuracy_score(target_y, prediction)), float(
        f1_score(target_y, prediction, average="macro", zero_division=0)
    )


def run_knn_downstream_validation(
    dataset: DomainAdaptationDataset,
    algorithms: list[str] | None = None,
    neighbors: int = 5,
) -> list[KNNValidationResult]:
    """Compare KNN before and after each unsupervised OT alignment."""
    source_x, source_y = dataset.source_x, dataset.source_y
    target_x, target_y = dataset.target_x, dataset.target_y
    baseline_accuracy, baseline_f1 = _knn_scores(
        source_x, source_y, target_x, target_y, neighbors
    )
    results = [KNNValidationResult(
        algorithm="No OT baseline", accuracy=baseline_accuracy, macro_f1=baseline_f1,
        accuracy_gain=0.0, runtime_seconds=0.0, peak_memory_mb=0.0, success=True,
    )]
    selected = algorithms or list(KNN_OT_ADAPTERS)
    cost = _normalized_cost(source_x, target_x)
    for name in selected:
        adapter = KNN_OT_ADAPTERS.get(name)
        if adapter is None:
            results.append(KNNValidationResult(
                name, 0.0, 0.0, 0.0, 0.0, 0.0, False, "没有可用的下游映射适配器"
            ))
            continue
        tracemalloc.start()
        started = time.perf_counter()
        try:
            aligned_source = adapter(source_x, target_x, cost)
            accuracy, macro_f1 = _knn_scores(
                aligned_source, source_y, target_x, target_y, neighbors
            )
            _, peak = tracemalloc.get_traced_memory()
            results.append(KNNValidationResult(
                algorithm=name, accuracy=accuracy, macro_f1=macro_f1,
                accuracy_gain=accuracy - baseline_accuracy,
                runtime_seconds=time.perf_counter() - started,
                peak_memory_mb=peak / (1024 * 1024), success=True,
            ))
        except Exception as exc:
            _, peak = tracemalloc.get_traced_memory()
            results.append(KNNValidationResult(
                algorithm=name, accuracy=0.0, macro_f1=0.0, accuracy_gain=0.0,
                runtime_seconds=time.perf_counter() - started,
                peak_memory_mb=peak / (1024 * 1024), success=False, error=str(exc),
            ))
        finally:
            tracemalloc.stop()
    return results


def results_as_dicts(results: list[KNNValidationResult]) -> list[dict]:
    return [asdict(result) for result in results]


def save_knn_validation_results(
    output_dir: str | Path,
    dataset: DomainAdaptationDataset,
    results: list[KNNValidationResult],
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "knn_downstream_results.csv"
    rows = results_as_dicts(results)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = destination / "knn_downstream_results.json"
    json_path.write_text(json.dumps({
        "dataset": dataset.description,
        "source_shape": list(dataset.source_x.shape),
        "target_shape": list(dataset.target_x.shape),
        "results": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def plot_knn_validation_results(
    output_path: str | Path,
    results: list[KNNValidationResult],
) -> Path:
    """Save a publication-ready local PNG comparing accuracy and runtime."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    valid = [result for result in results if result.success]
    if not valid:
        raise ValueError("没有成功的 KNN 下游结果可用于绘图")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = [result.algorithm for result in valid]
    accuracy = np.asarray([result.accuracy * 100 for result in valid])
    runtime = np.asarray([result.runtime_seconds for result in valid])
    baseline = accuracy[0]

    figure, (axis_accuracy, axis_runtime) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.45, 1]}
    )
    y = np.arange(len(names))
    colors = ["#94a3b8"] + ["#38bdf8"] * (len(names) - 1)
    if len(colors) > 1:
        fastest_index = 1 + int(np.argmin(runtime[1:]))
        colors[fastest_index] = "#22c55e"
    bars = axis_accuracy.barh(y, accuracy, color=colors, height=0.64)
    axis_accuracy.axvline(baseline, color="#64748b", linestyle="--", linewidth=1.5)
    axis_accuracy.set_yticks(y, labels=names)
    axis_accuracy.invert_yaxis()
    axis_accuracy.set_xlim(0, 108)
    axis_accuracy.set_xlabel("KNN accuracy (%)")
    axis_accuracy.set_title("Downstream classification after OT alignment")
    axis_accuracy.grid(axis="x", alpha=0.22)
    axis_accuracy.text(baseline + 1.0, -0.55, f"baseline {baseline:.1f}%",
                       color="#475569", fontsize=9, va="center")
    for index, bar in enumerate(bars):
        gain = accuracy[index] - baseline
        label = f"{accuracy[index]:.1f}%" if index == 0 else f"{accuracy[index]:.1f}%  ({gain:+.1f})"
        axis_accuracy.text(min(106, accuracy[index] + 1.0), bar.get_y() + bar.get_height() / 2,
                           label, va="center", fontsize=9)

    ot_names = names[1:]
    ot_runtime = runtime[1:]
    runtime_colors = ["#22c55e" if value == ot_runtime.min() else "#f59e0b"
                      for value in ot_runtime]
    runtime_bars = axis_runtime.barh(np.arange(len(ot_names)), ot_runtime,
                                     color=runtime_colors, height=0.64)
    axis_runtime.set_yticks(np.arange(len(ot_names)), labels=ot_names)
    axis_runtime.invert_yaxis()
    axis_runtime.set_xlabel("OT + KNN runtime (seconds)")
    axis_runtime.set_title("Runtime cost")
    axis_runtime.grid(axis="x", alpha=0.22)
    for value, bar in zip(ot_runtime, runtime_bars):
        axis_runtime.text(value + max(ot_runtime.max() * 0.015, 0.002),
                          bar.get_y() + bar.get_height() / 2,
                          f"{value:.3f}s", va="center", fontsize=9)
    axis_runtime.set_xlim(0, max(0.1, ot_runtime.max() * 1.22))
    figure.suptitle("OT Algorithm KNN Downstream Validation", fontsize=15, fontweight="bold")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return destination
