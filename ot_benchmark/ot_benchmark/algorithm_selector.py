"""Trainable decision-tree selector for practical OT algorithms."""

from __future__ import annotations

import csv
import json
import math
import hashlib
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier, export_text

try:
    from .data_generator import (
        SparseDataset, compute_ot_reference, create_benchmark_dataset,
        generate_cost_matrix, generate_sparse_distribution, generate_structured_sparsity,
    )
except ImportError:
    from data_generator import (
        SparseDataset, compute_ot_reference, create_benchmark_dataset,
        generate_cost_matrix, generate_sparse_distribution, generate_structured_sparsity,
    )


FEATURE_NAMES = [
    "weight_accuracy",
    "weight_speed",
    "weight_memory",
    "weight_success",
    "log_dimension",
    "log_geometry_dimension",
    "actual_sparsity",
    "noise_level",
    "source_active_ratio",
    "target_active_ratio",
    "support_overlap_ratio",
    "support_contiguity",
    "source_entropy_normalized",
    "target_entropy_normalized",
    "source_gini",
    "target_gini",
    "l1_distribution_shift",
    "js_divergence",
    "cost_mean",
    "cost_std",
    "cost_local_ratio",
]


DEFAULT_SELECTOR_ALGORITHMS = (
    "Sinkhorn (reg=0.1)",
    "Sinkhorn (reg=0.01)",
    "Greenkhorn",
    "Sliced OT (200)",
    "Sliced OT (500)",
    "Low-Rank OT",
    "Factored OT",
    "Sparse Sinkhorn",
)


@dataclass
class SelectorTrainingRecord:
    dataset_name: str
    features: list[float]
    winner: str
    winner_score: float
    runner_up: str = ""
    runner_up_score: float = 0.0
    dataset_group: str = ""
    algorithm_scores: dict[str, float] = field(default_factory=dict)
    feature_seconds: float = 0.0


@dataclass
class SelectorEvaluation:
    accuracy: float
    balanced_accuracy: float
    train_size: int
    test_size: int
    labels: list[str]
    confusion_matrix: list[list[int]]
    class_counts: dict[str, int]
    feature_importance: list[tuple[str, float]]
    rules: str
    train_groups: list[str] = field(default_factory=list)
    test_groups: list[str] = field(default_factory=list)
    fixed_baseline: str = ""
    fixed_baseline_accuracy: float = 0.0
    selector_mean_score_regret: float | None = None
    fixed_mean_score_regret: float | None = None
    prediction_seconds_per_case: float = 0.0
    test_mean_feature_seconds: float = 0.0


def _entropy_normalized(probabilities: np.ndarray) -> float:
    values = probabilities[probabilities > 1e-15]
    if len(values) <= 1:
        return 0.0
    entropy = -float(np.sum(values * np.log2(values)))
    return entropy / math.log2(len(probabilities))


def _gini(values: np.ndarray) -> float:
    positive = np.sort(values[values > 1e-15])
    if len(positive) == 0:
        return 0.0
    index = np.arange(1, len(positive) + 1)
    return float(
        2 * np.sum(index * positive) / (len(positive) * np.sum(positive))
        - (len(positive) + 1) / len(positive)
    )


def _js_divergence(source: np.ndarray, target: np.ndarray) -> float:
    eps = 1e-15
    p = (source + eps) / np.sum(source + eps)
    q = (target + eps) / np.sum(target + eps)
    midpoint = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * np.log2(p / midpoint))
        + 0.5 * np.sum(q * np.log2(q / midpoint))
    )


def _distribution_features(
    source: np.ndarray,
    target: np.ndarray,
    dimension: int,
    geometry_dimension: int,
    actual_sparsity: float,
    noise_level: float,
    cost_mean: float,
    cost_std: float,
    cost_local_ratio: float,
    preference_weights: tuple[float, float, float, float],
) -> dict[str, float]:
    source_active = source > 1e-15
    target_active = target > 1e-15
    union = source_active | target_active
    overlap = source_active & target_active
    union_count = max(1, int(union.sum()))
    source_pairs = np.count_nonzero(source_active[:-1] & source_active[1:])
    target_pairs = np.count_nonzero(target_active[:-1] & target_active[1:])
    source_contiguity = source_pairs / max(1, int(source_active.sum()) - 1)
    target_contiguity = target_pairs / max(1, int(target_active.sum()) - 1)
    return {
        "weight_accuracy": float(preference_weights[0]),
        "weight_speed": float(preference_weights[1]),
        "weight_memory": float(preference_weights[2]),
        "weight_success": float(preference_weights[3]),
        "log_dimension": math.log10(max(1, dimension)),
        "log_geometry_dimension": math.log10(max(1, geometry_dimension)),
        "actual_sparsity": float(actual_sparsity),
        "noise_level": float(noise_level),
        "source_active_ratio": float(source_active.mean()),
        "target_active_ratio": float(target_active.mean()),
        "support_overlap_ratio": float(overlap.sum() / union_count),
        "support_contiguity": float(0.5 * (source_contiguity + target_contiguity)),
        "source_entropy_normalized": _entropy_normalized(source),
        "target_entropy_normalized": _entropy_normalized(target),
        "source_gini": _gini(source),
        "target_gini": _gini(target),
        "l1_distribution_shift": float(np.sum(np.abs(source - target))),
        "js_divergence": _js_divergence(source, target),
        "cost_mean": float(cost_mean),
        "cost_std": float(cost_std),
        "cost_local_ratio": float(cost_local_ratio),
    }


def extract_selector_features(dataset: SparseDataset) -> dict[str, float]:
    """Extract algorithm-independent features from an existing OT task."""
    source = np.asarray(dataset.source_dist, dtype=np.float64)
    target = np.asarray(dataset.target_dist, dtype=np.float64)
    cost = np.asarray(dataset.cost_matrix, dtype=np.float64)
    return _distribution_features(
        source=source,
        target=target,
        dimension=max(1, int(getattr(dataset, "dim", len(source)))),
        geometry_dimension=(
            int(np.asarray(dataset.support_points).shape[1])
            if np.asarray(dataset.support_points).ndim > 1 else 1
        ),
        actual_sparsity=float(getattr(dataset, "sparsity", 0.0)),
        noise_level=float(getattr(dataset, "noise_level", 0.0)),
        cost_mean=float(np.mean(cost)) if cost.size else 0.0,
        cost_std=float(np.std(cost)) if cost.size else 0.0,
        cost_local_ratio=float(np.mean(cost <= 0.1)) if cost.size else 0.0,
        preference_weights=(0.35, 0.30, 0.25, 0.10),
    )


def extract_lightweight_features(
    dimension: int,
    sparsity: float,
    noise_level: float,
    source_dist_type: str,
    target_dist_type: str,
    random_state: int = 42,
    cost_sample_pairs: int = 50_000,
    geometry_dimension: int = 1,
    structure_type: str = "random",
    preference_weights: tuple[float, float, float, float] = (0.35, 0.30, 0.25, 0.10),
) -> dict[str, float]:
    """Extract selector features without allocating an O(n^2) cost matrix.

    The distributions use O(n) memory. Cost statistics for the standard 1-D
    geometry are estimated from a bounded, reproducible sample of index pairs.
    """
    if dimension < 10:
        raise ValueError("dimension must be at least 10")
    if structure_type == "random":
        source = generate_sparse_distribution(
            dimension, sparsity, source_dist_type, noise_level, random_state
        )
        target = generate_sparse_distribution(
            dimension, sparsity, target_dist_type, noise_level, random_state + 1
        )
    else:
        source = generate_structured_sparsity(
            dimension, sparsity, structure_type, random_state=random_state
        )
        target = generate_structured_sparsity(
            dimension, sparsity, structure_type, random_state=random_state + 1
        )
    actual_sparsity = 1.0 - (
        np.count_nonzero(source) + np.count_nonzero(target)
    ) / (2.0 * dimension)

    pair_count = min(max(2_000, int(cost_sample_pairs)), max(2_000, dimension * 20))
    rng = np.random.RandomState(random_state + 7919)
    if geometry_dimension <= 1:
        left = rng.randint(0, dimension, size=pair_count)
        right = rng.randint(0, dimension, size=pair_count)
        sampled_cost = ((left - right) / max(1, dimension - 1)) ** 2
    else:
        left = rng.random_sample((pair_count, geometry_dimension))
        right = rng.random_sample((pair_count, geometry_dimension))
        sampled_cost = np.sum((left - right) ** 2, axis=1) / geometry_dimension
    return _distribution_features(
        source=source,
        target=target,
        dimension=dimension,
        geometry_dimension=geometry_dimension,
        actual_sparsity=actual_sparsity,
        noise_level=noise_level,
        cost_mean=float(sampled_cost.mean()),
        cost_std=float(sampled_cost.std()),
        cost_local_ratio=float(np.mean(sampled_cost <= 0.1)),
        preference_weights=preference_weights,
    )


def feature_vector(dataset: SparseDataset, weights=None) -> list[float]:
    features = extract_selector_features(dataset)
    if weights is not None:
        features.update({
            "weight_accuracy": float(weights.w_accuracy),
            "weight_speed": float(weights.w_speed),
            "weight_memory": float(weights.w_memory),
            "weight_success": float(weights.w_success),
        })
    return [features[name] for name in FEATURE_NAMES]


def selector_weight_profiles():
    """Return diverse, defensible optimization objectives for selector labels."""
    try:
        from .benchmark_engine import WeightConfig
    except ImportError:
        from benchmark_engine import WeightConfig

    return [
        ("balanced", WeightConfig(0.35, 0.30, 0.25, 0.10)),
        ("accuracy", WeightConfig(0.65, 0.15, 0.10, 0.10)),
        ("speed", WeightConfig(0.15, 0.65, 0.10, 0.10)),
        ("memory", WeightConfig(0.15, 0.10, 0.65, 0.10)),
    ]


def select_training_algorithms(registry: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Keep comparable, practical CPU algorithms for stable training labels."""
    selected = {}
    for name in DEFAULT_SELECTOR_ALGORITHMS:
        info = registry.get(name)
        if info is None:
            continue
        if info.get("ranking_group", "balanced") != "balanced":
            continue
        selected[name] = info
    if len(selected) < 2:
        raise ValueError("决策树训练至少需要两个可比较的实用算法")
    return selected


def _selector_dataset(
    name: str,
    source: np.ndarray,
    target: np.ndarray,
    noise_level: float,
    support_points: np.ndarray,
    geometry_description: str,
) -> SparseDataset:
    cost_dense, cost_csr = generate_cost_matrix(
        source, target, "euclidean", support_points=support_points
    )
    reference, method, exact, reference_algorithm = compute_ot_reference(
        source, target, cost_dense
    )
    sparsity = 1.0 - (np.count_nonzero(source) + np.count_nonzero(target)) / (2 * len(source))
    return SparseDataset(
        name=name, source_dist=source, target_dist=target,
        cost_matrix=cost_dense, cost_matrix_csr=cost_csr,
        ground_truth_distance=reference, dim=len(source), sparsity=float(sparsity),
        noise_level=noise_level,
        description=f"dim={len(source)}, sparsity={sparsity:.2f}, {geometry_description}",
        ground_truth_method=method, ground_truth_is_exact=exact,
        reference_algorithm=reference_algorithm, support_points=support_points,
        geometry_description=geometry_description,
    )


def generate_training_datasets(n_samples: int = 60, random_state: int = 42) -> list[SparseDataset]:
    """Generate general, extreme-sparse, structured, and point-cloud OT tasks."""
    if n_samples < 10:
        raise ValueError("训练样本数至少为 10")
    rng = np.random.RandomState(random_state)
    dimensions = np.array([100, 150, 200, 300, 500, 800])
    sparsities = np.array([0.35, 0.50, 0.70, 0.85, 0.93, 0.97])
    noises = np.array([0.0, 0.01, 0.03, 0.08, 0.15])
    distributions = np.array(["uniform", "zipf", "pareto", "gaussian_mixture"])
    datasets = []
    for index in range(n_samples):
        scenario = index % 4
        dimension = int(dimensions[(index // 4) % len(dimensions)])
        sparsity = float(sparsities[(index * 5 + index // len(dimensions)) % len(sparsities)])
        noise = float(noises[(index * 3 + index // 4) % len(noises)])
        source_type = str(distributions[index % len(distributions)])
        target_type = str(distributions[(index * 3 + 1) % len(distributions)])
        seed = int(rng.randint(0, 2**31 - 1))
        if scenario == 0:
            datasets.append(create_benchmark_dataset(
                n_samples=dimension, sparsity=sparsity, noise_level=noise,
                source_dist_type=source_type, target_dist_type=target_type,
                random_state=seed, name=f"General-{index + 1:03d}-{dimension}D",
            ))
        elif scenario == 1:
            dimension = int([600, 1000, 1400][(index // 4) % 3])
            sparsity = float([0.95, 0.98, 0.99][(index // 4) % 3])
            datasets.append(create_benchmark_dataset(
                n_samples=dimension, sparsity=sparsity, noise_level=noise,
                source_dist_type=source_type, target_dist_type=target_type,
                random_state=seed, name=f"ExtremeSparse-{index + 1:03d}-{dimension}D",
            ))
        elif scenario == 2:
            structure = ["block", "group", "hierarchical"][(index // 4) % 3]
            source = generate_structured_sparsity(
                dimension, max(0.65, sparsity), structure, random_state=seed
            )
            target = generate_structured_sparsity(
                dimension, max(0.65, sparsity), structure, random_state=seed + 1
            )
            points = np.linspace(0.0, 1.0, dimension).reshape(-1, 1)
            datasets.append(_selector_dataset(
                f"Structured-{structure}-{index + 1:03d}-{dimension}D",
                source, target, noise, points, f"{structure} structured 1D support",
            ))
        else:
            geometry_dim = int([2, 5, 10][(index // 4) % 3])
            point_count = min(dimension, 500)
            source = generate_sparse_distribution(
                point_count, min(sparsity, 0.90), source_type, noise, seed
            )
            target = generate_sparse_distribution(
                point_count, min(sparsity, 0.90), target_type, noise, seed + 1
            )
            points = rng.random_sample((point_count, geometry_dim))
            datasets.append(_selector_dataset(
                f"PointCloud-{geometry_dim}D-{index + 1:03d}-{point_count}P",
                source, target, noise, points, f"{geometry_dim}D point-cloud geometry",
            ))
    return datasets


def collect_training_records(
    datasets: Iterable[SparseDataset],
    algorithms: dict[str, dict[str, Any]],
    weights=None,
    backend=None,
    progress: bool = True,
    weight_profiles=None,
    evidence_dir: str | Path | None = None,
) -> list[SelectorTrainingRecord]:
    try:
        from .benchmark_engine import benchmark_dataset
    except ImportError:
        from benchmark_engine import benchmark_dataset

    records = []
    dataset_list = list(datasets)
    profiles = weight_profiles or [("balanced", weights)]
    for index, dataset in enumerate(dataset_list, 1):
        zero_index = index - 1
        profile_index = (zero_index + zero_index // max(1, len(profiles))) % len(profiles)
        profile_name, profile_weights = profiles[profile_index]
        if progress:
            print(f"[{index:>3}/{len(dataset_list)}] 评测 {dataset.name} ...")
        report = benchmark_dataset(
            dataset, algorithms=algorithms, weights=profile_weights, verbose=False, backend=backend
        )
        if evidence_dir is not None:
            destination = Path(evidence_dir)
            destination.mkdir(parents=True, exist_ok=True)
            def clean(value):
                if isinstance(value, dict):
                    return {k: clean(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return [clean(v) for v in value]
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    return None
                return value
            (destination / f"case_{index:04d}.json").write_text(
                json.dumps(clean(asdict(report)), ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
        valid = [
            item for item in report.overall_ranking
            if any(
                result.algorithm_name == item[0] and result.success and not result.not_applicable
                for result in report.results
            )
        ]
        if not valid:
            continue
        runner_up = valid[1] if len(valid) > 1 else ("", 0.0)
        started = time.perf_counter()
        features = feature_vector(dataset, profile_weights)
        feature_seconds = time.perf_counter() - started
        digest = hashlib.sha256()
        for array in (dataset.source_dist, dataset.target_dist, dataset.cost_matrix):
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(np.asarray(array, dtype=np.float64).tobytes())
        records.append(SelectorTrainingRecord(
            dataset_name=f"{dataset.name} [{profile_name}]",
            features=features,
            winner=valid[0][0],
            winner_score=float(valid[0][1]),
            runner_up=runner_up[0],
            runner_up_score=float(runner_up[1]),
            dataset_group=digest.hexdigest(),
            algorithm_scores=dict(valid),
            feature_seconds=feature_seconds,
        ))
    return records


def train_decision_tree(
    records: list[SelectorTrainingRecord],
    max_depth: int = 5,
    min_samples_leaf: int = 2,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[DecisionTreeClassifier, SelectorEvaluation]:
    if len(records) < 10:
        raise ValueError("有效训练记录至少需要 10 条")
    x = np.asarray([record.features for record in records], dtype=np.float64)
    y = np.asarray([record.winner for record in records])
    counts = Counter(y)
    groups = np.asarray([r.dataset_group or r.dataset_name.split(" [")[0] for r in records])
    if len(set(groups)) < 2:
        raise ValueError("at least two independent dataset groups required")
    train_indices, test_indices = next(GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    ).split(x, y, groups))
    x_train, y_train = x[train_indices], y[train_indices]
    x_test, y_test = x[test_indices], y[test_indices]
    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    started = time.perf_counter()
    prediction = model.predict(x_test)
    prediction_seconds = (time.perf_counter() - started) / len(x_test)
    candidates = sorted({name for i in train_indices for name in records[i].algorithm_scores})
    baseline = (max(candidates, key=lambda name: np.mean([
        records[i].algorithm_scores.get(name, 0.0) for i in train_indices
    ])) if candidates else Counter(y_train).most_common(1)[0][0])
    have_scores = all(records[i].algorithm_scores for i in test_indices)
    def regret(choices):
        return float(np.mean([max(records[i].algorithm_scores.values())
                              - records[i].algorithm_scores.get(choice, 0.0)
                              for i, choice in zip(test_indices, choices)])) if have_scores else None
    labels = sorted(set(y) | set(prediction))
    balanced = (
        float(balanced_accuracy_score(y_test, prediction))
        if len(set(y_test)) > 1 else float(accuracy_score(y_test, prediction))
    )
    importance = sorted(
        zip(FEATURE_NAMES, model.feature_importances_.tolist()),
        key=lambda item: item[1], reverse=True,
    )
    evaluation = SelectorEvaluation(
        accuracy=float(accuracy_score(y_test, prediction)),
        balanced_accuracy=balanced,
        train_size=len(x_train),
        test_size=len(x_test),
        labels=labels,
        confusion_matrix=confusion_matrix(y_test, prediction, labels=labels).tolist(),
        class_counts=dict(sorted(counts.items())),
        feature_importance=importance,
        rules=export_text(model, feature_names=FEATURE_NAMES),
        train_groups=sorted(set(groups[train_indices])),
        test_groups=sorted(set(groups[test_indices])),
        fixed_baseline=baseline,
        fixed_baseline_accuracy=float(np.mean(y_test == baseline)),
        selector_mean_score_regret=regret(prediction),
        fixed_mean_score_regret=regret([baseline] * len(test_indices)),
        prediction_seconds_per_case=prediction_seconds,
        test_mean_feature_seconds=float(np.mean([records[i].feature_seconds for i in test_indices])),
    )
    return model, evaluation


def save_selector_model(
    path: str | Path,
    model: DecisionTreeClassifier,
    evaluation: SelectorEvaluation,
    training_config: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "format_version": 2,
        "scoring_protocol": "independent_reference_comparable_objective_v2",
        "model": model,
        "feature_names": FEATURE_NAMES,
        "evaluation": asdict(evaluation),
        "training_config": training_config or {},
    }
    joblib.dump(bundle, destination)
    metadata_path = destination.with_suffix(".json")
    metadata_path.write_text(
        json.dumps({key: value for key, value in bundle.items() if key != "model"},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination


def load_selector_model(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if bundle.get("format_version") != 2:
        raise ValueError("旧选择器使用历史评分口径，请用当前代码重新训练 v2 模型")
    if bundle.get("feature_names") != FEATURE_NAMES:
        raise ValueError("模型特征版本与当前程序不一致")
    return bundle


def predict_algorithm(dataset: SparseDataset, model_or_bundle) -> tuple[str, float, dict[str, float]]:
    bundle = model_or_bundle if isinstance(model_or_bundle, dict) else {"model": model_or_bundle}
    model = bundle["model"]
    vector = np.asarray([feature_vector(dataset)], dtype=np.float64)
    prediction = str(model.predict(vector)[0])
    confidence = 1.0
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(vector)[0]))
    return prediction, confidence, extract_selector_features(dataset)


def predict_algorithm_from_features(
    features: dict[str, float], model_or_bundle
) -> tuple[str, float]:
    """Predict directly from a validated feature dictionary."""
    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise ValueError(f"missing selector features: {', '.join(missing)}")
    bundle = model_or_bundle if isinstance(model_or_bundle, dict) else {"model": model_or_bundle}
    model = bundle["model"]
    vector = np.asarray([[features[name] for name in FEATURE_NAMES]], dtype=np.float64)
    prediction = str(model.predict(vector)[0])
    confidence = 1.0
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(vector)[0]))
    return prediction, confidence


def save_training_records(path: str | Path, records: list[SelectorTrainingRecord]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset_name", *FEATURE_NAMES, "winner", "winner_score", "runner_up", "runner_up_score",
              "dataset_group", "algorithm_scores", "feature_seconds"]
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {"dataset_name": record.dataset_name}
            row.update(dict(zip(FEATURE_NAMES, record.features)))
            row.update({
                "winner": record.winner,
                "winner_score": record.winner_score,
                "runner_up": record.runner_up,
                "runner_up_score": record.runner_up_score,
                "dataset_group": record.dataset_group,
                "algorithm_scores": json.dumps(record.algorithm_scores, ensure_ascii=False),
                "feature_seconds": record.feature_seconds,
            })
            writer.writerow(row)
    return destination


def load_training_records(path: str | Path) -> list[SelectorTrainingRecord]:
    """Load previously benchmarked selector records without rerunning OT algorithms."""
    records = []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            records.append(SelectorTrainingRecord(
                dataset_name=row["dataset_name"],
                features=[float(row[name]) for name in FEATURE_NAMES],
                winner=row["winner"],
                winner_score=float(row["winner_score"]),
                runner_up=row.get("runner_up", ""),
                runner_up_score=float(row.get("runner_up_score") or 0.0),
                dataset_group=row.get("dataset_group", ""),
                algorithm_scores=json.loads(row.get("algorithm_scores") or "{}"),
                feature_seconds=float(row.get("feature_seconds") or 0),
            ))
    return records
