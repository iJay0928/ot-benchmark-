#!/usr/bin/env python3
"""Build the versioned HD-Sparse-OT Benchmark dataset release.

The release is intentionally generated from code and a fixed seed manifest so
that the files can be regenerated instead of being an opaque binary dump.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ot_benchmark.data_generator import create_benchmark_dataset, diagnose_dataset


RELEASE_VERSION = "0.3.0"
DATASET_SPECS = [
    {
        "id": "HDSPARSE-SYN-0200-DENSE",
        "name": "Synth-PL-200",
        "n_samples": 200,
        "sparsity": 0.50,
        "source": "uniform",
        "target": "uniform",
        "structure": "random",
        "noise": 0.01,
        "geometry_dim": 8,
        "ground_truth_mode": "exact",
        "seed": 1001,
        "scenario": "low-dimensional dense precision reference",
    },
    {
        "id": "HDSPARSE-SYN-0300-SPARSE",
        "name": "Synth-PL-300",
        "n_samples": 300,
        "sparsity": 0.85,
        "source": "zipf",
        "target": "zipf",
        "structure": "random",
        "noise": 0.03,
        "geometry_dim": 8,
        "ground_truth_mode": "exact",
        "seed": 1002,
        "scenario": "sparse long-tail same-distribution",
    },
    {
        "id": "HDSPARSE-SYN-0500-SHIFT",
        "name": "Synth-PL-500-Shift",
        "n_samples": 500,
        "sparsity": 0.90,
        "source": "zipf",
        "target": "pareto",
        "structure": "random",
        "noise": 0.05,
        "geometry_dim": 8,
        "ground_truth_mode": "sinkhorn_ref",
        "seed": 1003,
        "scenario": "cross-distribution shift",
    },
    {
        "id": "HDSPARSE-SYN-0800-BLOCK",
        "name": "Synth-PL-800-Block",
        "n_samples": 800,
        "sparsity": 0.92,
        "source": "zipf",
        "target": "zipf",
        "structure": "block",
        "noise": 0.04,
        "geometry_dim": 8,
        "ground_truth_mode": "sinkhorn_ref",
        "seed": 1004,
        "scenario": "structured block sparsity",
    },
    {
        "id": "HDSPARSE-SYN-1000-LONGTAIL",
        "name": "Synth-PL-1000-LongTail",
        "n_samples": 1000,
        "sparsity": 0.97,
        "source": "pareto",
        "target": "zipf",
        "structure": "hierarchical",
        "noise": 0.05,
        "geometry_dim": 8,
        "ground_truth_mode": "sinkhorn_ref",
        "seed": 1005,
        "scenario": "high-dimensional extreme long-tail",
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def build_release(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in DATASET_SPECS:
        dataset = create_benchmark_dataset(
            n_samples=spec["n_samples"],
            sparsity=spec["sparsity"],
            noise_level=spec["noise"],
            source_dist_type=spec["source"],
            target_dist_type=spec["target"],
            random_state=spec["seed"],
            name=spec["name"],
            structure=spec["structure"],
            sparsity_semantics="zero_ratio",
            ground_truth_mode=spec["ground_truth_mode"],
            geometry_dim=spec["geometry_dim"],
        )
        filename = f"{spec['id']}.npz"
        path = output_dir / filename
        np.savez_compressed(
            path,
            name=np.array(dataset.name),
            dataset_id=np.array(spec["id"]),
            source_dist=dataset.source_dist,
            target_dist=dataset.target_dist,
            cost_matrix=dataset.cost_matrix,
            cost_csr_data=dataset.cost_matrix_csr.data,
            cost_csr_indices=dataset.cost_matrix_csr.indices,
            cost_csr_indptr=dataset.cost_matrix_csr.indptr,
            cost_csr_shape=np.asarray(dataset.cost_matrix_csr.shape, dtype=np.int64),
            support_points=dataset.support_points,
            ground_truth_distance=np.asarray(dataset.ground_truth_distance),
            ground_truth_method=np.array(dataset.ground_truth_method),
            ground_truth_is_exact=np.asarray(dataset.ground_truth_is_exact),
            dimension=np.asarray(dataset.dimension),
            geometry_dim=np.asarray(dataset.geometry_dim),
            sparsity=np.asarray(dataset.sparsity),
            noise_level=np.asarray(dataset.noise_level),
            dataset_seed=np.asarray(dataset.dataset_seed),
        )
        diagnosis = {k: _json_value(v) for k, v in diagnose_dataset(dataset).items()}
        records.append({
            **spec,
            "file": filename,
            "sha256": _sha256(path),
            "file_size_bytes": path.stat().st_size,
            "actual_dimension": int(dataset.dimension),
            "actual_sparsity": float(dataset.sparsity),
            "ground_truth_method_actual": dataset.ground_truth_method,
            "ground_truth_is_exact_actual": bool(dataset.ground_truth_is_exact),
            "diagnosis": diagnosis,
        })

    manifest = {
        "dataset_name": "HD-Sparse-OT Benchmark",
        "release_version": RELEASE_VERSION,
        "format": "NumPy NPZ with dense and CSR cost representations",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "ot_benchmark.tools.build_dataset_release",
        "generator_python": platform.python_version(),
        "sparsity_semantics": "zero_ratio",
        "cost_csr_semantics": "lossless_dense_equivalent",
        "dimension_semantics": "mass vector length / number of support points",
        "geometry_dimension_semantics": "coordinates per support point",
        "structure_semantics": "mass support layout in index order, not geometric clustering",
        "status": "internal validation candidate; not a public release",
        "generator_sha256": _sha256(PROJECT_ROOT / "ot_benchmark/data_generator.py"),
        "reproducibility": {
            "seed_is_fixed": True,
            "regenerate_command": "python tools/build_dataset_release.py --output dataset_release/hd_sparse_ot_v0.3.0",
        },
        "datasets": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(_readme(), encoding="utf-8")
    return manifest


def _readme() -> str:
    return """# HD-Sparse-OT Benchmark v0.3.0

This is the reproducible synthetic release accompanying the OTBench project.
It contains five fixed-seed subsets covering dense, random sparse, shifted,
block-sparse and hierarchical long-tail settings.

## File format

Each `.npz` file contains source and target probability vectors, support
points, a dense ground-cost matrix, the same cost matrix in CSR arrays, and
ground-truth/reference metadata. The `manifest.json` file records the exact
generator parameters, SHA-256 checksum and measured dataset statistics.

```python
import numpy as np
from scipy.sparse import csr_matrix

z = np.load("HDSPARSE-SYN-0300-SPARSE.npz", allow_pickle=False)
cost = csr_matrix(
    (z["cost_csr_data"], z["cost_csr_indices"], z["cost_csr_indptr"]),
    shape=tuple(z["cost_csr_shape"]),
)
```

## Reproduction

From the `ot_benchmark` project directory run:

```powershell
python tools/build_dataset_release.py --output dataset_release/hd_sparse_ot_v0.3.0
```

The release is synthetic and is intended for algorithm benchmarking; it does
not redistribute third-party raw data.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "dataset_release" / "hd_sparse_ot_v0.3.0",
    )
    args = parser.parse_args()
    manifest = build_release(args.output)
    print(f"Generated {len(manifest['datasets'])} datasets in {args.output}")
    for record in manifest["datasets"]:
        print(f"  {record['id']}: {record['actual_dimension']}D, sparsity={record['actual_sparsity']:.3f}")


if __name__ == "__main__":
    main()
