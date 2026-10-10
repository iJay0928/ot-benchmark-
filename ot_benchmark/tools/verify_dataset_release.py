#!/usr/bin/env python3
"""Verify a generated HD-Sparse-OT Benchmark release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path) -> list[str]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return [f"missing {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    records = manifest.get("datasets", [])
    if len(records) < 5:
        errors.append(f"expected at least 5 datasets, found {len(records)}")
    for record in records:
        path = root / record["file"]
        if not path.exists():
            errors.append(f"missing {path.name}")
            continue
        if sha256(path) != record["sha256"]:
            errors.append(f"checksum mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as data:
            source = data["source_dist"]
            target = data["target_dist"]
            cost = data["cost_matrix"]
            csr = csr_matrix(
                (data["cost_csr_data"], data["cost_csr_indices"], data["cost_csr_indptr"]),
                shape=tuple(data["cost_csr_shape"]),
            )
            if source.ndim != 1 or target.ndim != 1:
                errors.append(f"distribution shape invalid: {path.name}")
            if not np.isclose(source.sum(), 1.0, atol=1e-6) or not np.isclose(target.sum(), 1.0, atol=1e-6):
                errors.append(f"distribution does not sum to one: {path.name}")
            if cost.shape != (source.size, target.size) or csr.shape != cost.shape:
                errors.append(f"cost shape invalid: {path.name}")
            if not np.isfinite(cost).all() or not np.isfinite(source).all() or not np.isfinite(target).all():
                errors.append(f"non-finite values: {path.name}")
            csr_dense = csr.toarray()
            if manifest.get("cost_csr_semantics") == "lossless_dense_equivalent":
                if not np.allclose(csr_dense, cost, atol=1e-12, rtol=0):
                    errors.append(f"lossless dense/CSR mismatch: {path.name}")
            if np.any(source < 0) or np.any(target < 0):
                errors.append(f"negative probability: {path.name}")
            actual = 1 - (np.count_nonzero(source) + np.count_nonzero(target)) / (source.size + target.size)
            if not np.isclose(actual, record["actual_sparsity"], atol=1e-12):
                errors.append(f"sparsity metadata mismatch: {path.name}")
            if data["support_points"].shape != (source.size, record["geometry_dim"]):
                errors.append(f"geometry shape mismatch: {path.name}")
            if record["ground_truth_mode"] == "exact" and not bool(data["ground_truth_is_exact"]):
                errors.append(f"exact reference unavailable: {path.name}")
            if "fallback" in str(data["ground_truth_method"]).lower():
                errors.append(f"OT reference unavailable: {path.name}")
            if not np.isfinite(data["ground_truth_distance"]).all():
                errors.append(f"OT reference non-finite: {path.name}")
            # The benchmark CSR representation intentionally drops very large
            # costs (the cutoff is documented in generate_cost_matrix).
            if not np.allclose(csr_dense[csr_dense != 0], cost[csr_dense != 0], atol=1e-12):
                errors.append(f"dense/CSR retained-entry mismatch: {path.name}")
            if np.any(csr_dense < 0) or np.any(csr_dense > cost + 1e-12):
                errors.append(f"CSR contains invalid costs: {path.name}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    args = parser.parse_args()
    errors = verify(args.release)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"OK: verified {len(json.loads((args.release / 'manifest.json').read_text(encoding='utf-8'))['datasets'])} datasets")


if __name__ == "__main__":
    main()
