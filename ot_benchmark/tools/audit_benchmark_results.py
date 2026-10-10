#!/usr/bin/env python3
"""Audit a benchmark result directory and write a publication checklist.

This audit does not change experiment results. It checks completeness and
aggregates by algorithm across scenarios so repeated scenario rows cannot be
mistaken for a global algorithm ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def _float(value: str):
    if value in (None, "", "None", "null"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def audit(result_dir: Path) -> dict:
    raw_path = result_dir / "raw_results.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    with raw_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("raw_results.csv is empty")

    statuses = Counter(row.get("status", "unknown") for row in rows)
    scenarios = sorted({row.get("scenario_id", "") for row in rows})
    algorithms = sorted({row.get("algorithm", "") for row in rows})
    by_algorithm: dict[str, dict] = {}
    for algorithm in algorithms:
        selected = [r for r in rows if r.get("algorithm") == algorithm]
        success = [r for r in selected if r.get("status") == "success"]
        runtimes = [_float(r.get("runtime_seconds")) for r in success]
        memories = [_float(r.get("peak_memory_mb")) for r in success]
        gt_errors = [_float(r.get("ground_truth_error")) for r in success]
        by_algorithm[algorithm] = {
            "ranking_groups": sorted({r.get("ranking_group", "") for r in selected}),
            "total_runs": len(selected),
            "successful_runs": len(success),
            "success_rate": len(success) / len(selected),
            "mean_runtime_seconds": sum(x for x in runtimes if x is not None) / max(1, sum(x is not None for x in runtimes)),
            "mean_peak_memory_mb": sum(x for x in memories if x is not None) / max(1, sum(x is not None for x in memories)),
            "ground_truth_error_observations": sum(x is not None for x in gt_errors),
            "failure_types": dict(Counter(r.get("failure_type") or r.get("status") for r in selected if r.get("status") != "success")),
        }

    scenario_algorithm_counts = Counter((r.get("scenario_id"), r.get("algorithm")) for r in rows)
    incomplete_pairs = [
        {"scenario_id": sid, "algorithm": algo, "runs": count}
        for (sid, algo), count in sorted(scenario_algorithm_counts.items())
        if count == 0
    ]
    ground_truth_rows = sum(_float(r.get("ground_truth_error")) is not None for r in rows)
    reference_rows = sum(_float(r.get("reference_error")) is not None for r in rows)
    audit_result = {
        "result_directory": str(result_dir),
        "raw_rows": len(rows),
        "scenario_count": len(scenarios),
        "algorithm_count": len(algorithms),
        "scenarios": scenarios,
        "algorithms": algorithms,
        "status_counts": dict(statuses),
        "success_rate": statuses.get("success", 0) / len(rows),
        "ground_truth_error_observations": ground_truth_rows,
        "reference_error_observations": reference_rows,
        "by_algorithm": by_algorithm,
        "incomplete_scenario_algorithm_pairs": incomplete_pairs,
        "warnings": [],
    }
    if not (result_dir / "environment.json").exists():
        audit_result["warnings"].append("environment.json is missing")
    if not (result_dir / "protocol.md").exists():
        audit_result["warnings"].append("protocol.md is missing")
    if ground_truth_rows == 0 and reference_rows == 0:
        audit_result["warnings"].append("no ground-truth or reference error observations")
    if statuses.get("success", 0) < len(rows):
        audit_result["warnings"].append("failed/not-applicable runs require explicit explanation in the final report")
    return audit_result


def render_markdown(result: dict) -> str:
    lines = [
        "# Benchmark Result Audit",
        "",
        f"- Raw rows: **{result['raw_rows']}**",
        f"- Scenarios: **{result['scenario_count']}**",
        f"- Algorithms: **{result['algorithm_count']}**",
        f"- Overall success rate: **{result['success_rate']:.1%}**",
        f"- Ground-truth error observations: **{result['ground_truth_error_observations']}**",
        f"- Reference-error observations: **{result['reference_error_observations']}**",
        "",
        "## Aggregated algorithm view",
        "",
        "| Algorithm | Group | Runs | Success rate | Mean runtime (s) | Mean memory (MB) | GT error observations |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["by_algorithm"].items():
        lines.append(
            f"| {name} | {', '.join(row['ranking_groups'])} | {row['total_runs']} | "
            f"{row['success_rate']:.1%} | {row['mean_runtime_seconds']:.4f} | "
            f"{row['mean_peak_memory_mb']:.3f} | {row['ground_truth_error_observations']} |"
        )
    lines += ["", "## Status counts", ""]
    for status, count in sorted(result["status_counts"].items()):
        lines.append(f"- `{status}`: {count}")
    lines += ["", "## Warnings", ""]
    lines.extend(f"- {warning}" for warning in result["warnings"] or ["None"])
    lines += ["", "The algorithm table is aggregated across scenarios; it is not a replacement for scenario-level results.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result = audit(args.result_dir)
    (args.result_dir / "final_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.result_dir / "final_audit.md").write_text(render_markdown(result), encoding="utf-8")
    print(f"Audited {result['raw_rows']} rows; success rate={result['success_rate']:.1%}")
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
