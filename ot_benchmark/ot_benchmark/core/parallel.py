"""Parallel benchmark execution using multiprocessing."""

from __future__ import annotations

import os
import gc
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from .logging_config import get_logger

logger = get_logger("parallel")


class ParallelBenchmarkRunner:
    """Run dataset benchmarks in parallel using ProcessPoolExecutor.

    Each dataset is evaluated as an independent task. Results
    are collected and returned in order.
    """

    def __init__(
        self,
        n_workers: int = 0,
        timeout_per_task: float = 300.0,
        use_gpu: bool = False,
    ):
        """Initialize parallel runner.

        Args:
            n_workers: Number of worker processes. 0 = CPU count.
            timeout_per_task: Timeout per task in seconds.
            use_gpu: If True, force sequential execution (GPU not shareable).
        """
        self.n_workers = n_workers if n_workers > 0 else os.cpu_count() or 4
        if use_gpu:
            self.n_workers = max(1, self.n_workers // 2)
            logger.info("GPU mode: limiting to %d workers", self.n_workers)
        self.timeout = timeout_per_task

    def run_parallel(
        self,
        tasks: list[tuple[Any, ...]],
        worker_fn: callable,
        desc: str = "Benchmarking",
    ) -> list[Any]:
        """Execute tasks in parallel.

        Args:
            tasks: List of (args,) tuples for worker_fn.
            worker_fn: Function to execute per task.
            desc: Description for progress reporting.

        Returns:
            Results in the same order as tasks.
        """
        if not tasks:
            return []

        n_tasks = len(tasks)
        n_workers = min(self.n_workers, n_tasks)

        if n_workers <= 1:
            logger.info("%s: running %d tasks sequentially", desc, n_tasks)
            results = []
            for task in tasks:
                results.append(worker_fn(*task))
            return results

        logger.info(
            "%s: running %d tasks with %d workers (timeout=%ds)",
            desc, n_tasks, n_workers, self.timeout,
        )

        results: dict[int, Any] = {}
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(worker_fn, *task): idx
                for idx, task in enumerate(tasks)
            }

            iterator = as_completed(future_to_idx)
            if use_tqdm:
                iterator = tqdm(
                    iterator, total=n_tasks, desc=desc, unit="task"
                )

            for future in iterator:
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result(timeout=self.timeout)
                except Exception as e:
                    logger.error("Task %d failed: %s", idx, e)
                    results[idx] = None

        return [results.get(i) for i in range(n_tasks)]

    def cleanup(self):
        """Force garbage collection after parallel run."""
        gc.collect()
