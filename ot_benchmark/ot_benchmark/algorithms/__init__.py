"""Modular OT algorithms for the benchmark system."""

from .registry import ALGORITHMS, get_algorithm, register_algorithm, list_algorithms

__all__ = ["ALGORITHMS", "get_algorithm", "register_algorithm", "list_algorithms"]
