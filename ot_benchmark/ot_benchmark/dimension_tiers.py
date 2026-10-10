"""Dimension tiers used by the sparse OT benchmark."""

from __future__ import annotations

from typing import Any


DIMENSION_TIERS: dict[str, dict[str, Any]] = {
    "standard": {
        "label": "基础高维",
        "target_dim": 1000,
        "description": "1,000 维：日常开发和算法正确性比较",
        "preferred_bow": ("kos", "nips"),
    },
    "high": {
        "label": "较高维",
        "target_dim": 5000,
        "description": "5,000 维：观察算法在明显高维条件下的时间和内存变化",
        "preferred_bow": ("nips", "kos"),
    },
    "ultra": {
        "label": "超高维",
        "target_dim": 10000,
        "description": "10,000 维：测试高维稀疏算法的适用边界",
        "preferred_bow": ("nips", "enron"),
    },
}


def tier_names(selection: str) -> list[str]:
    """Expand a tier selection into concrete tier names."""
    key = (selection or "standard").lower()
    if key == "all":
        return list(DIMENSION_TIERS)
    if key not in DIMENSION_TIERS:
        raise ValueError(f"unknown dimension tier: {selection}")
    return [key]


def tier_target_dim(selection: str) -> int:
    """Return the target dimension for a single tier selection."""
    names = tier_names(selection)
    if len(names) != 1:
        raise ValueError("all does not have one target dimension")
    return int(DIMENSION_TIERS[names[0]]["target_dim"])


def tier_label(selection: str) -> str:
    if selection == "all":
        return "全部档位"
    return str(DIMENSION_TIERS[selection]["label"])
