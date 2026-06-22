"""Approximation-ratio / regret / budget metrics and summary statistics."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def approximation_ratio(cut_value: float, max_cut_value: float) -> float:
    return cut_value / max(1e-12, max_cut_value)


def regret(ratio: float) -> float:
    return max(0.0, 1.0 - ratio)


def queries_to_target(history: List[tuple], target: float) -> int:
    """First query index at which the running-best ratio reached ``target``.

    Returns -1 if the target was never reached within the recorded budget.
    """
    for q, best in history:
        if best >= target:
            return q
    return -1


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "n": 0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {"mean": round(mean, 6), "std": round(std, 6), "ci95": round(ci95, 6), "n": int(arr.size)}
