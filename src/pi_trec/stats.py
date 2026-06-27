"""Shared statistics helpers."""

from __future__ import annotations

import math

from scipy.stats import kendalltau


def kendall_tau(xs: list[float], ys: list[float]) -> float:
    """Kendall ``tau-b`` (ties-aware) via SciPy, matching the reference harness."""
    pairs = [(x, y) for x, y in zip(xs, ys, strict=False) if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 2:
        return math.nan
    tau = kendalltau([x for x, _ in pairs], [y for _, y in pairs], variant="b").statistic
    return float(tau)
