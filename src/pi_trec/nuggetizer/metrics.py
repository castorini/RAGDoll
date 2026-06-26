"""Nugget coverage scoring and run-level correlation.

Faithful port of the scoring used in the AutoPi reproduction harness
(``trec2024_compare.py``): per-cell coverage in ``[0, 1]`` over four metrics
(strict-vital, strict-all, vital, all), run-level aggregation, and a
SciPy-backed Kendall ``tau`` between a system and a reference assignment.

A *cell* is one ``(qid, run_id)`` answer judged against its nugget list. Inputs
are assignment JSONL rows shaped like::

    {"qid": ..., "run_id": ..., "nuggets": [{"importance": "vital"|"okay",
                                             "assignment": "support"|...}, ...]}
"""

from __future__ import annotations

import csv
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scipy.stats import kendalltau

from pi_trec.config import NuggetMetricsConfig
from pi_trec.jsonl import read_jsonl

logger = logging.getLogger("pi_trec.nuggetizer.metrics")

# Metric short name -> per-cell column produced by ``calculate_scores``.
METRIC_COLUMNS = {
    "V_strict": "strict_vital_score",
    "A_strict": "strict_all_score",
    "V": "vital_score",
    "A": "all_score",
}
METRICS = tuple(METRIC_COLUMNS)
SUPPORT_LABELS = ("support", "partial_support", "not_support", "failed")


def normalize_assignment(label: str | None) -> str:
    """Lower-case an assignment label; map the legacy ``fail`` to ``failed``."""
    value = str(label or "").strip().lower()
    return "failed" if value == "fail" else value


def calculate_scores(nuggets: list[dict[str, Any]]) -> dict[str, float]:
    """Coverage in ``[0, 1]`` for the four metrics (reference semantics).

    ``support`` counts 1.0; ``partial_support`` counts 0.5 only in the
    non-strict variants. ``vital`` metrics are restricted to vital nuggets.
    """
    vital = [n for n in nuggets if n.get("importance") == "vital"]
    all_nuggets = list(nuggets)

    def score(items: list[dict[str, Any]], strict: bool) -> float:
        if not items:
            return 0.0
        total = 0.0
        for nugget in items:
            assignment = normalize_assignment(nugget.get("assignment"))
            if assignment == "support":
                total += 1.0
            elif assignment == "partial_support" and not strict:
                total += 0.5
        return total / len(items)

    return {
        "strict_vital_score": score(vital, True),
        "strict_all_score": score(all_nuggets, True),
        "vital_score": score(vital, False),
        "all_score": score(all_nuggets, False),
    }


def mean(values: Iterable[float]) -> float:
    xs = [x for x in values if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else math.nan


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    xs = sorted(values)
    pos = (len(xs) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def kendall_tau(xs: list[float], ys: list[float]) -> float:
    """Kendall ``tau-b`` (ties-aware) via SciPy, matching the reference harness."""
    pairs = [(x, y) for x, y in zip(xs, ys, strict=False) if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 2:
        return math.nan
    tau = kendalltau([x for x, _ in pairs], [y for _, y in pairs], variant="b").statistic
    return float(tau)


def qid_of(row: dict[str, Any]) -> str:
    return str(row.get("qid") or row.get("topic_id") or row.get("query_id") or "")


def run_id_of(row: dict[str, Any]) -> str:
    return str(row.get("run_id") or row.get("runtag") or row.get("run") or "")


def response_length(row: dict[str, Any]) -> float:
    value = row.get("response_length")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(row.get("answer_text") or "")
    return float(len(text.split()))


@dataclass(frozen=True)
class CellMetric:
    qid: str
    run_id: str
    strict_vital_score: float
    strict_all_score: float
    vital_score: float
    all_score: float
    nugget_count: int
    vital_count: int
    failed_count: int
    response_length: float

    def metric(self, name: str) -> float:
        return float(getattr(self, METRIC_COLUMNS[name]))


def cell_metric(row: dict[str, Any]) -> CellMetric:
    nuggets = row.get("nuggets") or []
    scores = calculate_scores(nuggets)
    return CellMetric(
        qid=qid_of(row),
        run_id=run_id_of(row),
        strict_vital_score=scores["strict_vital_score"],
        strict_all_score=scores["strict_all_score"],
        vital_score=scores["vital_score"],
        all_score=scores["all_score"],
        nugget_count=len(nuggets),
        vital_count=sum(1 for n in nuggets if n.get("importance") == "vital"),
        failed_count=sum(1 for n in nuggets if normalize_assignment(n.get("assignment")) == "failed"),
        response_length=response_length(row),
    )


def label_distribution(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = {label: 0 for label in SUPPORT_LABELS}
    counts["other"] = 0
    for row in rows:
        for nugget in row.get("nuggets") or []:
            label = normalize_assignment(nugget.get("assignment"))
            counts[label if label in counts else "other"] += 1
    return counts


def run_means(cells: list[CellMetric]) -> dict[str, dict[str, Any]]:
    """Average each metric within a run across its topics, keyed by ``run_id``."""
    grouped: dict[str, list[CellMetric]] = {}
    for cell in cells:
        grouped.setdefault(cell.run_id, []).append(cell)
    out: dict[str, dict[str, Any]] = {}
    for run_id, run_cells in grouped.items():
        row: dict[str, Any] = {"run_id": run_id, "n_topics": len(run_cells)}
        for column in METRIC_COLUMNS.values():
            row[column] = mean(getattr(cell, column) for cell in run_cells)
        out[run_id] = row
    return out


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: float) -> float | str:
    return "" if isinstance(value, float) and math.isnan(value) else round(value, 6)


def compute_metrics(config: NuggetMetricsConfig) -> None:
    """Score an assignment file; optionally correlate against a reference."""
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    system_rows = list(read_jsonl(config.input_file))
    cells = [cell_metric(row) for row in system_rows]

    cell_columns = [
        "qid", "run_id", "strict_vital_score", "strict_all_score", "vital_score",
        "all_score", "nugget_count", "vital_count", "failed_count", "response_length",
    ]
    write_csv(
        output_dir / "cell_metrics.csv",
        cell_columns,
        [
            {
                "qid": c.qid, "run_id": c.run_id,
                "strict_vital_score": _fmt(c.strict_vital_score),
                "strict_all_score": _fmt(c.strict_all_score),
                "vital_score": _fmt(c.vital_score), "all_score": _fmt(c.all_score),
                "nugget_count": c.nugget_count, "vital_count": c.vital_count,
                "failed_count": c.failed_count, "response_length": _fmt(c.response_length),
            }
            for c in cells
        ],
    )

    sys_runs = run_means(cells)
    run_columns = ["run_id", "n_topics", *METRIC_COLUMNS.values()]
    write_csv(
        output_dir / "run_metrics.csv",
        run_columns,
        [
            {"run_id": r["run_id"], "n_topics": r["n_topics"],
             **{col: _fmt(r[col]) for col in METRIC_COLUMNS.values()}}
            for r in sorted(sys_runs.values(), key=lambda r: r["run_id"])
        ],
    )

    dist = label_distribution(system_rows)
    write_csv(
        output_dir / "label_distribution.csv",
        ["label", "count"],
        [{"label": label, "count": count} for label, count in dist.items()],
    )

    logger.info(
        "scored cells=%d runs=%d -> %s", len(cells), len(sys_runs), output_dir
    )
    print(f"scored cells={len(cells)} runs={len(sys_runs)} output_dir={output_dir}")
    total = sum(dist.values()) or 1
    print("label distribution: " + ", ".join(f"{k}={v} ({100 * v / total:.1f}%)" for k, v in dist.items() if v))

    if config.reference is not None:
        _write_correlations(config.reference, sys_runs, cells, output_dir)


def _write_correlations(
    reference: Path,
    sys_runs: dict[str, dict[str, Any]],
    sys_cells: list[CellMetric],
    output_dir: Path,
) -> None:
    ref_cells = [cell_metric(row) for row in read_jsonl(reference)]
    ref_runs = run_means(ref_cells)
    shared_runs = sorted(set(sys_runs) & set(ref_runs))
    shared_cell_keys = {(c.qid, c.run_id) for c in sys_cells} & {(c.qid, c.run_id) for c in ref_cells}

    sys_cell_by_key = {(c.qid, c.run_id): c for c in sys_cells}
    ref_cell_by_key = {(c.qid, c.run_id): c for c in ref_cells}

    rows: list[dict[str, Any]] = []
    for name, column in METRIC_COLUMNS.items():
        run_tau = kendall_tau(
            [sys_runs[r][column] for r in shared_runs],
            [ref_runs[r][column] for r in shared_runs],
        )
        cell_tau = kendall_tau(
            [sys_cell_by_key[k].metric(name) for k in sorted(shared_cell_keys)],
            [ref_cell_by_key[k].metric(name) for k in sorted(shared_cell_keys)],
        )
        rows.append(
            {
                "metric": name, "column": column,
                "run_tau": _fmt(run_tau), "n_runs": len(shared_runs),
                "topic_run_tau": _fmt(cell_tau), "n_cells": len(shared_cell_keys),
            }
        )
    write_csv(
        output_dir / "correlations.csv",
        ["metric", "column", "run_tau", "n_runs", "topic_run_tau", "n_cells"],
        rows,
    )
    print(f"correlations vs {reference.name}: " + ", ".join(f"{r['metric']} run_tau={r['run_tau']}" for r in rows))
