from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pi_trec.config import RubricScoreConfig
from pi_trec.jsonl import read_jsonl
from pi_trec.nuggetizer.metrics import _fmt, kendall_tau, mean, qid_of, response_length, run_id_of, write_csv
from pi_trec.rubric.prompts import CRITERION_TYPES

logger = logging.getLogger("pi_trec.rubric.metrics")

VERDICT_VALUES = {"satisfied": 1.0, "partially_satisfied": 0.5, "not_satisfied": 0.0}


def normalize_verdict(verdict: Any) -> str:
    return str(verdict or "").strip().lower().replace(" ", "_")


def verdict_value(verdict: Any, *, binary: bool) -> float:
    """Ternary value in ``{1, 0.5, 0}``; binary collapses partial to ``0``."""
    value = VERDICT_VALUES.get(normalize_verdict(verdict), 0.0)
    return (1.0 if value >= 1.0 else 0.0) if binary else value


def score_response(criteria: list[dict[str, Any]], *, binary: bool = False) -> float:
    """Normalized weighted score in [0, 1]; ``nan`` if there is no positive weight."""
    numerator = 0.0
    positive_weight = 0.0
    for criterion in criteria:
        weight = float(criterion.get("weight", 0))
        numerator += weight * verdict_value(criterion.get("verdict"), binary=binary)
        if weight > 0:
            positive_weight += weight
    return numerator / positive_weight if positive_weight else math.nan


def _is_failure(criterion: dict[str, Any]) -> bool:
    """A criterion the answer did not fully satisfy."""
    return verdict_value(criterion.get("verdict"), binary=False) < 1.0


@dataclass(frozen=True)
class CellScore:
    qid: str
    run_id: str
    ternary_score: float
    binary_score: float
    n_criteria: int
    n_mandatory_fail: int
    response_length: float


def cell_score(row: dict[str, Any]) -> CellScore:
    criteria = row.get("criteria") or []
    return CellScore(
        qid=qid_of(row),
        run_id=run_id_of(row),
        ternary_score=score_response(criteria, binary=False),
        binary_score=score_response(criteria, binary=True),
        n_criteria=len(criteria),
        n_mandatory_fail=sum(1 for c in criteria if c.get("tier") == "mandatory" and _is_failure(c)),
        response_length=response_length(row),
    )


def run_scores(cells: list[CellScore]) -> dict[str, dict[str, Any]]:
    """Average ternary/binary scores within a run across its topics, keyed by ``run_id``."""
    grouped: dict[str, list[CellScore]] = {}
    for cell in cells:
        grouped.setdefault(cell.run_id, []).append(cell)
    out: dict[str, dict[str, Any]] = {}
    for run_id, run_cells in grouped.items():
        out[run_id] = {
            "run_id": run_id,
            "n_topics": len(run_cells),
            "ternary_score": mean(c.ternary_score for c in run_cells),
            "binary_score": mean(c.binary_score for c in run_cells),
        }
    return out


def category_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per criterion-type failure rate across all cells (paper Eq. 2, simplified)."""
    totals = {t: 0 for t in CRITERION_TYPES}
    fails = {t: 0 for t in CRITERION_TYPES}
    for row in rows:
        for criterion in row.get("criteria") or []:
            ctype = criterion.get("type", "explicit")
            if ctype not in totals:
                continue
            totals[ctype] += 1
            if _is_failure(criterion):
                fails[ctype] += 1
    return [
        {
            "type": t,
            "n_total": totals[t],
            "n_failed": fails[t],
            "failure_rate": _fmt(fails[t] / totals[t] if totals[t] else math.nan),
        }
        for t in CRITERION_TYPES
    ]


def compute_scores(config: RubricScoreConfig) -> None:
    """Score a graded rubric file; optionally correlate against a reference."""
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(read_jsonl(config.input_file))
    cells = [cell_score(row) for row in rows]

    write_csv(
        output_dir / "cell_scores.csv",
        ["qid", "run_id", "ternary_score", "binary_score", "n_criteria", "n_mandatory_fail", "response_length"],
        [
            {
                "qid": c.qid,
                "run_id": c.run_id,
                "ternary_score": _fmt(c.ternary_score),
                "binary_score": _fmt(c.binary_score),
                "n_criteria": c.n_criteria,
                "n_mandatory_fail": c.n_mandatory_fail,
                "response_length": _fmt(c.response_length),
            }
            for c in cells
        ],
    )

    sys_runs = run_scores(cells)
    write_csv(
        output_dir / "run_scores.csv",
        ["run_id", "n_topics", "ternary_score", "binary_score"],
        [
            {
                "run_id": r["run_id"],
                "n_topics": r["n_topics"],
                "ternary_score": _fmt(r["ternary_score"]),
                "binary_score": _fmt(r["binary_score"]),
            }
            for r in sorted(sys_runs.values(), key=lambda r: r["run_id"])
        ],
    )

    write_csv(output_dir / "category_failures.csv", ["type", "n_total", "n_failed", "failure_rate"], category_failures(rows))

    _write_correlations(config.reference, sys_runs, output_dir)

    logger.info("scored cells=%d runs=%d -> %s", len(cells), len(sys_runs), output_dir)
    print(f"scored cells={len(cells)} runs={len(sys_runs)} output_dir={output_dir}")


def _write_correlations(reference: Path | None, sys_runs: dict[str, dict[str, Any]], output_dir: Path) -> None:
    shared = sorted(sys_runs)
    rows: list[dict[str, Any]] = [
        {
            "comparison": "ternary_vs_binary",
            "run_tau": _fmt(
                kendall_tau(
                    [sys_runs[r]["ternary_score"] for r in shared],
                    [sys_runs[r]["binary_score"] for r in shared],
                )
            ),
            "n_runs": len(shared),
        }
    ]
    if reference is not None:
        ref_runs = run_scores([cell_score(row) for row in read_jsonl(reference)])
        shared_ref = sorted(set(sys_runs) & set(ref_runs))
        for column in ("ternary_score", "binary_score"):
            rows.append(
                {
                    "comparison": f"{column}_vs_reference",
                    "run_tau": _fmt(
                        kendall_tau(
                            [sys_runs[r][column] for r in shared_ref],
                            [ref_runs[r][column] for r in shared_ref],
                        )
                    ),
                    "n_runs": len(shared_ref),
                }
            )
    write_csv(output_dir / "correlations.csv", ["comparison", "run_tau", "n_runs"], rows)
    print("correlations: " + ", ".join(f"{r['comparison']} run_tau={r['run_tau']}" for r in rows))
