from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ragdoll.arena.prompts import TIE_VERDICTS


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: float) -> float | str:
    return "" if math.isnan(value) else round(value, 6)


def pairwise_rows(judgments: Iterable[dict[str, Any]], coverage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"a_wins": 0, "b_wins": 0, "ties": 0})
    for row in judgments:
        if row.get("status") != "completed":
            continue
        pair = row.get("pair")
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        run_a, run_b = str(pair[0]), str(pair[1])
        preferred = row.get("preferred_run_id")
        if row.get("judge_verdict") in TIE_VERDICTS:
            counts[(run_a, run_b)]["ties"] += 1
        elif preferred == run_a:
            counts[(run_a, run_b)]["a_wins"] += 1
        elif preferred == run_b:
            counts[(run_a, run_b)]["b_wins"] += 1

    rows: list[dict[str, Any]] = []
    for cov in coverage:
        run_a, run_b = str(cov["run_a"]), str(cov["run_b"])
        item = counts[(run_a, run_b)]
        valid = item["a_wins"] + item["b_wins"] + item["ties"]
        a_rate = (item["a_wins"] + 0.5 * item["ties"]) / valid if valid else math.nan
        b_rate = (item["b_wins"] + 0.5 * item["ties"]) / valid if valid else math.nan
        rows.append(
            {
                "run_a": run_a,
                "run_b": run_b,
                "shared_topics": cov["shared_topics"],
                "valid_judgments": valid,
                "run_a_wins": item["a_wins"],
                "run_b_wins": item["b_wins"],
                "ties": item["ties"],
                "run_a_preference_rate": _fmt(a_rate),
                "run_b_preference_rate": _fmt(b_rate),
            }
        )
    return rows


def system_counts(judgments: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"n_judgments": 0, "wins": 0, "losses": 0, "ties": 0})
    for row in judgments:
        if row.get("status") != "completed":
            continue
        pair = row.get("pair")
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        run_a, run_b = str(pair[0]), str(pair[1])
        counts[run_a]["n_judgments"] += 1
        counts[run_b]["n_judgments"] += 1
        preferred = row.get("preferred_run_id")
        if row.get("judge_verdict") in TIE_VERDICTS:
            counts[run_a]["ties"] += 1
            counts[run_b]["ties"] += 1
        elif preferred == run_a:
            counts[run_a]["wins"] += 1
            counts[run_b]["losses"] += 1
        elif preferred == run_b:
            counts[run_b]["wins"] += 1
            counts[run_a]["losses"] += 1
    return counts


def _arena_rank_rows(judgments: Iterable[dict[str, Any]], run_ids: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    run_id_set = set(run_ids)
    for row in judgments:
        if row.get("status") != "completed":
            continue
        pair = row.get("pair")
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        run_a, run_b = str(pair[0]), str(pair[1])
        if run_a not in run_id_set or run_b not in run_id_set:
            continue
        if row.get("judge_verdict") in TIE_VERDICTS:
            winner = "tie"
        elif row.get("preferred_run_id") == run_a:
            winner = "model_a"
        elif row.get("preferred_run_id") == run_b:
            winner = "model_b"
        else:
            continue
        rows.append({"model_a": run_a, "model_b": run_b, "winner": winner})
    return rows


def fit_arena_ratings(
    judgments: Iterable[dict[str, Any]],
    run_ids: list[str],
    *,
    scale: float = 400.0,
    init_rating: float = 1000.0,
) -> dict[str, float]:
    """Fit ratings with the lmarena/arena-rank Bradley-Terry backend."""
    if not run_ids:
        return {}

    rows = _arena_rank_rows(judgments, run_ids)
    if len(run_ids) == 1 or not rows:
        return dict(zip(run_ids, [init_rating for _ in run_ids], strict=True))

    try:
        import pandas as pd
        from arena_rank.models.bradley_terry import BradleyTerry
        from arena_rank.utils.data_utils import PairDataset
    except ImportError as exc:
        raise RuntimeError("Arena ranking requires arena-rank; install project dependencies with `uv sync`.") from exc

    dataset = PairDataset.from_pandas(pd.DataFrame(rows), reweighted=False)
    model = BradleyTerry(
        n_competitors=len(dataset.competitors),
        scale=scale,
        init_rating=init_rating,
    )
    result = model.compute_ratings_and_cis(dataset, ci_method="sandwich")
    scores = dict(
        zip(
            result["competitors"],
            [float(score) for score in result["ratings"]],
            strict=True,
        )
    )
    return {run_id: scores.get(run_id, init_rating) for run_id in run_ids}


def leaderboard_rows(judgments: Iterable[dict[str, Any]], run_ids: list[str]) -> list[dict[str, Any]]:
    judgment_list = list(judgments)
    counts = system_counts(judgment_list)
    scores = fit_arena_ratings(judgment_list, run_ids)
    ranked = sorted(run_ids, key=lambda run_id: (-scores[run_id], run_id))
    rows: list[dict[str, Any]] = []
    for rank, run_id in enumerate(ranked, start=1):
        item = counts[run_id]
        rows.append(
            {
                "rank": rank,
                "run_id": run_id,
                "arena_score": _fmt(scores[run_id]),
                "n_judgments": item["n_judgments"],
                "wins": item["wins"],
                "losses": item["losses"],
                "ties": item["ties"],
            }
        )
    return rows
