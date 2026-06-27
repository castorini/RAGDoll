import csv
import json
import math
from pathlib import Path

from pi_trec.config import NuggetMetricsConfig
from pi_trec.nuggetizer.metrics import (
    calculate_scores,
    cell_metric,
    compute_metrics,
    label_distribution,
    normalize_assignment,
    run_means,
)
from pi_trec.stats import kendall_tau


def test_calculate_scores_matches_reference_semantics() -> None:
    nuggets = [
        {"importance": "vital", "assignment": "support"},
        {"importance": "vital", "assignment": "partial_support"},
        {"importance": "okay", "assignment": "not_support"},
    ]
    scores = calculate_scores(nuggets)
    assert scores["strict_vital_score"] == 0.5  # 1 support / 2 vital
    assert math.isclose(scores["strict_all_score"], 1 / 3)
    assert scores["vital_score"] == 0.75  # (1 + 0.5) / 2
    assert scores["all_score"] == 0.5  # (1 + 0.5 + 0) / 3


def test_calculate_scores_empty_is_zero() -> None:
    assert calculate_scores([]) == {
        "strict_vital_score": 0.0,
        "strict_all_score": 0.0,
        "vital_score": 0.0,
        "all_score": 0.0,
    }


def test_normalize_assignment_maps_fail() -> None:
    assert normalize_assignment("Fail") == "failed"
    assert normalize_assignment(" Support ") == "support"
    assert normalize_assignment(None) == ""


def test_kendall_tau_extremes() -> None:
    assert kendall_tau([1, 2, 3], [1, 2, 3]) == 1.0
    assert kendall_tau([1, 2, 3], [3, 2, 1]) == -1.0
    assert math.isnan(kendall_tau([1.0], [1.0]))


def test_cell_metric_counts() -> None:
    row = {
        "qid": "q1",
        "run_id": "rA",
        "nuggets": [
            {"importance": "vital", "assignment": "support"},
            {"importance": "okay", "assignment": "fail"},
        ],
    }
    cell = cell_metric(row)
    assert cell.qid == "q1" and cell.run_id == "rA"
    assert cell.nugget_count == 2 and cell.vital_count == 1 and cell.failed_count == 1


def test_label_distribution_and_run_means() -> None:
    rows = [
        {"qid": "q1", "run_id": "rA", "nuggets": [{"importance": "vital", "assignment": "support"}]},
        {"qid": "q2", "run_id": "rA", "nuggets": [{"importance": "vital", "assignment": "not_support"}]},
    ]
    dist = label_distribution(rows)
    assert dist["support"] == 1 and dist["not_support"] == 1
    means = run_means([cell_metric(r) for r in rows])
    assert means["rA"]["n_topics"] == 2
    assert means["rA"]["strict_vital_score"] == 0.5


def test_compute_metrics_writes_csvs_and_correlations(tmp_path: Path) -> None:
    def cell(qid: str, run: str, support: bool) -> dict:
        label = "support" if support else "not_support"
        return {"qid": qid, "run_id": run, "nuggets": [{"importance": "vital", "assignment": label}]}

    system = tmp_path / "assign.jsonl"
    system.write_text(
        "\n".join(json.dumps(c) for c in [cell("q1", "rA", True), cell("q1", "rB", False)]) + "\n",
        encoding="utf-8",
    )
    reference = tmp_path / "gold.jsonl"
    reference.write_text(
        "\n".join(json.dumps(c) for c in [cell("q1", "rA", True), cell("q1", "rB", False)]) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "metrics"
    compute_metrics(NuggetMetricsConfig(input_file=system, output_dir=out, reference=reference))

    assert (out / "cell_metrics.csv").exists()
    assert (out / "run_metrics.csv").exists()
    assert (out / "label_distribution.csv").exists()
    corr = list(csv.DictReader((out / "correlations.csv").open(encoding="utf-8")))
    by_metric = {row["metric"]: row for row in corr}
    # System equals reference, so run-level rankings agree perfectly.
    assert by_metric["V_strict"]["run_tau"] == "1.0"
