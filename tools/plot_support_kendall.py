from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

from pi_trec.nuggetizer.metrics import kendall_tau

METRICS = (
    ("weighted_precision", "weighted_precision_first_citation"),
    ("weighted_recall", "weighted_recall_first_citation"),
)


def read_metrics(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["run_id"]), str(row["topic_id"]))
            if key in rows:
                raise ValueError(f"{path}:{line_number}: duplicate run/topic row: {key}")
            rows[key] = row
    return rows


def topic_sort_key(topic_id: str) -> tuple[str, int | str]:
    prefix, _, suffix = topic_id.rpartition("-")
    if suffix:
        try:
            return (prefix, int(suffix))
        except ValueError:
            pass
    try:
        return ("", int(topic_id))
    except ValueError:
        return (topic_id, topic_id)


def mean_non_nan(values: list[float]) -> float:
    values = [value for value in values if not math.isnan(value)]
    return sum(values) / len(values) if values else math.nan


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score(row: dict[str, Any] | None, metric: str) -> float:
    return float(row.get(metric, 0.0)) if row else 0.0


def plot_support_kendall(
    *,
    human_metrics: Path,
    ragdoll_metrics: Path,
    output_dir: Path,
    run_level_denominator: int | None = None,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir.parent / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    human = read_metrics(human_metrics)
    ragdoll = read_metrics(ragdoll_metrics)
    keys = set(human) | set(ragdoll)
    runs = sorted({run_id for run_id, _ in keys})
    topics = sorted({topic_id for _, topic_id in keys}, key=topic_sort_key)
    denominator = run_level_denominator or len(topics)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: list[dict[str, Any]] = []
    topic_tau_rows: list[dict[str, Any]] = []

    for display_metric, metric in METRICS:
        paired_rows: list[dict[str, Any]] = []
        paired_x: list[float] = []
        paired_y: list[float] = []
        topic_taus: list[float] = []

        for topic_id in topics:
            topic_x: list[float] = []
            topic_y: list[float] = []
            for run_id in runs:
                key = (run_id, topic_id)
                if key not in human or key not in ragdoll:
                    continue
                x = score(ragdoll[key], metric)
                y = score(human[key], metric)
                topic_x.append(x)
                topic_y.append(y)
                paired_x.append(x)
                paired_y.append(y)
                paired_rows.append({"topic_id": topic_id, "run_id": run_id, "ragdoll": x, "nist": y})
            topic_tau = kendall_tau(topic_x, topic_y)
            topic_taus.append(topic_tau)
            topic_tau_rows.append(
                {"metric": display_metric, "topic_id": topic_id, "paired_runs": len(topic_x), "kendall_tau": topic_tau}
            )

        run_rows = []
        for run_id in runs:
            run_rows.append(
                {
                    "run_id": run_id,
                    "ragdoll": sum(score(ragdoll.get((run_id, topic_id)), metric) for topic_id in topics) / denominator,
                    "nist": sum(score(human.get((run_id, topic_id)), metric) for topic_id in topics) / denominator,
                }
            )

        run_tau = kendall_tau([row["ragdoll"] for row in run_rows], [row["nist"] for row in run_rows])
        topic_avg_tau = mean_non_nan(topic_taus)
        all_rows_tau = kendall_tau(paired_x, paired_y)

        write_csv(output_dir / f"{display_metric}_paired_topic_rows.csv", paired_rows, ["topic_id", "run_id", "ragdoll", "nist"])
        write_csv(output_dir / f"{display_metric}_run_level_divide_by_{denominator}.csv", run_rows, ["run_id", "ragdoll", "nist"])

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(paired_x, paired_y, s=42, color="#add8e6", alpha=0.58, edgecolors="none")
        ax.scatter(
            [row["ragdoll"] for row in run_rows],
            [row["nist"] for row in run_rows],
            s=54,
            color="#ff3b3b",
            alpha=0.82,
            edgecolors="white",
            linewidths=0.45,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#cfcfcf", linewidth=0.8, alpha=0.45)
        ax.set_title(f"Metric: {display_metric}", fontsize=16)
        ax.set_xlabel("RagDoll", fontsize=13)
        ax.set_ylabel("NIST", fontsize=13)
        ax.text(
            0.03,
            0.97,
            (
                f"Kendall's tau (run-level, /{denominator}): {run_tau:.3f}\n"
                f"Kendall's tau (topic-avg, missing ignored): {topic_avg_tau:.3f}\n"
                f"Kendall's tau (all paired rows): {all_rows_tau:.3f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eeeeee", "edgecolor": "#777777", "alpha": 0.95},
        )
        fig.tight_layout()
        png = output_dir / f"ragdoll_vs_nist_{display_metric}.png"
        pdf = output_dir / f"ragdoll_vs_nist_{display_metric}.pdf"
        fig.savefig(png, dpi=200)
        fig.savefig(pdf)
        plt.close(fig)

        stats.append(
            {
                "metric": display_metric,
                "runs": len(runs),
                "topics": len(topics),
                "run_level_denominator": denominator,
                "paired_topic_rows": len(paired_rows),
                "missing_cells_ignored_for_topic_tau": len(runs) * len(topics) - len(paired_rows),
                "kendall_tau_run_level_divide_by_denominator": run_tau,
                "kendall_tau_topic_avg_missing_ignored": topic_avg_tau,
                "kendall_tau_all_paired_rows": all_rows_tau,
                "plot_png": str(png),
                "plot_pdf": str(pdf),
            }
        )

    write_csv(output_dir / "kendall_stats.csv", stats, list(stats[0]))
    write_csv(output_dir / "topic_level_taus.csv", topic_tau_rows, ["metric", "topic_id", "paired_runs", "kendall_tau"])
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "rules": {
                    "run_level": f"Sum each run's topic scores and divide by {denominator}; missing topic scores contribute 0.",
                    "topic_avg_tau": "For each topic, ignore missing run-topic rows and average the topic taus.",
                    "all_rows_tau": "Compute tau over paired run-topic rows only.",
                },
                "human_metrics": str(human_metrics),
                "ragdoll_metrics": str(ragdoll_metrics),
                "topics": topics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for row in stats:
        print(
            f"{row['metric']} paired_topic_rows={row['paired_topic_rows']} "
            f"run_tau={row['kendall_tau_run_level_divide_by_denominator']:.3f} "
            f"topic_avg_tau={row['kendall_tau_topic_avg_missing_ignored']:.3f} "
            f"all_paired_tau={row['kendall_tau_all_paired_rows']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RagDoll-vs-NIST support Kendall tau comparisons.")
    parser.add_argument("--human-metrics", type=Path, required=True)
    parser.add_argument("--ragdoll-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-level-denominator", type=int)
    args = parser.parse_args()
    plot_support_kendall(
        human_metrics=args.human_metrics,
        ragdoll_metrics=args.ragdoll_metrics,
        output_dir=args.output_dir,
        run_level_denominator=args.run_level_denominator,
    )


if __name__ == "__main__":
    main()
