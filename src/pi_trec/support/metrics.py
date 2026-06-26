from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pi_trec.config import SupportMetricRowsConfig, SupportMetricsConfig
from pi_trec.jsonl import read_jsonl, write_jsonl

WEIGHTED_SCORES = {-1: 0.0, 0: 0.0, 1: 0.5, 2: 1.0}
HARD_SCORES = {-1: 0.0, 0: 0.0, 1: 0.0, 2: 1.0}
SUPPORT_METRIC_COLUMNS = (
    "weighted_precision_first",
    "weighted_recall_first",
    "weighted_precision_all",
    "weighted_recall_all",
    "weighted_precision",
    "hard_precision",
    "weighted_recall",
    "hard_recall",
)
DEFAULT_METRIC_ROWS = (
    "weighted_precision_first",
    "weighted_recall_first",
    "weighted_precision_all",
    "weighted_recall_all",
)


@dataclass(frozen=True)
class SupportMetric:
    topic_id: str
    run_id: str
    weighted_precision_first: float
    weighted_recall_first: float
    weighted_precision_all: float
    weighted_recall_all: float
    weighted_precision: float
    hard_precision: float
    weighted_recall: float
    hard_recall: float
    sentences: int


def topic_id_of(row: dict[str, Any]) -> str:
    return str(row.get("topic_id") or row.get("narrative_id") or row.get("qid") or row.get("query_id") or "")


def run_id_of(row: dict[str, Any]) -> str:
    return str(row.get("run_id") or row.get("runtag") or row.get("run") or "")


def _support_score(citation: dict[str, Any], *, topic_id: str, run_id: str, sentence_index: int) -> int:
    try:
        score = int(citation.get("support"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid support score for topic={topic_id!r} run={run_id!r} sentence_index={sentence_index}"
        ) from exc
    if score not in WEIGHTED_SCORES:
        raise ValueError(
            f"unsupported support score {score!r} for topic={topic_id!r} run={run_id!r} sentence_index={sentence_index}"
        )
    return score


def support_metric(row: dict[str, Any]) -> SupportMetric:
    topic_id = topic_id_of(row)
    run_id = run_id_of(row)
    sentences = row.get("sentences") or []
    if not isinstance(sentences, list):
        raise ValueError(f"support metric input row must contain a `sentences` list: topic={topic_id!r} run={run_id!r}")

    first_weighted_score = 0.0
    first_hard_score = 0.0
    first_sent_with_citations = 0
    first_total_count_sentences = len(sentences)
    all_weighted_score = 0.0
    all_sent_with_citations = 0
    all_total_count_sentences = len(sentences)

    for sentence_index, sentence in enumerate(sentences):
        if not isinstance(sentence, dict):
            continue
        citations = sentence.get("citations") or []
        if not citations:
            continue
        citation_scores = [
            _support_score(citation, topic_id=topic_id, run_id=run_id, sentence_index=sentence_index)
            for citation in citations
            if isinstance(citation, dict)
        ]
        if not citation_scores:
            continue
        first_score = citation_scores[0]
        if first_score > -1:
            first_weighted_score += WEIGHTED_SCORES[first_score]
            first_hard_score += HARD_SCORES[first_score]
            first_sent_with_citations += 1
        elif first_score == -1:
            first_total_count_sentences -= 1
        valid_scores = [score for score in citation_scores if score > -1]
        if valid_scores:
            all_weighted_score += sum(WEIGHTED_SCORES[score] for score in valid_scores) / len(valid_scores)
            all_sent_with_citations += 1
        else:
            all_total_count_sentences -= 1

    weighted_precision_first = (
        first_weighted_score / first_sent_with_citations if first_sent_with_citations else 0.0
    )
    weighted_recall_first = first_weighted_score / first_total_count_sentences if first_total_count_sentences else 0.0
    weighted_precision_all = all_weighted_score / all_sent_with_citations if all_sent_with_citations else 0.0
    weighted_recall_all = all_weighted_score / all_total_count_sentences if all_total_count_sentences else 0.0

    return SupportMetric(
        topic_id=topic_id,
        run_id=run_id,
        weighted_precision_first=weighted_precision_first,
        weighted_recall_first=weighted_recall_first,
        weighted_precision_all=weighted_precision_all,
        weighted_recall_all=weighted_recall_all,
        weighted_precision=weighted_precision_first,
        hard_precision=first_hard_score / first_sent_with_citations if first_sent_with_citations else 0.0,
        weighted_recall=weighted_recall_first,
        hard_recall=first_hard_score / first_total_count_sentences if first_total_count_sentences else 0.0,
        sentences=len(sentences),
    )


def _fmt(value: float) -> float:
    return 0.0 if math.isnan(value) else round(value, 6)


def metric_row(metric: SupportMetric) -> dict[str, Any]:
    row = asdict(metric)
    for key in SUPPORT_METRIC_COLUMNS:
        row[key] = _fmt(row[key])
    return row


def support_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [metric_row(support_metric(row)) for row in rows]


def compute_metrics(config: SupportMetricsConfig) -> None:
    rows = support_metric_rows(list(read_jsonl(config.input_file)))
    write_jsonl(config.output_file, rows)
    print(f"scored cells={len(rows)} output={config.output_file}")


def _topic_sort_key(row: dict[str, Any]) -> tuple[int, int | str]:
    topic_id = str(row.get("topic_id", ""))
    try:
        return (0, int(topic_id))
    except ValueError:
        return (1, topic_id)


def _score_text(value: Any) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def metric_lines(rows: list[dict[str, Any]], metrics: list[str] | tuple[str, ...] = DEFAULT_METRIC_ROWS) -> list[str]:
    unknown = [metric for metric in metrics if metric not in SUPPORT_METRIC_COLUMNS]
    if unknown:
        raise ValueError(f"unknown support metric(s): {', '.join(unknown)}")

    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_run.setdefault(str(row.get("run_id", "")), []).append(row)

    lines: list[str] = []
    for run_id in sorted(by_run):
        run_rows = sorted(by_run[run_id], key=_topic_sort_key)
        for metric in metrics:
            for row in run_rows:
                lines.append(f"{run_id} {row.get('topic_id', '')} {metric} {_score_text(row.get(metric, 0.0))}")
    return lines


def write_metric_rows(config: SupportMetricRowsConfig) -> None:
    lines = metric_lines(list(read_jsonl(config.input_file)), config.metrics)
    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    config.output_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"wrote rows={len(lines)} output={config.output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute support precision/recall metrics from support judgments JSONL.")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    compute_metrics(SupportMetricsConfig(input_file=args.input_file, output_file=args.output_file))


if __name__ == "__main__":
    main()
