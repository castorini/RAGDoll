from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pi_trec.config import SupportMetricsConfig
from pi_trec.jsonl import read_jsonl, write_jsonl

WEIGHTED_SCORES = {-1: 0.0, 0: 0.0, 1: 0.5, 2: 1.0}
HARD_SCORES = {-1: 0.0, 0: 0.0, 1: 0.0, 2: 1.0}


@dataclass(frozen=True)
class SupportMetric:
    topic_id: str
    run_id: str
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

    weighted_score = 0.0
    hard_score = 0.0
    sent_with_citations = 0
    total_count_sentences = len(sentences)

    for sentence_index, sentence in enumerate(sentences):
        if not isinstance(sentence, dict):
            continue
        citations = sentence.get("citations") or []
        if not citations:
            continue
        first_citation = citations[0]
        if not isinstance(first_citation, dict):
            continue
        score = _support_score(first_citation, topic_id=topic_id, run_id=run_id, sentence_index=sentence_index)
        if score > -1:
            weighted_score += WEIGHTED_SCORES[score]
            hard_score += HARD_SCORES[score]
            sent_with_citations += 1
        elif score == -1:
            total_count_sentences -= 1

    return SupportMetric(
        topic_id=topic_id,
        run_id=run_id,
        weighted_precision=weighted_score / sent_with_citations if sent_with_citations else 0.0,
        hard_precision=hard_score / sent_with_citations if sent_with_citations else 0.0,
        weighted_recall=weighted_score / total_count_sentences if total_count_sentences else 0.0,
        hard_recall=hard_score / total_count_sentences if total_count_sentences else 0.0,
        sentences=len(sentences),
    )


def _fmt(value: float) -> float:
    return 0.0 if math.isnan(value) else round(value, 6)


def metric_row(metric: SupportMetric) -> dict[str, Any]:
    row = asdict(metric)
    for key in ("weighted_precision", "hard_precision", "weighted_recall", "hard_recall"):
        row[key] = _fmt(row[key])
    return row


def support_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [metric_row(support_metric(row)) for row in rows]


def compute_metrics(config: SupportMetricsConfig) -> None:
    rows = support_metric_rows(list(read_jsonl(config.input_file)))
    write_jsonl(config.output_file, rows)
    print(f"scored cells={len(rows)} output={config.output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute support precision/recall metrics from support judgments JSONL.")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    compute_metrics(SupportMetricsConfig(input_file=args.input_file, output_file=args.output_file))


if __name__ == "__main__":
    main()
