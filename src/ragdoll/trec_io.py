"""Adapters between TREC RAG artifacts and ragdoll nugget JSONL."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ragdoll.config import MaterializeAssignInputsConfig, MaterializeTrecAnswersConfig
from ragdoll.jsonl import read_jsonl, write_jsonl

logger = logging.getLogger("ragdoll.trec_io")


def _qid(row: dict[str, Any]) -> str:
    return str(row.get("qid") or row.get("topic_id") or row.get("query_id") or "")


def load_nuggets_by_qid(path: Path) -> dict[str, dict[str, Any]]:
    """Index a nuggets JSONL (one row per topic) by ``qid``."""
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        out[_qid(row)] = row
    return out


def iter_assign_payloads_from_files(
    answers_path: Path, nuggets_path: Path
) -> Iterator[dict[str, Any]]:
    """Yield ``{answer_record, nugget_record}`` payloads joined by ``qid``.

    Answers without a matching topic in the nuggets file are skipped (logged).
    """
    nuggets_by_qid = load_nuggets_by_qid(nuggets_path)
    missing = 0
    for answer in read_jsonl(answers_path):
        nugget = nuggets_by_qid.get(_qid(answer))
        if nugget is None:
            missing += 1
            continue
        yield {"answer_record": answer, "nugget_record": nugget}
    if missing:
        logger.warning("%d answers had no matching nuggets and were skipped", missing)


def materialize_assign_inputs(config: MaterializeAssignInputsConfig) -> None:
    """Write joined ``{answer_record, nugget_record}`` rows for batch assign."""
    nuggets_by_qid = load_nuggets_by_qid(config.nuggets_file)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for answer in read_jsonl(config.answers_file):
        qid = _qid(answer)
        nugget = nuggets_by_qid.get(qid)
        if nugget is None:
            missing.append(qid)
            continue
        rows.append({"answer_record": answer, "nugget_record": nugget})
    count = write_jsonl(config.output_file, rows)
    print(f"wrote={count} output={config.output_file}")
    if missing:
        uniq = sorted(set(missing))
        preview = ", ".join(uniq[:10]) + ("..." if len(uniq) > 10 else "")
        print(f"warning: {len(missing)} answers had no matching nuggets (qids: {preview})")


def _normalize_sentences(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, str):
        return [{"text": response, "citations": []}]
    if not isinstance(response, list):
        return []
    sentences: list[dict[str, Any]] = []
    for item in response:
        if isinstance(item, dict):
            sentences.append({"text": str(item.get("text", "")), "citations": item.get("citations", [])})
        else:
            sentences.append({"text": str(item), "citations": []})
    return sentences


def read_trec_answers(path: Path, *, run_id: str | None = None) -> Iterator[dict[str, Any]]:
    """Normalize a TREC RAG answer run into the ``answers`` schema.

    Tolerates the common fields: ``run_id``/``metadata.run_id``, ``topic_id``,
    ``topic``/``query``, and ``response``/``answer`` as a string or a list of
    ``{text, citations}`` sentences.
    """
    for row in read_jsonl(path):
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        rid = str(run_id or row.get("run_id") or metadata.get("run_id") or metadata.get("team_id") or "")
        qid = _qid(row)
        topic = str(row.get("topic") or row.get("query") or "")
        sentences = _normalize_sentences(row.get("response") if "response" in row else row.get("answer", []))
        answer_text = str(row.get("answer_text") or " ".join(s["text"] for s in sentences)).strip()
        yield {
            "run_id": rid,
            "qid": qid,
            "topic_id": qid,
            "query": topic,
            "topic": topic,
            "response_length": row.get("response_length", len(answer_text.split())),
            "answer_text": answer_text,
            "answer": sentences,
        }


def materialize_trec_answers(config: MaterializeTrecAnswersConfig) -> None:
    """Convert a TREC RAG answer run into an ``answers`` JSONL."""
    count = write_jsonl(
        config.output_file,
        read_trec_answers(config.input_file, run_id=config.run_id),
    )
    print(f"wrote={count} output={config.output_file}")
