from __future__ import annotations

from typing import Any

from pi_trec.config import MaterializeSupportConfig
from pi_trec.jsonl import read_jsonl, write_jsonl
from pi_trec.support.prompts import render_support_prompt


def iter_support_tasks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for record_index, record in enumerate(records, start=1):
        if isinstance(record.get("statement"), str) and isinstance(record.get("citation"), str):
            task_id = str(record.get("task_id") or f"support:{record_index:06d}")
            sentence_context = record.get("sentence_context")
            tasks.append(
                _task(
                    task_id=task_id,
                    statement=record["statement"],
                    citation=record["citation"],
                    sentence_context=sentence_context if isinstance(sentence_context, str) else None,
                    source=record,
                )
            )
            continue
        tasks.extend(_tasks_from_answer_row(record, record_index=record_index))
    return tasks


def _tasks_from_answer_row(record: dict[str, Any], *, record_index: int) -> list[dict[str, Any]]:
    answer = record.get("answer")
    references = record.get("references")
    segments = record.get("segments")
    if not isinstance(answer, list) or not isinstance(references, list) or not isinstance(segments, dict):
        raise ValueError(
            "support input rows must contain `statement`/`citation` strings or "
            "TREC answer rows with `answer`, `references`, and resolved `segments`"
        )
    tasks: list[dict[str, Any]] = []
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    topic_id = str(
        record.get("topic_id") or metadata.get("narrative_id") or metadata.get("topic_id") or f"record{record_index:06d}"
    )
    run_id = str(record.get("run_id") or metadata.get("run_id") or metadata.get("team_id") or "run")
    for sentence_index, sentence in enumerate(answer):
        if not isinstance(sentence, dict) or not isinstance(sentence.get("text"), str):
            continue
        citations = sentence.get("citations")
        if not isinstance(citations, list):
            continue
        for citation_index, citation_ref in enumerate(citations):
            docid = _citation_docid(citation_ref, references)
            citation_text = segments.get(docid)
            if not isinstance(citation_text, str):
                continue
            task_id = f"{run_id}:{topic_id}:s{sentence_index}:c{citation_index}"
            sentence_context = _answer_context(answer, target_index=sentence_index)
            tasks.append(
                _task(
                    task_id=task_id,
                    statement=sentence["text"],
                    citation=citation_text,
                    sentence_context=sentence_context,
                    source={
                        "topic_id": topic_id,
                        "run_id": run_id,
                        "sentence_index": sentence_index,
                        "citation_index": citation_index,
                        "docid": docid,
                        "sentence_context": sentence_context,
                    },
                )
            )
    return tasks


def _citation_docid(citation_ref: Any, references: list[Any]) -> str:
    if isinstance(citation_ref, int) and 0 <= citation_ref < len(references):
        return str(references[citation_ref])
    return str(citation_ref)


def _answer_context(answer: list[Any], *, target_index: int) -> str:
    sentences: list[str] = []
    for sentence_index, sentence in enumerate(answer):
        if not isinstance(sentence, dict) or not isinstance(sentence.get("text"), str):
            continue
        text = sentence["text"].strip()
        if not text:
            continue
        sentences.append(f"**{text}**" if sentence_index == target_index else text)
    return " ".join(sentences)


def _bold_statement(sentence_context: str, *, statement: str) -> str:
    stripped_context = sentence_context.strip()
    stripped_statement = statement.strip()
    if not stripped_context:
        return f"**{stripped_statement}**"
    if f"**{stripped_statement}**" in stripped_context:
        return stripped_context
    if stripped_statement in stripped_context:
        return stripped_context.replace(stripped_statement, f"**{stripped_statement}**", 1)
    return stripped_context


def _task(
    *,
    task_id: str,
    statement: str,
    citation: str,
    sentence_context: str | None,
    source: dict[str, Any],
) -> dict[str, Any]:
    rendered_context = _bold_statement(sentence_context or statement, statement=statement)
    return {
        "task_id": task_id,
        "evaluator": "support",
        "instruction": render_support_prompt(statement=statement, citation=citation, sentence_context=rendered_context),
        "metadata": {"statement": statement, "citation": citation, "sentence_context": rendered_context, "source": source},
    }


def materialize(config: MaterializeSupportConfig) -> None:
    count = write_jsonl(config.output_file, iter_support_tasks(list(read_jsonl(config.input_file))))
    print(f"wrote={count} output={config.output_file}")
