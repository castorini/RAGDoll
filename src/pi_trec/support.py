"""Support-evaluation prompt materialization and Pi execution."""

from __future__ import annotations

import asyncio
from typing import Any

from pi_trec.jsonl import append_jsonl, read_jsonl, write_jsonl
from pi_trec.prompts import SUPPORT_EVAL_PROMPT, parse_support_label
from pi_trec.runner import LocalAgentConfig, run_prompt


def render_support_prompt(*, statement: str, citation: str) -> str:
    return SUPPORT_EVAL_PROMPT.format(statement=statement, citation=citation)


def iter_support_tasks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for record_index, record in enumerate(records, start=1):
        if isinstance(record.get("statement"), str) and isinstance(record.get("citation"), str):
            task_id = str(record.get("task_id") or f"support:{record_index:06d}")
            tasks.append(_task(task_id=task_id, statement=record["statement"], citation=record["citation"], source=record))
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
    topic_id = str(record.get("topic_id", f"record{record_index:06d}"))
    run_id = str(record.get("run_id", "run"))
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
            tasks.append(
                _task(
                    task_id=task_id,
                    statement=sentence["text"],
                    citation=citation_text,
                    source={
                        "topic_id": topic_id,
                        "run_id": run_id,
                        "sentence_index": sentence_index,
                        "citation_index": citation_index,
                        "docid": docid,
                    },
                )
            )
    return tasks


def _citation_docid(citation_ref: Any, references: list[Any]) -> str:
    if isinstance(citation_ref, int) and 0 <= citation_ref < len(references):
        return str(references[citation_ref])
    return str(citation_ref)


def _task(*, task_id: str, statement: str, citation: str, source: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "evaluator": "support",
        "instruction": render_support_prompt(statement=statement, citation=citation),
        "metadata": {"statement": statement, "citation": citation, "source": source},
    }


def materialize(args: Any) -> None:
    count = write_jsonl(args.output_file, iter_support_tasks(list(read_jsonl(args.input_file))))
    print(f"wrote={count} output={args.output_file}")


async def judge(args: Any) -> None:
    if args.overwrite and args.output_file.exists():
        args.output_file.unlink()
    tasks = iter_support_tasks(list(read_jsonl(args.input_file)))
    if args.limit is not None:
        tasks = tasks[: args.limit]
    config = LocalAgentConfig(
        agent_binary=args.agent_binary,
        model=args.model,
        thinking=args.thinking,
        timeout_seconds=args.timeout_seconds,
        agent_state_dir=args.agent_state_dir,
        system_prompt=args.system_prompt,
        extension_path=getattr(args, "extension_path", None),
        extension_cwd=getattr(args, "extension_cwd", None),
        extension_env=dict(getattr(args, "extension_env", []) or []),
    )
    raw_events_dir = args.raw_events_dir or args.output_file.parent / "raw-events" / args.output_file.stem
    semaphore = asyncio.Semaphore(max(1, args.max_concurrency))

    async def one(task: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            result = await run_prompt(
                task_id=task["task_id"],
                evaluator="support",
                instruction=task["instruction"],
                raw_events_dir=raw_events_dir,
                config=config,
                metadata=task["metadata"],
            )
            label = parse_support_label(result["output_text"]) if result["status"] == "completed" else None
            row = {
                "task_id": task["task_id"],
                "statement": task["metadata"]["statement"],
                "citation": task["metadata"]["citation"],
                "support_label": label,
                "raw_output": result["output_text"],
                "status": result["status"] if label is not None else "failed",
                "error": result["error"] if label is not None else result["error"] or "could not parse support label",
                "metadata": task["metadata"]["source"],
            }
            if args.include_prompt:
                row["prompt"] = task["instruction"]
            return row

    for future in asyncio.as_completed([asyncio.create_task(one(task)) for task in tasks]):
        append_jsonl(args.output_file, await future)
    print(f"processed={len(tasks)} output={args.output_file} raw_events_dir={raw_events_dir}")
