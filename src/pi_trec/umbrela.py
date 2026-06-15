"""UMBRELA-compatible prompt materialization and judging."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pi_trec.jsonl import append_jsonl, read_jsonl, write_jsonl
from pi_trec.prompts import parse_umbrela_judgment, render_umbrela_prompt
from pi_trec.runner import LocalAgentConfig, run_prompt


def query_text(record: dict[str, Any]) -> tuple[str, str | None]:
    query = record.get("query")
    if isinstance(query, str):
        return query, None
    if isinstance(query, dict) and isinstance(query.get("text"), str):
        qid = str(query["qid"]) if query.get("qid") is not None else None
        return query["text"], qid
    raise ValueError("UMBRELA input requires `query` as a string or object with `text`")


def candidate_passage(candidate: Any) -> tuple[str, str | None]:
    if isinstance(candidate, str):
        return candidate, None
    if isinstance(candidate, dict):
        if isinstance(candidate.get("text"), str):
            return candidate["text"], str(candidate.get("docid")) if candidate.get("docid") is not None else None
        doc = candidate.get("doc")
        if isinstance(doc, dict) and isinstance(doc.get("segment"), str):
            docid = doc.get("docid", candidate.get("docid"))
            return doc["segment"], str(docid) if docid is not None else None
    raise ValueError("UMBRELA candidates must be strings or objects with `text` or `doc.segment`")


def iter_prompt_tasks(records: list[dict[str, Any]], *, prompt_type: str) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for record_index, record in enumerate(records, start=1):
        query, qid = query_text(record)
        candidates = record.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("UMBRELA input requires `candidates` as a list")
        for candidate_index, candidate in enumerate(candidates):
            passage, docid = candidate_passage(candidate)
            task_id = str(
                record.get("task_id")
                or f"{qid or f'record{record_index:06d}'}:candidate{candidate_index:04d}"
            )
            if len(candidates) > 1 or "task_id" not in record:
                task_id = f"{task_id}:{candidate_index}"
            tasks.append(
                {
                    "task_id": task_id,
                    "evaluator": "umbrela",
                    "instruction": render_umbrela_prompt(query=query, passage=passage, prompt_type=prompt_type),
                    "metadata": {
                        "query": query,
                        "qid": qid,
                        "passage": passage,
                        "docid": docid,
                        "candidate_index": candidate_index,
                        "prompt_type": prompt_type,
                    },
                }
            )
    return tasks


def materialize(args: Any) -> None:
    tasks = iter_prompt_tasks(list(read_jsonl(args.input_file)), prompt_type=args.prompt_type)
    count = write_jsonl(args.output_file, tasks)
    print(f"wrote={count} output={args.output_file}")


async def judge(args: Any) -> None:
    if args.overwrite and args.output_file.exists():
        args.output_file.unlink()
    if args.overwrite and args.failed_output and args.failed_output.exists():
        args.failed_output.unlink()
    tasks = iter_prompt_tasks(list(read_jsonl(args.input_file)), prompt_type=args.prompt_type)
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
                evaluator="umbrela",
                instruction=task["instruction"],
                raw_events_dir=raw_events_dir,
                config=config,
                metadata=task["metadata"],
            )
            metadata = task["metadata"]
            judgment = parse_umbrela_judgment(result["output_text"]) if result["status"] == "completed" else None
            row = {
                "query": metadata["query"],
                "passage": metadata["passage"],
                "judgment": -1 if judgment is None else judgment,
            }
            if metadata.get("qid") is not None:
                row["qid"] = metadata["qid"]
            if metadata.get("docid") is not None:
                row["docid"] = metadata["docid"]
            if args.include_trace:
                row.update(
                    {
                        "task_id": task["task_id"],
                        "prompt": None if args.redact_prompts else task["instruction"],
                        "prediction": result["output_text"],
                        "result_status": result["status"],
                        "error": result["error"],
                    }
                )
            return row if result["status"] == "completed" and judgment is not None else {**row, "result_status": "failed", "error": result["error"]}

    for future in asyncio.as_completed([asyncio.create_task(one(task)) for task in tasks]):
        row = await future
        append_jsonl(args.output_file, row)
    print(f"processed={len(tasks)} output={args.output_file} raw_events_dir={raw_events_dir}")
