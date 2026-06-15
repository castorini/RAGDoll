"""Nuggetizer-compatible prompt materialization and Pi execution."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from pi_trec.jsonl import append_jsonl, read_jsonl, write_jsonl
from pi_trec.prompts import (
    NUGGET_AGENTIC_CREATOR_SYSTEM,
    NUGGET_AGENTIC_CREATOR_USER,
    NUGGET_ASSIGNER_2GRADE_USER,
    NUGGET_ASSIGNER_SYSTEM,
    NUGGET_ASSIGNER_USER,
    NUGGET_CREATOR_SYSTEM,
    NUGGET_CREATOR_USER,
    NUGGET_SCORER_SYSTEM,
    NUGGET_SCORER_USER,
    list_text,
    parse_label_list,
)
from pi_trec.runner import LocalAgentConfig, run_prompt, select_rows


def normalize_query(query: Any) -> tuple[str, str]:
    if isinstance(query, str):
        return "q0", query
    if isinstance(query, dict) and isinstance(query.get("text"), str):
        return str(query.get("qid", "q0")), query["text"]
    raise ValueError("Nuggetizer input requires `query` as a string or object with `text`")


def candidate_text(candidate: Any) -> str:
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, dict):
        if isinstance(candidate.get("text"), str):
            return candidate["text"]
        doc = candidate.get("doc")
        if isinstance(doc, dict) and isinstance(doc.get("segment"), str):
            return doc["segment"]
    raise ValueError("Nuggetizer candidates must be strings or objects with `text` or `doc.segment`")


def create_context(candidates: list[Any]) -> str:
    parts = [candidate_text(candidate) for candidate in candidates]
    return "\n\n".join(f"Document {index}:\n{text}" for index, text in enumerate(parts, start=1))


def render_create_prompt(
    *,
    query: str,
    context: str,
    nuggets: list[str] | None = None,
    creator_max_nuggets: int = 30,
) -> str:
    initial = nuggets or []
    user = NUGGET_CREATOR_USER.format(
        query=query,
        context=context,
        nuggets=initial,
        nuggets_length=len(initial),
        creator_max_nuggets=creator_max_nuggets,
    )
    return user


def render_agentic_create_prompt(
    *,
    query: str,
    nuggets: list[str] | None = None,
    creator_max_nuggets: int = 30,
) -> str:
    initial = nuggets or []
    return NUGGET_AGENTIC_CREATOR_USER.format(
        query=query,
        nuggets=initial,
        nuggets_length=len(initial),
        creator_max_nuggets=creator_max_nuggets,
    )


def render_score_prompt(*, query: str, nuggets: list[str]) -> str:
    return NUGGET_SCORER_USER.format(query=query, nuggets=nuggets, num_nuggets=len(nuggets))


def render_assign_prompt(
    *,
    query: str,
    context: str,
    nuggets: list[str],
    assign_mode: str,
) -> str:
    template = NUGGET_ASSIGNER_2GRADE_USER if assign_mode == "support-grade-2" else NUGGET_ASSIGNER_USER
    return template.format(query=query, context=context, nuggets=nuggets, num_nuggets=len(nuggets))


def iter_create_tasks(records: list[dict[str, Any]], *, creator_max_nuggets: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        qid, query = normalize_query(record.get("query"))
        candidates = record.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("Nuggetizer create input requires `candidates` as a list")
        context = create_context(candidates)
        task_id = str(record.get("task_id") or qid or f"record{index:06d}")
        tasks.append(
            {
                "task_id": task_id,
                "evaluator": "nugget-create",
                "system_prompt": NUGGET_CREATOR_SYSTEM,
                "instruction": render_create_prompt(
                    query=query,
                    context=context,
                    nuggets=list_text(record.get("nuggets")),
                    creator_max_nuggets=creator_max_nuggets,
                ),
                "metadata": {"qid": qid, "query": query, "context": context},
            }
        )
    return tasks


def materialize_create(args: Any) -> None:
    count = write_jsonl(
        args.output_file,
        iter_create_tasks(list(read_jsonl(args.input_file)), creator_max_nuggets=args.max_nuggets),
    )
    print(f"wrote={count} output={args.output_file}")


def iter_agentic_create_tasks(records: list[dict[str, Any]], *, creator_max_nuggets: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        qid, query = normalize_query(record.get("query"))
        initial_nuggets = list_text(record.get("nuggets"))
        task_id = str(record.get("task_id") or qid or f"record{index:06d}")
        tasks.append(
            {
                "task_id": task_id,
                "evaluator": "nugget-agentic-create",
                "system_prompt": NUGGET_AGENTIC_CREATOR_SYSTEM,
                "instruction": render_agentic_create_prompt(
                    query=query,
                    nuggets=initial_nuggets,
                    creator_max_nuggets=creator_max_nuggets,
                ),
                "metadata": {"qid": qid, "query": query, "initial_nuggets": initial_nuggets},
            }
        )
    return tasks


def materialize_agentic_create(args: Any) -> None:
    count = write_jsonl(
        args.output_file,
        iter_agentic_create_tasks(list(read_jsonl(args.input_file)), creator_max_nuggets=args.max_nuggets),
    )
    print(f"wrote={count} output={args.output_file}")


def materialize_score(args: Any) -> None:
    rows = []
    for record in read_jsonl(args.input_file):
        qid = str(record.get("qid", "q0"))
        query = str(record.get("query", ""))
        nuggets = list_text(record.get("nuggets"))
        rows.append(
            {
                "task_id": str(record.get("task_id") or qid),
                "evaluator": "nugget-score",
                "system_prompt": NUGGET_SCORER_SYSTEM,
                "instruction": render_score_prompt(query=query, nuggets=nuggets),
                "metadata": {"qid": qid, "query": query, "nuggets": nuggets},
            }
        )
    count = write_jsonl(args.output_file, rows)
    print(f"wrote={count} output={args.output_file}")


def direct_assign_inputs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if all(key in payload for key in ["query", "context", "nuggets"]):
        nuggets = normalize_nuggets(payload["nuggets"])
        return [
            {
                "task_id": str(payload.get("task_id", "direct-assign")),
                "query": str(payload["query"]),
                "context": str(payload["context"]),
                "nuggets": nuggets,
                "nugget_texts": [nugget["text"] for nugget in nuggets],
                "source": payload,
            }
        ]
    if all(key in payload for key in ["answer_record", "nugget_record"]):
        return [_answer_nugget_to_assign(payload["answer_record"], payload["nugget_record"])]
    if all(key in payload for key in ["answer_records", "nugget_record"]):
        return [_answer_nugget_to_assign(answer, payload["nugget_record"]) for answer in payload["answer_records"]]
    raise ValueError(
        "assign input requires `query`/`context`/`nuggets`, `answer_record`/`nugget_record`, "
        "or `answer_records`/`nugget_record`"
    )


def _answer_nugget_to_assign(answer_record: dict[str, Any], nugget_record: dict[str, Any]) -> dict[str, Any]:
    topic_id = str(answer_record.get("topic_id", nugget_record.get("qid", "q0")))
    query = str(answer_record.get("topic", nugget_record.get("query", "")))
    answer = answer_record.get("answer", "")
    if isinstance(answer, list):
        context = " ".join(str(sentence.get("text", sentence)) if isinstance(sentence, dict) else str(sentence) for sentence in answer)
    else:
        context = str(answer)
    nuggets = normalize_nuggets(nugget_record.get("nuggets"))
    return {
        "task_id": f"{answer_record.get('run_id', 'direct-assign')}:{topic_id}",
        "query": query,
        "context": context,
        "nuggets": nuggets,
        "nugget_texts": [nugget["text"] for nugget in nuggets],
        "source": {"answer_record": answer_record, "nugget_record": nugget_record},
    }


def normalize_nuggets(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    nuggets: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            text = str(item.get("text", ""))
            if text:
                row = {"text": text}
                if item.get("importance") is not None:
                    row["importance"] = str(item["importance"])
                nuggets.append(row)
        else:
            text = str(item)
            if text:
                nuggets.append({"text": text})
    return nuggets


def iter_assign_tasks(assign_inputs: list[dict[str, Any]], *, assign_mode: str) -> list[dict[str, Any]]:
    return [
        {
            "task_id": item["task_id"],
            "evaluator": "nugget-assign",
            "system_prompt": NUGGET_ASSIGNER_SYSTEM,
            "instruction": render_assign_prompt(
                query=item["query"],
                context=item["context"],
                nuggets=item["nugget_texts"],
                assign_mode=assign_mode,
            ),
            "metadata": item,
        }
        for item in assign_inputs
    ]


def materialize_assign(args: Any) -> None:
    inputs = _load_assign_payloads(args)
    count = write_jsonl(args.output_file, iter_assign_tasks(inputs, assign_mode=args.assign_mode))
    print(f"wrote={count} output={args.output_file}")


def _load_assign_payloads(args: Any) -> list[dict[str, Any]]:
    if args.input_json:
        return direct_assign_inputs(json.loads(args.input_json))
    payloads: list[dict[str, Any]] = []
    for row in read_jsonl(args.input_file):
        payloads.extend(direct_assign_inputs(row))
    return payloads


async def create(args: Any) -> None:
    if args.overwrite and args.output_file.exists():
        args.output_file.unlink()
    config = _config(args)
    raw_events_dir = args.raw_events_dir or args.output_file.parent / "raw-events" / args.output_file.stem
    tasks = iter_create_tasks(list(read_jsonl(args.input_file)), creator_max_nuggets=args.max_nuggets)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    for task in tasks:
        created = await run_prompt(
            task_id=f"{task['task_id']}:create",
            evaluator="nugget-create",
            instruction=task["instruction"],
            raw_events_dir=raw_events_dir,
            config=replace(config, system_prompt=task["system_prompt"]),
            metadata=task["metadata"],
        )
        nuggets = parse_label_list(created["output_text"]) or []
        score_prompt = render_score_prompt(query=task["metadata"]["query"], nuggets=nuggets)
        scored = await run_prompt(
            task_id=f"{task['task_id']}:score",
            evaluator="nugget-score",
            instruction=score_prompt,
            raw_events_dir=raw_events_dir,
            config=replace(config, system_prompt=NUGGET_SCORER_SYSTEM),
            metadata={**task["metadata"], "nuggets": nuggets},
        )
        labels = parse_label_list(scored["output_text"]) or []
        if len(labels) != len(nuggets):
            labels = ["okay"] * len(nuggets)
        row = {
            "qid": task["metadata"]["qid"],
            "query": task["metadata"]["query"],
            "nuggets": [
                {"text": nugget, "importance": label if label in {"vital", "okay"} else "okay"}
                for nugget, label in zip(nuggets, labels, strict=False)
            ],
        }
        if args.include_trace:
            row["creator_trace"] = created
            row["scorer_trace"] = scored
        append_jsonl(args.output_file, row)
        print(f"completed task_id={task['task_id']}", flush=True)
    print(f"processed={len(tasks)} output={args.output_file} raw_events_dir={raw_events_dir}")


async def agentic_create(args: Any) -> None:
    if args.overwrite:
        if args.output_file.exists():
            args.output_file.unlink()
        if args.failed_output and args.failed_output.exists():
            args.failed_output.unlink()
    tasks = select_rows(
        iter_agentic_create_tasks(list(read_jsonl(args.input_file)), creator_max_nuggets=args.max_nuggets),
        output=args.output_file,
        resume=args.resume,
        overwrite=args.overwrite,
        shuffle=args.shuffle,
        seed=args.seed,
        limit=args.limit,
    )
    config = _config(args)
    raw_events_dir = args.raw_events_dir or args.output_file.parent / "raw-events" / args.output_file.stem
    semaphore = asyncio.Semaphore(max(1, args.max_concurrency))

    async def one(task: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await _run_agentic_create_task(task, config=config, raw_events_dir=raw_events_dir, args=args)

    pending = [asyncio.create_task(one(task)) for task in tasks]
    for future in asyncio.as_completed(pending):
        row = await future
        if row["status"] == "completed":
            append_jsonl(args.output_file, row)
        elif args.failed_output:
            append_jsonl(args.failed_output, row)
        print(f"{row['status']} task_id={row['task_id']}", flush=True)
    print(f"processed={len(tasks)} output={args.output_file} raw_events_dir={raw_events_dir}")


async def _run_agentic_create_task(
    task: dict[str, Any],
    *,
    config: LocalAgentConfig,
    raw_events_dir: Path,
    args: Any,
) -> dict[str, Any]:
    metadata = task["metadata"]
    created = await run_prompt(
        task_id=f"{task['task_id']}:agentic-create",
        evaluator="nugget-agentic-create",
        instruction=task["instruction"],
        raw_events_dir=raw_events_dir,
        config=replace(config, system_prompt=task["system_prompt"]),
        metadata=metadata,
    )
    if created["status"] != "completed":
        return _agentic_failed_row(task=task, error=created["error"] or "agentic creator failed", trace=created, args=args)
    nuggets = parse_label_list(created["output_text"])
    if nuggets is None:
        return _agentic_failed_row(
            task=task,
            error="could not parse agentic creator output as a Python list",
            trace=created,
            args=args,
        )
    nuggets = nuggets[: args.max_nuggets]
    scored = await run_prompt(
        task_id=f"{task['task_id']}:score",
        evaluator="nugget-score",
        instruction=render_score_prompt(query=metadata["query"], nuggets=nuggets),
        raw_events_dir=raw_events_dir,
        config=replace(
            config,
            system_prompt=NUGGET_SCORER_SYSTEM,
            extension_path=None,
            extension_cwd=None,
            extension_env=None,
        ),
        metadata={**metadata, "nuggets": nuggets},
    )
    if scored["status"] != "completed":
        return _agentic_failed_row(task=task, error=scored["error"] or "nugget scorer failed", trace=scored, args=args)
    labels = parse_label_list(scored["output_text"]) or []
    if len(labels) != len(nuggets):
        labels = ["okay"] * len(nuggets)
    row: dict[str, Any] = {
        "task_id": task["task_id"],
        "status": "completed",
        "qid": metadata["qid"],
        "query": metadata["query"],
        "initial_nuggets": metadata["initial_nuggets"],
        "nuggets": [
            {"text": nugget, "importance": label if label in {"vital", "okay"} else "okay"}
            for nugget, label in zip(nuggets, labels, strict=False)
        ],
    }
    if args.include_trace:
        row["creator_trace"] = created
        row["scorer_trace"] = scored
    return row


def _agentic_failed_row(*, task: dict[str, Any], error: str, trace: dict[str, Any], args: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task_id": task["task_id"],
        "status": "failed",
        "evaluator": "nugget-agentic-create",
        "error": error,
        "metadata": task["metadata"],
    }
    if args.include_trace:
        row["trace"] = trace
    return row


async def assign(args: Any) -> None:
    if args.overwrite and args.output_file.exists():
        args.output_file.unlink()
    config = _config(args)
    raw_events_dir = args.raw_events_dir or args.output_file.parent / "raw-events" / args.output_file.stem
    tasks = iter_assign_tasks(_load_assign_payloads(args), assign_mode=args.assign_mode)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    for task in tasks:
        result = await run_prompt(
            task_id=task["task_id"],
            evaluator="nugget-assign",
            instruction=task["instruction"],
            raw_events_dir=raw_events_dir,
            config=replace(config, system_prompt=task["system_prompt"]),
            metadata=task["metadata"],
        )
        labels = parse_label_list(result["output_text"]) or []
        nuggets = task["metadata"]["nuggets"]
        if len(labels) != len(nuggets):
            labels = ["not_support"] * len(nuggets)
        row = {
            "query": task["metadata"]["query"],
            "context": task["metadata"]["context"],
            "nuggets": [
                {
                    **nugget,
                    "assignment": label if label in {"support", "partial_support", "not_support"} else "not_support",
                }
                for nugget, label in zip(nuggets, labels, strict=False)
            ],
        }
        if args.include_trace:
            row["trace"] = result
        append_jsonl(args.output_file, row)
        print(f"completed task_id={task['task_id']}", flush=True)
    print(f"processed={len(tasks)} output={args.output_file} raw_events_dir={raw_events_dir}")


def _config(args: Any) -> LocalAgentConfig:
    return LocalAgentConfig(
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
