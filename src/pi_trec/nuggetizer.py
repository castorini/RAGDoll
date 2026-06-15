"""Nuggetizer-compatible prompt materialization and Pi execution."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from pi_trec.config import (
    DEFAULT_MAX_TRIALS,
    DEFAULT_WINDOW_SIZE,
    LocalAgentConfig,
    MaterializeNuggetAgenticCreateConfig,
    MaterializeNuggetAssignConfig,
    MaterializeNuggetCreateConfig,
    MaterializeNuggetScoreConfig,
    NuggetAgenticCreateConfig,
    NuggetAssignConfig,
    NuggetCreateConfig,
)
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
from pi_trec.runner import run_prompt, select_rows

# DEFAULT_WINDOW_SIZE / DEFAULT_MAX_TRIALS now live in pi_trec.config and are
# re-exported above so existing imports (and the CLI) keep working.


def iter_window_bounds(total: int, window_size: int) -> list[tuple[int, int]]:
    """`[start, end)` windows over `total` items, at most `window_size` per window.

    Mirrors the reference `_iter_window_bounds`: the creator slides over
    documents, the scorer and assigner slide over nuggets, one window per call.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be a positive integer, got {window_size}")
    return [(start, min(start + window_size, total)) for start in range(0, total, window_size)]


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
    return "\n".join(f"[{index}] {text}" for index, text in enumerate(parts, start=1))


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


def iter_create_inputs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-record creation inputs for the windowed runtime (raw candidates kept).

    Unlike `iter_create_tasks` (which renders a single full-context prompt for
    materialization/debugging), the runtime windows over `candidates`, so it
    keeps the raw candidate list rather than a pre-rendered context.
    """
    inputs: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        qid, query = normalize_query(record.get("query"))
        candidates = record.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("Nuggetizer create input requires `candidates` as a list")
        task_id = str(record.get("task_id") or qid or f"record{index:06d}")
        inputs.append(
            {
                "task_id": task_id,
                "qid": qid,
                "query": query,
                "candidates": candidates,
                "initial_nuggets": list_text(record.get("nuggets")),
            }
        )
    return inputs


def materialize_create(config: MaterializeNuggetCreateConfig) -> None:
    count = write_jsonl(
        config.output_file,
        iter_create_tasks(list(read_jsonl(config.input_file)), creator_max_nuggets=config.max_nuggets),
    )
    print(f"wrote={count} output={config.output_file}")


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


def materialize_agentic_create(config: MaterializeNuggetAgenticCreateConfig) -> None:
    count = write_jsonl(
        config.output_file,
        iter_agentic_create_tasks(list(read_jsonl(config.input_file)), creator_max_nuggets=config.max_nuggets),
    )
    print(f"wrote={count} output={config.output_file}")


def materialize_score(config: MaterializeNuggetScoreConfig) -> None:
    rows = []
    for record in read_jsonl(config.input_file):
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
    count = write_jsonl(config.output_file, rows)
    print(f"wrote={count} output={config.output_file}")


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


def materialize_assign(config: MaterializeNuggetAssignConfig) -> None:
    inputs = _load_assign_payloads(config)
    count = write_jsonl(config.output_file, iter_assign_tasks(inputs, assign_mode=config.assign_mode))
    print(f"wrote={count} output={config.output_file}")


def _load_assign_payloads(config: NuggetAssignConfig | MaterializeNuggetAssignConfig) -> list[dict[str, Any]]:
    if config.input_json:
        return direct_assign_inputs(json.loads(config.input_json))
    payloads: list[dict[str, Any]] = []
    for row in read_jsonl(config.input_file):
        payloads.extend(direct_assign_inputs(row))
    return payloads


async def _generate_labels(
    *,
    task_id: str,
    evaluator: str,
    instruction: str,
    system_prompt: str,
    raw_events_dir: Path,
    config: LocalAgentConfig,
    metadata: dict[str, Any],
    expected_length: int | None,
    max_trials: int,
) -> dict[str, Any]:
    """Run a label-list prompt, retrying on a parse miss.

    Mirrors the reference temperature-bump retry loop. Through the Pi CLI a retry
    is a plain re-run (the default reasoning model ignores temperature, so no
    temperature is plumbed). Returns ``{"labels", "result", "status"}`` where
    status is one of:

    - ``"parsed"``: a list was parsed (matching ``expected_length`` when given);
    - ``"parse_exhausted"``: every trial completed but none parsed;
    - ``"provider_error"``: a trial failed to complete (no parse-retry, matching
      the reference, which lets provider errors short-circuit the parse loop).
    """
    last: dict[str, Any] | None = None
    for trial in range(max(1, max_trials)):
        result = await run_prompt(
            task_id=task_id if trial == 0 else f"{task_id}:t{trial}",
            evaluator=evaluator,
            instruction=instruction,
            raw_events_dir=raw_events_dir,
            config=replace(config, system_prompt=system_prompt),
            metadata=metadata,
        )
        last = result
        if result["status"] != "completed":
            return {"labels": None, "result": result, "status": "provider_error"}
        labels = parse_label_list(result["output_text"])
        if labels is not None and (expected_length is None or len(labels) == expected_length):
            return {"labels": labels, "result": result, "status": "parsed"}
    return {"labels": None, "result": last, "status": "parse_exhausted"}


async def _create_windowed(
    *,
    base_task_id: str,
    qid: str,
    query: str,
    candidates: list[Any],
    initial_nuggets: list[str],
    agent_config: LocalAgentConfig,
    raw_events_dir: Path,
    window_size: int,
    max_trials: int,
    max_nuggets: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Fold document windows into a running nugget list (reference creator loop)."""
    nuggets = list(initial_nuggets)
    traces: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(iter_window_bounds(len(candidates), window_size)):
        window = candidates[start:end]
        outcome = await _generate_labels(
            task_id=f"{base_task_id}:create:w{index}",
            evaluator="nugget-create",
            instruction=render_create_prompt(
                query=query,
                context=create_context(window),
                nuggets=nuggets,
                creator_max_nuggets=max_nuggets,
            ),
            system_prompt=NUGGET_CREATOR_SYSTEM,
            raw_events_dir=raw_events_dir,
            config=agent_config,
            metadata={"qid": qid, "query": query, "window": [start, end]},
            expected_length=None,
            max_trials=max_trials,
        )
        traces.append(outcome["result"])
        if outcome["status"] == "parsed":
            nuggets = outcome["labels"][:max_nuggets]
        # parse_exhausted / provider_error: keep the nuggets gathered so far.
    return nuggets, traces


async def _score_windowed(
    *,
    base_task_id: str,
    qid: str,
    query: str,
    nuggets: list[str],
    agent_config: LocalAgentConfig,
    raw_events_dir: Path,
    window_size: int,
    max_trials: int,
) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    """Label nugget importance in windows (reference scorer loop)."""
    score_config = replace(agent_config, extension_path=None, extension_cwd=None, extension_env=None)
    scored: list[tuple[str, str]] = []
    traces: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(iter_window_bounds(len(nuggets), window_size)):
        window = nuggets[start:end]
        outcome = await _generate_labels(
            task_id=f"{base_task_id}:score:w{index}",
            evaluator="nugget-score",
            instruction=render_score_prompt(query=query, nuggets=window),
            system_prompt=NUGGET_SCORER_SYSTEM,
            raw_events_dir=raw_events_dir,
            config=score_config,
            metadata={"qid": qid, "query": query, "nuggets": window},
            expected_length=len(window),
            max_trials=max_trials,
        )
        traces.append(outcome["result"])
        if outcome["status"] == "parsed":
            labels = [label.strip().lower() for label in outcome["labels"]]
        elif outcome["status"] == "parse_exhausted":
            labels = ["okay"] * len(window)
        else:  # provider_error: drop this window (reference scorer returns []).
            continue
        scored.extend(zip(window, labels, strict=False))
    return scored, traces


async def _assign_windowed(
    *,
    base_task_id: str,
    query: str,
    context: str,
    nuggets: list[dict[str, Any]],
    assign_mode: str,
    agent_config: LocalAgentConfig,
    raw_events_dir: Path,
    window_size: int,
    max_trials: int,
) -> tuple[list[tuple[dict[str, Any], str]], list[dict[str, Any]]]:
    """Label nugget support against a passage in windows (reference assigner loop)."""
    assign_config = replace(agent_config, extension_path=None, extension_cwd=None, extension_env=None)
    assigned: list[tuple[dict[str, Any], str]] = []
    traces: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(iter_window_bounds(len(nuggets), window_size)):
        window = nuggets[start:end]
        outcome = await _generate_labels(
            task_id=f"{base_task_id}:assign:w{index}",
            evaluator="nugget-assign",
            instruction=render_assign_prompt(
                query=query,
                context=context,
                nuggets=[nugget["text"] for nugget in window],
                assign_mode=assign_mode,
            ),
            system_prompt=NUGGET_ASSIGNER_SYSTEM,
            raw_events_dir=raw_events_dir,
            config=assign_config,
            metadata={"query": query, "context": context, "nuggets": window},
            expected_length=len(window),
            max_trials=max_trials,
        )
        traces.append(outcome["result"])
        if outcome["status"] == "parsed":
            labels = [label.strip().lower() for label in outcome["labels"]]
        else:  # parse_exhausted / provider_error: mark the window failed.
            labels = ["failed"] * len(window)
        assigned.extend(zip(window, labels, strict=False))
    return assigned, traces


async def create(config: NuggetCreateConfig) -> None:
    if config.overwrite and config.output_file.exists():
        config.output_file.unlink()
    agent_config = config.local_agent_config()
    raw_events_dir = config.raw_events_dir or config.output_file.parent / "raw-events" / config.output_file.stem
    inputs = iter_create_inputs(list(read_jsonl(config.input_file)))
    if config.limit is not None:
        inputs = inputs[: config.limit]
    for item in inputs:
        nuggets, creator_traces = await _create_windowed(
            base_task_id=item["task_id"],
            qid=item["qid"],
            query=item["query"],
            candidates=item["candidates"],
            initial_nuggets=item["initial_nuggets"],
            agent_config=agent_config,
            raw_events_dir=raw_events_dir,
            window_size=config.window_size,
            max_trials=config.max_trials,
            max_nuggets=config.max_nuggets,
        )
        scored, scorer_traces = await _score_windowed(
            base_task_id=item["task_id"],
            qid=item["qid"],
            query=item["query"],
            nuggets=nuggets,
            agent_config=agent_config,
            raw_events_dir=raw_events_dir,
            window_size=config.window_size,
            max_trials=config.max_trials,
        )
        row: dict[str, Any] = {
            "qid": item["qid"],
            "query": item["query"],
            "nuggets": [{"text": text, "importance": importance} for text, importance in scored],
        }
        if config.include_trace:
            row["creator_trace"] = creator_traces
            row["scorer_trace"] = scorer_traces
        append_jsonl(config.output_file, row)
        print(f"completed task_id={item['task_id']}", flush=True)
    print(f"processed={len(inputs)} output={config.output_file} raw_events_dir={raw_events_dir}")


async def agentic_create(config: NuggetAgenticCreateConfig) -> None:
    if config.overwrite:
        if config.output_file.exists():
            config.output_file.unlink()
        if config.failed_output and config.failed_output.exists():
            config.failed_output.unlink()
    tasks = select_rows(
        iter_agentic_create_tasks(list(read_jsonl(config.input_file)), creator_max_nuggets=config.max_nuggets),
        output=config.output_file,
        resume=config.resume,
        overwrite=config.overwrite,
        shuffle=config.shuffle,
        seed=config.seed,
        limit=config.limit,
    )
    agent_config = config.local_agent_config()
    raw_events_dir = config.raw_events_dir or config.output_file.parent / "raw-events" / config.output_file.stem
    semaphore = asyncio.Semaphore(max(1, config.max_concurrency))

    async def one(task: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await _run_agentic_create_task(
                task,
                agent_config=agent_config,
                raw_events_dir=raw_events_dir,
                max_nuggets=config.max_nuggets,
                window_size=config.window_size,
                max_trials=config.max_trials,
                include_trace=config.include_trace,
            )

    pending = [asyncio.create_task(one(task)) for task in tasks]
    for future in asyncio.as_completed(pending):
        row = await future
        if row["status"] == "completed":
            append_jsonl(config.output_file, row)
        elif config.failed_output:
            append_jsonl(config.failed_output, row)
        print(f"{row['status']} task_id={row['task_id']}", flush=True)
    print(f"processed={len(tasks)} output={config.output_file} raw_events_dir={raw_events_dir}")


async def _run_agentic_create_task(
    task: dict[str, Any],
    *,
    agent_config: LocalAgentConfig,
    raw_events_dir: Path,
    max_nuggets: int,
    window_size: int,
    max_trials: int,
    include_trace: bool,
) -> dict[str, Any]:
    metadata = task["metadata"]
    created = await run_prompt(
        task_id=f"{task['task_id']}:agentic-create",
        evaluator="nugget-agentic-create",
        instruction=task["instruction"],
        raw_events_dir=raw_events_dir,
        config=replace(agent_config, system_prompt=task["system_prompt"]),
        metadata=metadata,
    )
    if created["status"] != "completed":
        return _agentic_failed_row(
            task=task, error=created["error"] or "agentic creator failed", trace=created, include_trace=include_trace
        )
    nuggets = parse_label_list(created["output_text"])
    if nuggets is None:
        return _agentic_failed_row(
            task=task,
            error="could not parse agentic creator output as a Python list",
            trace=created,
            include_trace=include_trace,
        )
    nuggets = nuggets[:max_nuggets]
    # The agentic creator is a single tool-using session (it does its own
    # retrieval), but scoring its final nugget list is windowed, matching the
    # reference scorer.
    scored, scorer_traces = await _score_windowed(
        base_task_id=task["task_id"],
        qid=metadata["qid"],
        query=metadata["query"],
        nuggets=nuggets,
        agent_config=agent_config,
        raw_events_dir=raw_events_dir,
        window_size=window_size,
        max_trials=max_trials,
    )
    row: dict[str, Any] = {
        "task_id": task["task_id"],
        "status": "completed",
        "qid": metadata["qid"],
        "query": metadata["query"],
        "initial_nuggets": metadata["initial_nuggets"],
        "nuggets": [{"text": text, "importance": importance} for text, importance in scored],
    }
    if include_trace:
        row["creator_trace"] = created
        row["scorer_trace"] = scorer_traces
    return row


def _agentic_failed_row(
    *, task: dict[str, Any], error: str, trace: dict[str, Any], include_trace: bool
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task_id": task["task_id"],
        "status": "failed",
        "evaluator": "nugget-agentic-create",
        "error": error,
        "metadata": task["metadata"],
    }
    if include_trace:
        row["trace"] = trace
    return row


async def assign(config: NuggetAssignConfig) -> None:
    if config.overwrite and config.output_file.exists():
        config.output_file.unlink()
    agent_config = config.local_agent_config()
    raw_events_dir = config.raw_events_dir or config.output_file.parent / "raw-events" / config.output_file.stem
    items = _load_assign_payloads(config)
    if config.limit is not None:
        items = items[: config.limit]
    for item in items:
        assigned, traces = await _assign_windowed(
            base_task_id=item["task_id"],
            query=item["query"],
            context=item["context"],
            nuggets=item["nuggets"],
            assign_mode=config.assign_mode,
            agent_config=agent_config,
            raw_events_dir=raw_events_dir,
            window_size=config.window_size,
            max_trials=config.max_trials,
        )
        row = {
            "query": item["query"],
            "context": item["context"],
            "nuggets": [{**nugget, "assignment": assignment} for nugget, assignment in assigned],
        }
        if config.include_trace:
            row["trace"] = traces
        append_jsonl(config.output_file, row)
        print(f"completed task_id={item['task_id']}", flush=True)
    print(f"processed={len(items)} output={config.output_file} raw_events_dir={raw_events_dir}")
