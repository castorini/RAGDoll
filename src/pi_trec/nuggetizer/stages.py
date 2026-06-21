from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from pi_trec.config import (
    LocalAgentConfig,
    MaterializeNuggetAgenticCreateConfig,
    MaterializeNuggetAssignConfig,
    MaterializeNuggetCreateConfig,
    MaterializeNuggetScoreConfig,
    NuggetAssignConfig,
)
from pi_trec.jsonl import read_jsonl, write_jsonl
from pi_trec.nuggetizer.prompts import (
    NUGGET_AGENTIC_CREATOR_SYSTEM,
    NUGGET_ASSIGNER_SYSTEM,
    NUGGET_CREATOR_SYSTEM,
    NUGGET_SCORER_SYSTEM,
    parse_label_list,
    render_agentic_create_prompt,
    render_assign_prompt,
    render_create_prompt,
    render_score_prompt,
)
from pi_trec.runner import run_prompt

# Exponential backoff between retries on a parse miss or provider error.
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_MAX_SECONDS = 8.0


def list_text(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in value]
    return []


def iter_window_bounds(total: int, window_size: int) -> list[tuple[int, int]]:
    """`[start, end)` windows over `total` items, at most `window_size` each."""
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
    """Per-record creation inputs for the windowed runtime (raw candidates kept)."""
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
        context = str(payload["context"])
        return [
            {
                "task_id": str(payload.get("task_id", "direct-assign")),
                "qid": str(payload.get("qid", "")),
                "run_id": str(payload.get("run_id", "")),
                "query": str(payload["query"]),
                "context": context,
                "answer_text": str(payload.get("answer_text", context)),
                "response_length": payload.get("response_length"),
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
    topic_id = str(answer_record.get("topic_id", answer_record.get("qid", nugget_record.get("qid", "q0"))))
    query = str(answer_record.get("topic", nugget_record.get("query", "")))
    answer = answer_record.get("answer", "")
    if isinstance(answer, list):
        context = " ".join(str(sentence.get("text", sentence)) if isinstance(sentence, dict) else str(sentence) for sentence in answer)
    else:
        context = str(answer)
    if not context:
        context = str(answer_record.get("answer_text", ""))
    nuggets = normalize_nuggets(nugget_record.get("nuggets"))
    run_id = str(answer_record.get("run_id", "direct-assign"))
    return {
        "task_id": f"{run_id}:{topic_id}",
        "qid": topic_id,
        "run_id": run_id,
        "query": query,
        "context": context,
        "answer_text": str(answer_record.get("answer_text", context)),
        "response_length": answer_record.get("response_length"),
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
    """Run a label-list prompt, retrying with backoff on a parse miss or provider error.

    Returns ``{"labels", "result", "status"}`` with status ``parsed``,
    ``parse_exhausted``, or ``provider_error``.
    """
    last: dict[str, Any] | None = None
    for trial in range(max(1, max_trials)):
        if trial > 0:
            await asyncio.sleep(min(_BACKOFF_BASE_SECONDS * 2 ** (trial - 1), _BACKOFF_MAX_SECONDS))
        result = await run_prompt(
            task_id=task_id if trial == 0 else f"{task_id}:t{trial}",
            evaluator=evaluator,
            instruction=instruction,
            raw_events_dir=raw_events_dir,
            config=replace(config, system_prompt=system_prompt),
            metadata=metadata,
            bust_cache=trial > 0,
        )
        last = result
        if result["status"] == "completed":
            labels = parse_label_list(result["output_text"])
            if labels is not None and (expected_length is None or len(labels) == expected_length):
                return {"labels": labels, "result": result, "status": "parsed"}
    status = "provider_error" if last is not None and last["status"] != "completed" else "parse_exhausted"
    return {"labels": None, "result": last, "status": status}


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
        else:
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
        else:
            labels = ["failed"] * len(window)
        assigned.extend(zip(window, labels, strict=False))
    return assigned, traces
