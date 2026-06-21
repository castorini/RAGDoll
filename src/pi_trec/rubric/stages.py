from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from pi_trec.config import LocalAgentConfig, RubricGradeConfig
from pi_trec.jsonl import read_jsonl
from pi_trec.nuggetizer.metrics import qid_of
from pi_trec.nuggetizer.stages import (
    _BACKOFF_BASE_SECONDS,
    _BACKOFF_MAX_SECONDS,
    _generate_labels,
    iter_window_bounds,
    normalize_nuggets,
    normalize_query,
)
from pi_trec.rubric.prompts import (
    RUBRIC_AUTHOR_SYSTEM,
    RUBRIC_GRADER_SYSTEM,
    finalize_rubric,
    normalize_criteria,
    parse_rubric,
    render_author_prompt,
    render_grade_prompt,
)
from pi_trec.runner import run_prompt
from pi_trec.trec_io import load_nuggets_by_qid


def _topic_identity(record: dict[str, Any]) -> tuple[str, str]:
    """Resolve ``(qid, query)`` from a nuggets/rubric row (qid field or query object)."""
    if record.get("qid") is not None:
        return str(record["qid"]), str(record.get("query", ""))
    return normalize_query(record.get("query"))


def iter_author_inputs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-topic authoring inputs from a nuggets JSONL (one row per topic)."""
    inputs: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        qid, query = _topic_identity(record)
        nuggets = normalize_nuggets(record.get("nuggets"))
        for position, nugget in enumerate(nuggets):
            nugget["id"] = f"n{position}"  # stable id the author references as source_nugget
        task_id = str(record.get("task_id") or qid or f"record{index:06d}")
        inputs.append({"task_id": task_id, "qid": qid, "query": query, "nuggets": nuggets})
    return inputs


async def _generate_rubric(
    *,
    task_id: str,
    qid: str,
    query: str,
    nuggets: list[dict[str, Any]],
    agent_config: LocalAgentConfig,
    raw_events_dir: Path,
    max_trials: int,
) -> dict[str, Any]:
    """Author a rubric in a single call, retrying with backoff on a parse/provider miss."""
    author_config = replace(agent_config, extension_path=None, extension_cwd=None, extension_env=None)
    last: dict[str, Any] | None = None
    for trial in range(max(1, max_trials)):
        if trial > 0:
            await asyncio.sleep(min(_BACKOFF_BASE_SECONDS * 2 ** (trial - 1), _BACKOFF_MAX_SECONDS))
        result = await run_prompt(
            task_id=task_id if trial == 0 else f"{task_id}:t{trial}",
            evaluator="rubric-author",
            instruction=render_author_prompt(query=query, nuggets=nuggets),
            raw_events_dir=raw_events_dir,
            config=replace(author_config, system_prompt=RUBRIC_AUTHOR_SYSTEM),
            metadata={"qid": qid, "query": query},
            bust_cache=trial > 0,
        )
        last = result
        if result["status"] == "completed":
            criteria = parse_rubric(result["output_text"])
            if criteria is not None:
                finalized = finalize_rubric(criteria, nuggets)
                return {"criteria": finalized, "result": result, "status": "parsed"}
    status = "provider_error" if last is not None and last["status"] != "completed" else "parse_exhausted"
    return {"criteria": None, "result": last, "status": status}


def _answer_rubric_to_grade(answer_record: dict[str, Any], rubric_record: dict[str, Any]) -> dict[str, Any]:
    topic_id = str(answer_record.get("topic_id", answer_record.get("qid", rubric_record.get("qid", "q0"))))
    query = str(answer_record.get("topic", rubric_record.get("query", "")))
    answer = answer_record.get("answer", "")
    if isinstance(answer, list):
        context = " ".join(str(s.get("text", s)) if isinstance(s, dict) else str(s) for s in answer)
    else:
        context = str(answer)
    if not context:
        context = str(answer_record.get("answer_text", ""))
    run_id = str(answer_record.get("run_id", "direct-grade"))
    return {
        "task_id": f"{run_id}:{topic_id}",
        "qid": topic_id,
        "run_id": run_id,
        "query": query,
        "context": context,
        "answer_text": str(answer_record.get("answer_text", context)),
        "response_length": answer_record.get("response_length"),
        "criteria": normalize_criteria(rubric_record.get("criteria")),
        "source": {"answer_record": answer_record, "rubric_record": rubric_record},
    }


def direct_grade_inputs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if all(key in payload for key in ["query", "answer", "criteria"]):
        answer = str(payload["answer"])
        return [
            {
                "task_id": str(payload.get("task_id", "direct-grade")),
                "qid": str(payload.get("qid", "")),
                "run_id": str(payload.get("run_id", "")),
                "query": str(payload["query"]),
                "context": answer,
                "answer_text": str(payload.get("answer_text", answer)),
                "response_length": payload.get("response_length"),
                "criteria": normalize_criteria(payload["criteria"]),
                "source": payload,
            }
        ]
    if all(key in payload for key in ["answer_record", "rubric_record"]):
        return [_answer_rubric_to_grade(payload["answer_record"], payload["rubric_record"])]
    raise ValueError("grade input requires `query`/`answer`/`criteria` or `answer_record`/`rubric_record`")


def _load_grade_payloads(config: RubricGradeConfig) -> list[dict[str, Any]]:
    if config.input_json:
        return direct_grade_inputs(json.loads(config.input_json))
    payloads: list[dict[str, Any]] = []
    for row in read_jsonl(config.input_file):
        payloads.extend(direct_grade_inputs(row))
    return payloads


def iter_grade_payloads_from_files(answers_path: Path, rubric_path: Path) -> Iterator[dict[str, Any]]:
    """Yield ``{answer_record, rubric_record}`` payloads joined by ``qid``."""
    rubric_by_qid = load_nuggets_by_qid(rubric_path)
    for answer in read_jsonl(answers_path):
        rubric = rubric_by_qid.get(qid_of(answer))
        if rubric is not None:
            yield {"answer_record": answer, "rubric_record": rubric}


async def _grade_windowed(
    *,
    base_task_id: str,
    query: str,
    context: str,
    criteria: list[dict[str, Any]],
    agent_config: LocalAgentConfig,
    raw_events_dir: Path,
    window_size: int,
    max_trials: int,
) -> tuple[list[tuple[dict[str, Any], str]], list[dict[str, Any]]]:
    """Grade criteria against the full answer in windows (full answer as context)."""
    grade_config = replace(agent_config, extension_path=None, extension_cwd=None, extension_env=None)
    graded: list[tuple[dict[str, Any], str]] = []
    traces: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(iter_window_bounds(len(criteria), window_size)):
        window = criteria[start:end]
        outcome = await _generate_labels(
            task_id=f"{base_task_id}:grade:w{index}",
            evaluator="rubric-grade",
            instruction=render_grade_prompt(query=query, answer=context, criteria=window),
            system_prompt=RUBRIC_GRADER_SYSTEM,
            raw_events_dir=raw_events_dir,
            config=grade_config,
            metadata={"query": query, "criteria": window},
            expected_length=len(window),
            max_trials=max_trials,
        )
        traces.append(outcome["result"])
        if outcome["status"] == "parsed":
            verdicts = [v.strip().lower().replace(" ", "_") for v in outcome["labels"]]
        else:
            verdicts = ["failed"] * len(window)
        graded.extend(zip(window, verdicts, strict=False))
    return graded, traces
