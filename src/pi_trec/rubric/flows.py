from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pi_trec.config import (
    NuggetCreateConfig,
    RubricAuthorConfig,
    RubricEvalConfig,
    RubricGradeConfig,
    RubricScoreConfig,
)
from pi_trec.jsonl import append_jsonl, read_jsonl, write_jsonl
from pi_trec.nuggetizer.flows import _select_tasks, _shared, create
from pi_trec.rubric.metrics import compute_scores
from pi_trec.rubric.stages import (
    _generate_rubric,
    _grade_windowed,
    _load_grade_payloads,
    iter_author_inputs,
    iter_grade_payloads_from_files,
)


async def author(config: RubricAuthorConfig) -> None:
    if config.overwrite:
        if config.output_file.exists():
            config.output_file.unlink()
        if config.failed_output and config.failed_output.exists():
            config.failed_output.unlink()
    agent_config = config.local_agent_config()
    raw_events_dir = config.raw_events_dir or config.output_file.parent / "raw-events" / config.output_file.stem
    inputs = iter_author_inputs(list(read_jsonl(config.input_file)))
    inputs = _select_tasks(inputs, config=config, output=config.output_file)
    if config.dry_run:
        print(f"[dry-run] author would process {len(inputs)} topics -> {config.output_file}")
        return
    semaphore = asyncio.Semaphore(max(1, config.max_concurrency))

    async def one(item: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            outcome = await _generate_rubric(
                task_id=item["task_id"],
                qid=item["qid"],
                query=item["query"],
                nuggets=item["nuggets"],
                agent_config=agent_config,
                raw_events_dir=raw_events_dir,
                max_trials=config.max_trials,
            )
            status = "completed" if outcome["status"] == "parsed" else "failed"
            row: dict[str, Any] = {
                "task_id": item["task_id"],
                "status": status,
                "qid": item["qid"],
                "query": item["query"],
                "criteria": outcome["criteria"] or [],
            }
            if status == "failed":
                row["error"] = outcome["status"]
            if config.include_trace:
                row["author_trace"] = outcome["result"]
            return row

    pending = [asyncio.create_task(one(item)) for item in inputs]
    for future in asyncio.as_completed(pending):
        row = await future
        if row["status"] == "completed":
            append_jsonl(config.output_file, row)
        elif config.failed_output:
            append_jsonl(config.failed_output, row)
        print(f"{row['status']} task_id={row['task_id']}", flush=True)
    print(f"processed={len(inputs)} output={config.output_file} raw_events_dir={raw_events_dir}")


async def grade(config: RubricGradeConfig) -> None:
    if config.overwrite and config.output_file.exists():
        config.output_file.unlink()
    agent_config = config.local_agent_config()
    raw_events_dir = config.raw_events_dir or config.output_file.parent / "raw-events" / config.output_file.stem
    items = _load_grade_payloads(config)
    items = _select_tasks(items, config=config, output=config.output_file)
    if config.dry_run:
        print(f"[dry-run] grade would process {len(items)} cells -> {config.output_file}")
        return
    semaphore = asyncio.Semaphore(max(1, config.max_concurrency))

    async def one(item: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            graded, traces = await _grade_windowed(
                base_task_id=item["task_id"],
                query=item["query"],
                context=item["context"],
                criteria=item["criteria"],
                agent_config=agent_config,
                raw_events_dir=raw_events_dir,
                window_size=config.window_size,
                max_trials=config.max_trials,
            )
            row: dict[str, Any] = {
                "task_id": item["task_id"],
                "status": "completed",
                "qid": item.get("qid", ""),
                "run_id": item.get("run_id", ""),
                "query": item["query"],
                "answer_text": item.get("answer_text", item["context"]),
                "response_length": item.get("response_length"),
                "criteria": [{**criterion, "verdict": verdict} for criterion, verdict in graded],
            }
            if config.include_trace:
                row["trace"] = traces
            return row

    pending = [asyncio.create_task(one(item)) for item in items]
    for future in asyncio.as_completed(pending):
        row = await future
        append_jsonl(config.output_file, row)
        print(f"completed task_id={row['task_id']}", flush=True)
    print(f"processed={len(items)} output={config.output_file} raw_events_dir={raw_events_dir}")


async def rubric_eval_pipeline(config: RubricEvalConfig) -> None:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    shared = _shared(config)

    # 1) Rubric: use a pre-authored file, or author one from nuggets (created or fixed).
    if config.rubric_file is not None:
        rubric_path: Path = config.rubric_file
        print(f"[eval] using fixed rubric: {rubric_path}")
    else:
        if config.create_input is not None:
            nuggets_path = output_dir / "nuggets.jsonl"
            print(f"[eval] create nuggets: {config.create_input} -> {nuggets_path}")
            await create(
                NuggetCreateConfig(input_file=config.create_input, output_file=nuggets_path, max_nuggets=config.max_nuggets, **shared)
            )
        else:
            nuggets_path = config.nuggets_file
            print(f"[eval] using fixed nuggets: {nuggets_path}")
        rubric_path = output_dir / "rubric.jsonl"
        print(f"[eval] author rubric: {nuggets_path} -> {rubric_path}")
        await author(RubricAuthorConfig(input_file=nuggets_path, output_file=rubric_path, **shared))

    # 2) Join answers + rubric (by qid), then grade each cell.
    grade_inputs = output_dir / "grade_inputs.jsonl"
    written = write_jsonl(grade_inputs, iter_grade_payloads_from_files(config.answers_file, rubric_path))
    print(f"[eval] joined {written} grade payloads -> {grade_inputs}")
    graded = output_dir / "graded.jsonl"
    await grade(RubricGradeConfig(input_file=grade_inputs, output_file=graded, **shared))

    if config.dry_run:
        print("[eval] dry-run: skipping scoring")
        return

    # 3) Score (ternary + binary) and correlate against gold when provided.
    print(f"[eval] scoring {graded} -> {output_dir / 'scores'}")
    compute_scores(RubricScoreConfig(input_file=graded, output_dir=output_dir / "scores", reference=config.gold))
    print(f"[eval] done -> {output_dir}")
