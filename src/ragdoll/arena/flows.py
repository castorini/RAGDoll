from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any

from ragdoll.arena.metrics import leaderboard_rows, pairwise_rows, write_csv
from ragdoll.arena.prompts import parse_verdict
from ragdoll.arena.stages import coverage_rows, iter_arena_tasks, load_answer_sets, load_nuggets, load_rubrics
from ragdoll.config import ArenaCompareAllConfig, MaterializeArenaConfig
from ragdoll.jsonl import append_jsonl, read_jsonl, write_jsonl
from ragdoll.runner import run_prompt, select_rows


def _answer_paths(*, answers: list[Path], answers_dir: Path | None) -> list[Path]:
    paths = sorted(answers_dir.glob("*.jsonl")) if answers_dir is not None else list(answers)
    if len(paths) < 2:
        source = f"{answers_dir}/*.jsonl" if answers_dir is not None else "--answers"
        raise ValueError(f"arena requires at least two answer files from {source}")
    return paths


def materialize(config: MaterializeArenaConfig) -> None:
    answer_sets = load_answer_sets(_answer_paths(answers=config.answers, answers_dir=config.answers_dir))
    nuggets_by_qid = load_nuggets(config.nuggets_file) if config.nuggets_file is not None else None
    rubrics_by_qid = load_rubrics(config.rubrics_file) if config.rubrics_file is not None else None
    tasks = iter_arena_tasks(
        answer_sets,
        seed=config.seed,
        sample_topics_per_pair=config.sample_topics_per_pair,
        sample_battles_per_topic=config.sample_battles_per_topic,
        sample_battles_per_system_per_topic=config.sample_battles_per_system_per_topic,
        sampling_seed=config.sampling_seed,
        rubrics=config.rubrics,
        nuggets_by_qid=nuggets_by_qid,
        nuggets_source=str(config.nuggets_file) if config.nuggets_file is not None else None,
        rubrics_by_qid=rubrics_by_qid,
        rubrics_source=str(config.rubrics_file) if config.rubrics_file is not None else None,
        prompt_variant=config.prompt_variant,
    )
    count = write_jsonl(config.output_file, tasks)
    print(f"wrote={count} output={config.output_file}")


def _judgment_row(task: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    metadata = task["metadata"]
    verdict = parse_verdict(result["output_text"]) if result["status"] == "completed" else None
    if verdict == "A":
        preferred_run_id = metadata["assistant_a_run_id"]
    elif verdict == "B":
        preferred_run_id = metadata["assistant_b_run_id"]
    else:
        preferred_run_id = None
    status = result["status"] if verdict is not None else "failed"
    error = result["error"] if verdict is not None else result["error"] or "could not parse arena verdict"
    return {
        "task_id": task["task_id"],
        "qid": metadata["qid"],
        "query": metadata["query"],
        "pair": metadata["pair"],
        "assistant_a_run_id": metadata["assistant_a_run_id"],
        "assistant_b_run_id": metadata["assistant_b_run_id"],
        "judge_verdict": verdict,
        "preferred_run_id": preferred_run_id,
        "provider": result["provider"],
        "model": result["model"],
        "thinking": result["thinking"],
        "status": status,
        "error": error,
        "raw_output": result["output_text"],
        "elapsed_seconds": result["elapsed_seconds"],
        "usage": result["usage"],
    }


def _unlink_outputs(output_dir: Path) -> None:
    for name in ["tasks.jsonl", "judgments.jsonl", "pairwise.csv", "coverage.csv", "leaderboard.csv"]:
        (output_dir / name).unlink(missing_ok=True)


def _write_summaries(output_dir: Path, judgments_path: Path, coverage: list[dict[str, Any]], run_ids: list[str]) -> None:
    judgments = list(read_jsonl(judgments_path)) if judgments_path.exists() else []
    write_csv(
        output_dir / "coverage.csv",
        ["run_a", "run_b", "run_a_topics", "run_b_topics", "shared_topics", "run_a_only_topics", "run_b_only_topics"],
        coverage,
    )
    write_csv(
        output_dir / "pairwise.csv",
        [
            "run_a",
            "run_b",
            "shared_topics",
            "valid_judgments",
            "run_a_wins",
            "run_b_wins",
            "ties",
            "run_a_preference_rate",
            "run_b_preference_rate",
        ],
        pairwise_rows(judgments, coverage),
    )
    write_csv(
        output_dir / "leaderboard.csv",
        ["rank", "run_id", "arena_score", "n_judgments", "wins", "losses", "ties"],
        leaderboard_rows(judgments, run_ids),
    )


async def compare_all(config: ArenaCompareAllConfig) -> None:
    answer_sets = load_answer_sets(_answer_paths(answers=config.answers, answers_dir=config.answers_dir))
    nuggets_by_qid = load_nuggets(config.nuggets_file) if config.nuggets_file is not None else None
    rubrics_by_qid = load_rubrics(config.rubrics_file) if config.rubrics_file is not None else None
    tasks = iter_arena_tasks(
        answer_sets,
        seed=config.seed,
        sample_topics_per_pair=config.sample_topics_per_pair,
        sample_battles_per_topic=config.sample_battles_per_topic,
        sample_battles_per_system_per_topic=config.sample_battles_per_system_per_topic,
        sampling_seed=config.sampling_seed,
        rubrics=config.rubrics,
        nuggets_by_qid=nuggets_by_qid,
        nuggets_source=str(config.nuggets_file) if config.nuggets_file is not None else None,
        rubrics_by_qid=rubrics_by_qid,
        rubrics_source=str(config.rubrics_file) if config.rubrics_file is not None else None,
        prompt_variant=config.prompt_variant,
    )

    if config.dry_run:
        print(f"[dry-run] arena compare-all would process {len(tasks)} tasks -> {config.output_dir}")
        return

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.overwrite:
        _unlink_outputs(output_dir)

    coverage = coverage_rows(answer_sets)
    tasks_path = output_dir / "tasks.jsonl"
    judgments_path = output_dir / "judgments.jsonl"
    write_jsonl(tasks_path, tasks)

    selected = select_rows(
        tasks,
        output=judgments_path,
        resume=config.resume,
        overwrite=config.overwrite,
        shuffle=config.shuffle,
        seed=config.seed,
        limit=config.limit,
    )
    agent_config = config.local_agent_config()
    raw_events_dir = config.raw_events_dir or output_dir / "raw-events"
    semaphore = asyncio.Semaphore(max(1, config.max_concurrency))

    async def one(task: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            row_config = replace(agent_config, system_prompt=str(task.get("system_prompt", agent_config.system_prompt)))
            result = await run_prompt(
                task_id=task["task_id"],
                evaluator="arena",
                instruction=task["instruction"],
                raw_events_dir=raw_events_dir,
                config=row_config,
                metadata=task["metadata"],
            )
            return _judgment_row(task, result)

    selected_iter = iter(selected)

    async def worker() -> int:
        processed = 0
        while True:
            try:
                task = next(selected_iter)
            except StopIteration:
                return processed
            row = await one(task)
            append_jsonl(judgments_path, row)
            print(f"{row['status']} task_id={row['task_id']}", flush=True)
            processed += 1

    worker_count = min(max(1, config.max_concurrency), len(selected))
    if worker_count:
        await asyncio.gather(*(worker() for _ in range(worker_count)))

    _write_summaries(output_dir, judgments_path, coverage, [answer_set.run_id for answer_set in answer_sets])
    print(f"processed={len(selected)} output_dir={output_dir} raw_events_dir={raw_events_dir}")
