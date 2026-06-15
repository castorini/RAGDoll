"""Pi local-agent runner."""

from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from pi_trec.jsonl import append_jsonl, completed_task_ids, read_jsonl

DEFAULT_MODEL = "openai-codex/gpt-5.5"
DEFAULT_THINKING = "medium"
DEFAULT_PROVIDER = "pi"
DEFAULT_TIMEOUT_SECONDS = 900.0
DEFAULT_MAX_CONCURRENCY = 8
DEFAULT_SYSTEM_PROMPT = ""
AGENT_STATE_FILENAMES = ("auth.json", "oauth.json", "models.json")
STDERR_TAIL_MAX_CHARS = 64_000


@dataclass(frozen=True)
class LocalAgentConfig:
    agent_binary: str = "pi"
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    thinking: str = DEFAULT_THINKING
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    agent_state_dir: Path | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    extension_path: Path | None = None
    extension_cwd: Path | None = None
    extension_env: dict[str, str] | None = None


def build_agent_args(
    *,
    model: str,
    thinking: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    extension_path: str | None = None,
    extra_extension_paths: list[str] | None = None,
) -> list[str]:
    extension_paths = [path for path in [extension_path, *(extra_extension_paths or [])] if path]
    if extension_paths:
        args = [
            "--no-builtin-tools",
            "--no-session",
            "--no-skills",
            "--no-context-files",
        ]
        for path in extension_paths:
            args.extend(["-e", path])
        args.extend(
            [
                "--mode",
                "json",
                "--model",
                model,
                "--thinking",
                thinking,
            ]
        )
        return args
    return [
        "--no-tools",
        "--no-session",
        "--no-skills",
        "--no-extensions",
        "--no-prompt-templates",
        "--no-themes",
        "--no-context-files",
        "--system-prompt",
        system_prompt,
        "--mode",
        "json",
        "--model",
        model,
        "--thinking",
        thinking,
    ]


def write_system_prompt_extension(directory: Path, system_prompt: str) -> Path:
    extension_path = directory / "pi_trec_system_prompt_override.mjs"
    prompt_json = json.dumps(system_prompt, ensure_ascii=False)
    extension_path.write_text(
        "\n".join(
            [
                "export default function(pi) {",
                "  pi.on('before_agent_start', async () => ({",
                f"    systemPrompt: {prompt_json},",
                "  }));",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return extension_path


def default_agent_state_dir() -> Path:
    return Path(os.environ.get("PI_CODING_AGENT_DIR", Path.home() / ".pi" / "agent"))


def safe_task_filename(task_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in task_id)
    return safe or "missing_task_id"


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
            text = item.get("text")
            if text is not None:
                parts.append(str(text))
    return "\n".join(parts).strip()


def extract_assistant_text(events: Iterable[dict[str, Any]]) -> str:
    final_text = ""
    for event in events:
        if event.get("type") != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        text = extract_text(message.get("content"))
        if text:
            final_text = text
    return final_text


def copy_agent_state(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for filename in AGENT_STATE_FILENAMES:
        source_path = source / filename
        if source_path.is_file():
            shutil.copy2(source_path, destination / filename)


def append_stderr_tail(current: str, chunk: str) -> str:
    text = current + chunk
    if len(text) <= STDERR_TAIL_MAX_CHARS:
        return text
    return text[-STDERR_TAIL_MAX_CHARS:]


async def run_prompt(
    *,
    task_id: str,
    evaluator: str,
    instruction: str,
    raw_events_dir: Path,
    config: LocalAgentConfig,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start = time.time()
    raw_events_dir.mkdir(parents=True, exist_ok=True)
    raw_events_path = raw_events_dir / f"{safe_task_filename(task_id)}.jsonl"
    agent_state_dir = config.agent_state_dir or default_agent_state_dir()

    with tempfile.TemporaryDirectory(prefix="pi-trec-") as tmp:
        tmp_path = Path(tmp)
        prompt_path = tmp_path / "prompt.txt"
        prompt_path.write_text(instruction, encoding="utf-8")
        extra_extension_paths: list[str] = []
        if config.extension_path is not None:
            extra_extension_paths.append(str(write_system_prompt_extension(tmp_path, config.system_prompt)))
        resolved_extension_path = config.extension_path.resolve() if config.extension_path else None
        isolated_agent_dir = Path(tmp) / "agent"
        copy_agent_state(agent_state_dir, isolated_agent_dir)
        env = {
            **os.environ,
            "PI_CODING_AGENT_DIR": str(isolated_agent_dir),
            **(config.extension_env or {}),
        }
        try:
            process = await asyncio.create_subprocess_exec(
                config.agent_binary,
                *build_agent_args(
                    model=config.model,
                    thinking=config.thinking,
                    system_prompt=config.system_prompt,
                    extension_path=str(resolved_extension_path) if resolved_extension_path else None,
                    extra_extension_paths=extra_extension_paths,
                ),
                f"@{prompt_path}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=config.extension_cwd,
            )
        except OSError as exc:
            return result_row(
                task_id=task_id,
                evaluator=evaluator,
                config=config,
                output_text="",
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=time.time() - start,
                metadata=metadata,
            )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=config.timeout_seconds
            )
            timed_out = False
        except TimeoutError:
            process.kill()
            stdout_bytes, stderr_bytes = await process.communicate()
            timed_out = True

    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    stderr_tail = append_stderr_tail("", stderr_bytes.decode("utf-8", errors="replace"))
    events: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    with raw_events_path.open("w", encoding="utf-8") as out:
        for line_number, line in enumerate(stdout_text.splitlines(), start=1):
            if not line.strip():
                continue
            out.write(line + "\n")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors.append(f"line {line_number}: {exc}")
                continue
            if isinstance(value, dict):
                events.append(value)
            else:
                parse_errors.append(f"line {line_number}: expected JSON object")

    elapsed = time.time() - start
    output_text = extract_assistant_text(events)
    if timed_out:
        return result_row(
            task_id=task_id,
            evaluator=evaluator,
            config=config,
            output_text=output_text,
            status="failed",
            error=f"TimeoutError: local agent task exceeded {config.timeout_seconds:g} seconds",
            elapsed_seconds=elapsed,
            metadata=metadata,
        )
    if process.returncode != 0:
        error = f"local agent exited with code {process.returncode}"
        if stderr_tail.strip():
            error += f": {stderr_tail.strip()}"
        return result_row(
            task_id=task_id,
            evaluator=evaluator,
            config=config,
            output_text=output_text,
            status="failed",
            error=error,
            elapsed_seconds=elapsed,
            metadata=metadata,
        )
    if parse_errors:
        return result_row(
            task_id=task_id,
            evaluator=evaluator,
            config=config,
            output_text=output_text,
            status="failed",
            error="Failed to parse local agent JSON output: " + "; ".join(parse_errors[:3]),
            elapsed_seconds=elapsed,
            metadata=metadata,
        )
    if not output_text:
        return result_row(
            task_id=task_id,
            evaluator=evaluator,
            config=config,
            output_text="",
            status="failed",
            error="local agent completed without assistant text",
            elapsed_seconds=elapsed,
            metadata=metadata,
        )
    return result_row(
        task_id=task_id,
        evaluator=evaluator,
        config=config,
        output_text=output_text,
        status="completed",
        error=None,
        elapsed_seconds=elapsed,
        metadata=metadata,
    )


def result_row(
    *,
    task_id: str,
    evaluator: str,
    config: LocalAgentConfig,
    output_text: str,
    status: str,
    error: str | None,
    elapsed_seconds: float,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "evaluator": evaluator,
        "provider": config.provider,
        "model": config.model,
        "thinking": config.thinking,
        "status": status,
        "output_text": output_text,
        "parsed_output": None,
        "error": error,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "metadata": metadata or {},
    }


def select_rows(
    rows: list[dict[str, Any]],
    *,
    output: Path,
    resume: bool,
    overwrite: bool,
    shuffle: bool,
    seed: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = rows
    if resume and not overwrite:
        done = completed_task_ids(output)
        selected = [row for row in selected if str(row.get("task_id", "")) not in done]
    if shuffle:
        selected = list(selected)
        random.Random(seed).shuffle(selected)
    if limit is not None:
        selected = selected[:limit]
    return selected


async def run_task_rows(args: Any) -> None:
    if args.overwrite:
        if args.output_file.exists():
            args.output_file.unlink()
        if args.failed_output and args.failed_output.exists():
            args.failed_output.unlink()
    rows = select_rows(
        list(read_jsonl(args.input_file)),
        output=args.output_file,
        resume=args.resume,
        overwrite=args.overwrite,
        shuffle=args.shuffle,
        seed=args.seed,
        limit=args.limit,
    )
    config = LocalAgentConfig(
        agent_binary=args.agent_binary,
        provider=args.provider,
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

    async def guarded(row: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            row_config = replace(config, system_prompt=str(row.get("system_prompt", config.system_prompt)))
            return await run_prompt(
                task_id=str(row["task_id"]),
                evaluator=str(row.get("evaluator", "generic")),
                instruction=str(row["instruction"]),
                raw_events_dir=raw_events_dir,
                config=row_config,
                metadata=row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
            )

    pending = [asyncio.create_task(guarded(row)) for row in rows]
    for future in asyncio.as_completed(pending):
        result = await future
        if result["status"] == "completed":
            append_jsonl(args.output_file, result)
        elif args.failed_output:
            append_jsonl(args.failed_output, result)
        print(f"{result['status']} task_id={result['task_id']}", flush=True)
    print(f"processed={len(rows)} output={args.output_file} raw_events_dir={raw_events_dir}")
