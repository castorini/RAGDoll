import asyncio
import json
from pathlib import Path

from pi_trec.runner import (
    LocalAgentConfig,
    RunConfig,
    build_agent_args,
    extract_assistant_text,
    run_prompt,
    run_task_rows,
    safe_task_filename,
    write_system_prompt_extension,
)


def test_build_agent_args_default_surface() -> None:
    assert build_agent_args(model="m", thinking="medium") == [
        "--no-tools",
        "--no-session",
        "--no-skills",
        "--no-extensions",
        "--no-prompt-templates",
        "--no-themes",
        "--no-context-files",
        "--system-prompt",
        "",
        "--mode",
        "json",
        "--model",
        "m",
        "--thinking",
        "medium",
    ]


def test_build_agent_args_have_no_context_or_tools() -> None:
    args = build_agent_args(model="m", thinking="medium")
    assert "--no-tools" in args
    assert "--no-session" in args
    assert "--no-skills" in args
    assert "--no-context-files" in args
    assert "--no-extensions" in args
    assert "--no-prompt-templates" in args
    assert "--no-themes" in args
    assert args[args.index("--system-prompt") + 1] == ""


def test_build_agent_args_with_extension_matches_pine_surface() -> None:
    assert build_agent_args(
        model="m",
        thinking="medium",
        extension_path="src/extensions/pi_search.ts",
        extra_extension_paths=["/tmp/system_prompt.mjs"],
    ) == [
        "--no-builtin-tools",
        "--no-session",
        "--no-skills",
        "--no-context-files",
        "-e",
        "src/extensions/pi_search.ts",
        "-e",
        "/tmp/system_prompt.mjs",
        "--mode",
        "json",
        "--model",
        "m",
        "--thinking",
        "medium",
    ]


def test_write_system_prompt_extension_replaces_prompt(tmp_path: Path) -> None:
    extension_path = write_system_prompt_extension(tmp_path, "exact prompt")

    assert extension_path.name == "pi_trec_system_prompt_override.mjs"
    text = extension_path.read_text(encoding="utf-8")
    assert "before_agent_start" in text
    assert 'systemPrompt: "exact prompt"' in text


def test_extract_assistant_text_uses_last_assistant_message() -> None:
    events = [
        {"type": "message_end", "message": {"role": "assistant", "content": "first"}},
        {"type": "message_end", "message": {"role": "user", "content": "ignored"}},
        {"type": "message_end", "message": {"role": "assistant", "content": [{"type": "text", "text": "second"}]}},
    ]
    assert extract_assistant_text(events) == "second"


def test_safe_task_filename() -> None:
    assert safe_task_filename("a/b:c") == "a_b_c"


def test_run_prompt_with_fake_agent(tmp_path: Path) -> None:
    agent = tmp_path / "fake_pi.py"
    agent.write_text(
        """#!/usr/bin/env python3
import json
import pathlib
import sys
prompt_arg = sys.argv[-1]
assert prompt_arg.startswith("@")
assert prompt_arg.endswith("prompt.txt")
assert pathlib.Path(prompt_arg[1:]).read_text(encoding="utf-8") == "prompt"
system_value = sys.argv[sys.argv.index("--system-prompt") + 1]
assert system_value == ""
print(json.dumps({"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"done"}]}}))
""",
        encoding="utf-8",
    )
    agent.chmod(0o755)
    result = asyncio.run(
        run_prompt(
            task_id="task/1",
            evaluator="test",
            instruction="prompt",
            raw_events_dir=tmp_path / "events",
            config=LocalAgentConfig(agent_binary=str(agent), model="m", thinking="medium", agent_state_dir=tmp_path / "missing"),
            metadata={"x": 1},
        )
    )
    assert result["status"] == "completed"
    assert result["output_text"] == "done"
    assert result["metadata"] == {"x": 1}
    assert json.loads((tmp_path / "events" / "task_1.jsonl").read_text())["type"] == "message_end"


def test_run_prompt_with_extension_uses_env_and_cwd(tmp_path: Path) -> None:
    extension_dir = tmp_path / "extension"
    extension_dir.mkdir()
    extension_path = extension_dir / "pi_search.ts"
    extension_path.write_text("export default function(pi) {}", encoding="utf-8")
    agent = tmp_path / "fake_pi.py"
    agent.write_text(
        """#!/usr/bin/env python3
import json
import os
import pathlib
import sys
assert "--no-builtin-tools" in sys.argv
assert "--no-tools" not in sys.argv
assert "--system-prompt" not in sys.argv
extension_indexes = [i for i, arg in enumerate(sys.argv) if arg == "-e"]
assert len(extension_indexes) == 2
assert sys.argv[extension_indexes[0] + 1].endswith("pi_search.ts")
override = pathlib.Path(sys.argv[extension_indexes[1] + 1])
assert 'systemPrompt: "agentic system"' in override.read_text(encoding="utf-8")
assert os.environ["PI_SEARCH_EXTENSION_CONFIG"] == "config-json"
assert pathlib.Path.cwd().name == "extension"
prompt_arg = sys.argv[-1]
assert pathlib.Path(prompt_arg[1:]).read_text(encoding="utf-8") == "prompt"
print(json.dumps({"type":"message_end","message":{"role":"assistant","content":"done"}}))
""",
        encoding="utf-8",
    )
    agent.chmod(0o755)

    result = asyncio.run(
        run_prompt(
            task_id="task/2",
            evaluator="test",
            instruction="prompt",
            raw_events_dir=tmp_path / "events",
            config=LocalAgentConfig(
                agent_binary=str(agent),
                model="m",
                thinking="medium",
                agent_state_dir=tmp_path / "missing",
                system_prompt="agentic system",
                extension_path=extension_path,
                extension_cwd=extension_dir,
                extension_env={"PI_SEARCH_EXTENSION_CONFIG": "config-json"},
            ),
        )
    )

    assert result["status"] == "completed"
    assert result["output_text"] == "done"


def test_run_task_rows_uses_row_system_prompt(tmp_path: Path) -> None:
    agent = tmp_path / "fake_pi.py"
    agent.write_text(
        """#!/usr/bin/env python3
import json
import pathlib
import sys
system_value = sys.argv[sys.argv.index("--system-prompt") + 1]
prompt_arg = sys.argv[-1]
assert system_value == "row system"
assert pathlib.Path(prompt_arg[1:]).read_text(encoding="utf-8") == "row prompt"
print(json.dumps({"type":"message_end","message":{"role":"assistant","content":"ok"}}))
""",
        encoding="utf-8",
    )
    agent.chmod(0o755)
    input_path = tmp_path / "tasks.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "evaluator": "test",
                "system_prompt": "row system",
                "instruction": "row prompt",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = RunConfig(
        input_file=input_path,
        output_file=output_path,
        failed_output=None,
        overwrite=True,
        resume=False,
        shuffle=False,
        seed=13,
        limit=None,
        agent_binary=str(agent),
        provider="pi",
        model="m",
        thinking="medium",
        timeout_seconds=30,
        agent_state_dir=tmp_path / "missing",
        system_prompt="fallback system",
        raw_events_dir=None,
        max_concurrency=1,
    )
    asyncio.run(run_task_rows(config))
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["status"] == "completed"
