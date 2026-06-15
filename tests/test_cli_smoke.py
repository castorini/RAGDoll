import json
import sys
from pathlib import Path

from pi_trec.cli import main


def test_materialize_umbrela_cli(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "requests.jsonl"
    output_path = tmp_path / "tasks.jsonl"
    input_path.write_text('{"query":"q","candidates":["p"]}\n', encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pi-trec",
            "materialize",
            "umbrela",
            "--input-file",
            str(input_path),
            "--output-file",
            str(output_path),
        ],
    )
    main()
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["evaluator"] == "umbrela"
    assert "Query: q" in row["instruction"]


def test_support_judge_cli_with_fake_pi(tmp_path: Path, monkeypatch) -> None:
    fake_pi = tmp_path / "fake_pi.py"
    fake_pi.write_text(
        """#!/usr/bin/env python3
import json
print(json.dumps({"type":"message_end","message":{"role":"assistant","content":"Full Support"}}))
""",
        encoding="utf-8",
    )
    fake_pi.chmod(0o755)
    input_path = tmp_path / "support.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text('{"statement":"s","citation":"c"}\n', encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pi-trec",
            "support",
            "judge",
            "--input-file",
            str(input_path),
            "--output-file",
            str(output_path),
            "--agent-binary",
            str(fake_pi),
            "--agent-state-dir",
            str(tmp_path / "missing"),
            "--overwrite",
        ],
    )
    main()
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["support_label"] == "FS"
    assert (output_path.parent / "raw-events" / output_path.stem / "support_000001.jsonl").exists()


def test_nuggetizer_agentic_create_cli_with_fake_pi(tmp_path: Path, monkeypatch) -> None:
    fake_pi = tmp_path / "fake_pi.py"
    fake_pi.write_text(
        """#!/usr/bin/env python3
import json
import pathlib
import sys
prompt = pathlib.Path(sys.argv[-1][1:]).read_text(encoding="utf-8")
if "--system-prompt" in sys.argv:
    assert "Labels:" in prompt
    content = "['vital', 'okay']"
else:
    assert "--no-builtin-tools" in sys.argv
    assert "-e" in sys.argv
    assert "Initial Nugget List: ['seed nugget']" in prompt
    content = "['seed nugget', 'retrieved nugget', 'extra nugget']"
print(json.dumps({"type":"message_end","message":{"role":"assistant","content":content}}))
""",
        encoding="utf-8",
    )
    fake_pi.chmod(0o755)
    extension_dir = tmp_path / "extension"
    extension_dir.mkdir()
    extension_path = extension_dir / "pi_search.ts"
    extension_path.write_text("export default function(pi) {}", encoding="utf-8")
    input_path = tmp_path / "agentic.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(
        '{"query":{"qid":"q1","text":"what is python used for"},"nuggets":["seed nugget"]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pi-trec",
            "nuggetizer",
            "agentic-create",
            "--input-file",
            str(input_path),
            "--output-file",
            str(output_path),
            "--agent-binary",
            str(fake_pi),
            "--agent-state-dir",
            str(tmp_path / "missing"),
            "--extension-path",
            str(extension_path),
            "--extension-cwd",
            str(extension_dir),
            "--extension-env",
            "PI_SEARCH_EXTENSION_CONFIG={}",
            "--max-nuggets",
            "2",
            "--overwrite",
        ],
    )
    main()
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["task_id"] == "q1"
    assert row["status"] == "completed"
    assert row["initial_nuggets"] == ["seed nugget"]
    assert row["nuggets"] == [
        {"text": "seed nugget", "importance": "vital"},
        {"text": "retrieved nugget", "importance": "okay"},
    ]


def test_nuggetizer_agentic_create_writes_parse_failure(tmp_path: Path, monkeypatch) -> None:
    fake_pi = tmp_path / "fake_pi.py"
    fake_pi.write_text(
        """#!/usr/bin/env python3
import json
print(json.dumps({"type":"message_end","message":{"role":"assistant","content":"not a list"}}))
""",
        encoding="utf-8",
    )
    fake_pi.chmod(0o755)
    extension_path = tmp_path / "pi_search.ts"
    extension_path.write_text("export default function(pi) {}", encoding="utf-8")
    input_path = tmp_path / "agentic.jsonl"
    output_path = tmp_path / "out.jsonl"
    failed_path = tmp_path / "failed.jsonl"
    input_path.write_text('{"query":"q"}\n', encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pi-trec",
            "nuggetizer",
            "agentic-create",
            "--input-file",
            str(input_path),
            "--output-file",
            str(output_path),
            "--failed-output",
            str(failed_path),
            "--agent-binary",
            str(fake_pi),
            "--agent-state-dir",
            str(tmp_path / "missing"),
            "--extension-path",
            str(extension_path),
            "--overwrite",
        ],
    )
    main()
    row = json.loads(failed_path.read_text(encoding="utf-8"))
    assert row["status"] == "failed"
    assert row["error"] == "could not parse agentic creator output as a Python list"
    assert not output_path.exists()
