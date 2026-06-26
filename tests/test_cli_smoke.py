import json
import sys
from pathlib import Path

from pi_trec.cli import main
from pi_trec.support import resolve as resolve_mod


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


def test_support_metrics_cli(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "support-human.jsonl"
    output_path = tmp_path / "support-metrics.jsonl"
    input_path.write_text(
        '{"narrative_id":"14","run_id":"r1","sentences":[{"citations":[{"support":"2"}]}]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pi-trec",
            "support",
            "metrics",
            "--input-file",
            str(input_path),
            "--output-file",
            str(output_path),
        ],
    )
    main()
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["topic_id"] == "14"
    assert row["run_id"] == "r1"
    assert row["weighted_precision_first_citation"] == 1.0
    assert row["weighted_recall_first_citation"] == 1.0
    assert row["weighted_precision_all_judged_citations"] == 1.0
    assert row["weighted_recall_all_judged_citations"] == 1.0


def test_support_resolve_references_cli(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "answers.jsonl"
    output_path = tmp_path / "resolved.jsonl"
    input_path.write_text(
        '{"metadata":{"run_id":"r1","narrative_id":"14"},"references":["doc-a"],"answer":[{"text":"s","citations":[0]}]}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        resolve_mod,
        "read_pyserini_document",
        lambda config, request_body: {"found": True, "docid": request_body["docid"], "text": "Resolved passage."},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pi-trec",
            "support",
            "resolve-references",
            "--input-file",
            str(input_path),
            "--output-file",
            str(output_path),
            "--pyserini-api",
            "http://pyserini",
            "--pyserini-index",
            "climbmix",
        ],
    )
    main()
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["segments"] == {"doc-a": "Resolved passage."}


def test_support_resolve_references_prebuilt_cli(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "answers.jsonl"
    output_path = tmp_path / "resolved.jsonl"
    input_path.write_text(
        '{"metadata":{"run_id":"r1","narrative_id":"14"},"references":["doc-a"],"answer":[{"text":"s","citations":[0]}]}\n',
        encoding="utf-8",
    )

    class FakeDoc:
        def raw(self) -> str:
            return json.dumps({"contents": "Prebuilt passage."})

    class FakeSearcher:
        @classmethod
        def from_prebuilt_index(cls, index: str):
            assert index == "prebuilt-name"
            return cls()

        def doc(self, docid: str):
            assert docid == "doc-a"
            return FakeDoc()

    monkeypatch.setattr(resolve_mod, "_lucene_searcher_cls", lambda: FakeSearcher)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pi-trec",
            "support",
            "resolve-references",
            "--input-file",
            str(input_path),
            "--output-file",
            str(output_path),
            "--pyserini-index",
            "prebuilt-name",
        ],
    )
    main()
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["segments"] == {"doc-a": "Prebuilt passage."}


def test_support_metric_rows_cli(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "support-metrics.jsonl"
    output_path = tmp_path / "support-metric-rows.txt"
    input_path.write_text(
        '{"topic_id":"14","run_id":"r1","weighted_precision_first_citation":0.5,"weighted_recall_first_citation":0.25,"weighted_precision_all_judged_citations":0.75,"weighted_recall_all_judged_citations":0.375,"hard_precision":0.0,"hard_recall":0.0,"sentences":2}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pi-trec",
            "support",
            "metric-rows",
            "--input-file",
            str(input_path),
            "--output-file",
            str(output_path),
        ],
    )
    main()
    assert output_path.read_text(encoding="utf-8") == (
        "r1 14 weighted_precision_first_citation 0.5\n"
        "r1 14 weighted_recall_first_citation 0.25\n"
        "r1 14 weighted_precision_all_judged_citations 0.75\n"
        "r1 14 weighted_recall_all_judged_citations 0.375\n"
    )


def test_support_assemble_cli(tmp_path: Path, monkeypatch) -> None:
    answers = tmp_path / "answers.jsonl"
    judgments = tmp_path / "judgments.parsed.jsonl"
    output = tmp_path / "support_assignments.jsonl"
    answers.write_text('{"topic_id":"14","answer":[{"text":"s","citations":["d"]}]}\n', encoding="utf-8")
    judgments.write_text(
        '{"status":"completed","support_label":"PS","metadata":{"run_id":"r1","topic_id":"14","sentence_index":0,"citation_index":0}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pi-trec",
            "support",
            "assemble",
            "--answers-file",
            str(answers),
            "--judgments",
            str(judgments),
            "--output-file",
            str(output),
            "--run-id",
            "r1",
        ],
    )
    main()
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["run_id"] == "r1"
    assert row["sentences"][0]["citations"][0]["support"] == "1"


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
