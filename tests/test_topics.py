import json
import sys
from pathlib import Path

import pytest

from pi_trec import topics
from pi_trec.cli import main


def valid_prompt() -> str:
    return (
        "I am preparing a practical research memo for city transportation staff about how transit agencies "
        "can improve late-night bus reliability while balancing driver staffing, passenger safety, budget "
        "limits, and communication with riders. Please compare several strategies, explain tradeoffs, and "
        "recommend an implementation plan for a mid-sized North American city."
    )


def valid_answer() -> str:
    sentences = []
    for index in range(1, 9):
        sentences.append(
            f"Evidence point {index} explains a distinct operational issue for the planning memo [shard_{index:05d}_{index:05d}]."
        )
    return " ".join(sentences)


def test_render_synthesis_prompt_matches_karl_surface() -> None:
    prompt = topics.render_synthesis_prompt(
        max_search_calls=50,
        candidates_per_episode=1,
        search_topk=20,
        min_unique_cited_docids=8,
        min_search_calls_per_candidate=8,
        informal_user_prompt_style=True,
        target_topic_category="Urban transportation",
        icl_examples=(topics.IclExample(prompt="example prompt"),),
    )
    assert "You may make up to 50 search calls" in prompt
    assert "Create exactly 1 new benchmark examples" in prompt
    assert "at least 8 distinct search calls" in prompt
    assert "Synthesize evidence from at least 8 distinct source documents" in prompt
    assert "Urban transportation" in prompt
    assert "informal, personal" in prompt
    assert '"prompt": "example prompt"' in prompt


def test_iter_tasks_defaults_and_deterministic_sampling() -> None:
    pool = (
        topics.IclExample(prompt="a", sample_id="1", domain="d1"),
        topics.IclExample(prompt="b", sample_id="2", domain="d2"),
    )
    first = topics.iter_tasks(
        episodes=2,
        candidates_per_episode=topics.DEFAULT_CANDIDATES_PER_EPISODE,
        max_search_calls=topics.DEFAULT_MAX_SEARCH_CALLS,
        search_topk=topics.DEFAULT_SEARCH_TOPK,
        min_unique_cited_docids=topics.DEFAULT_MIN_UNIQUE_CITED_DOCIDS,
        min_search_calls_per_candidate=topics.DEFAULT_MIN_SEARCH_CALLS_PER_CANDIDATE,
        informal_style_probability=topics.DEFAULT_INFORMAL_STYLE_PROBABILITY,
        topic_categories=("History", "Science"),
        topic_category_seed=topics.DEFAULT_TOPIC_CATEGORY_SEED,
        icl_pool=pool,
        icl_source="researchrubrics",
        icl_examples_per_episode=topics.DEFAULT_ICL_EXAMPLES,
        icl_seed=topics.DEFAULT_ICL_SEED,
    )
    second = topics.iter_tasks(
        episodes=2,
        candidates_per_episode=topics.DEFAULT_CANDIDATES_PER_EPISODE,
        max_search_calls=topics.DEFAULT_MAX_SEARCH_CALLS,
        search_topk=topics.DEFAULT_SEARCH_TOPK,
        min_unique_cited_docids=topics.DEFAULT_MIN_UNIQUE_CITED_DOCIDS,
        min_search_calls_per_candidate=topics.DEFAULT_MIN_SEARCH_CALLS_PER_CANDIDATE,
        informal_style_probability=topics.DEFAULT_INFORMAL_STYLE_PROBABILITY,
        topic_categories=("History", "Science"),
        topic_category_seed=topics.DEFAULT_TOPIC_CATEGORY_SEED,
        icl_pool=pool,
        icl_source="researchrubrics",
        icl_examples_per_episode=topics.DEFAULT_ICL_EXAMPLES,
        icl_seed=topics.DEFAULT_ICL_SEED,
    )
    assert first == second
    assert first[0]["task_id"] == "karl_synth_episode_000001"
    assert first[0]["input_text"] == ""
    metadata = first[0]["metadata"]
    assert metadata["pipeline"] == "karl_style_qa_synthesis"
    assert metadata["max_search_calls"] == 50
    assert metadata["search_topk"] == 20
    assert metadata["candidates_requested"] == 1
    assert metadata["min_unique_cited_docids"] == 8
    assert metadata["min_search_calls_per_candidate"] == 8
    assert metadata["icl_seed"] == 13
    assert len(metadata["icl_examples"]) == 4
    assert metadata["target_topic_category"] in {"History", "Science"}


def test_iter_tasks_rejects_impossible_search_budget() -> None:
    with pytest.raises(ValueError, match="requires 12 searches"):
        topics.iter_tasks(
            episodes=1,
            candidates_per_episode=3,
            max_search_calls=10,
            search_topk=20,
            min_unique_cited_docids=8,
            min_search_calls_per_candidate=4,
            informal_style_probability=0,
            topic_categories=None,
            topic_category_seed=13,
            icl_pool=topics.ICL_EXAMPLES,
            icl_source="fixed",
            icl_examples_per_episode=1,
            icl_seed=13,
        )


def test_candidate_parsing_validation_and_reporting() -> None:
    output = json.dumps([{"prompt": valid_prompt(), "reference_answer": valid_answer()}])
    accepted, rejected, summary = topics.process_results(
        [{"task_id": "karl_synth_episode_000001", "status": "completed", "output_text": f"```json\n{output}\n```"}],
        candidates_per_episode=1,
        existing_prompts=[],
    )
    assert rejected == []
    assert accepted[0]["candidate_id"] == "karl_synth_episode_000001_00"
    assert summary["pine_results"] == 1
    assert summary["accepted_candidates"] == 1
    report = topics.render_report(accepted, summary)
    assert "# KARL Synthesis Candidate Report" in report
    assert "Accepted candidates: 1" in report
    assert "Unique cited docids per answer:" in report


def test_candidate_rejections_cover_parse_policy_citations_and_duplicates() -> None:
    bad_prompt = "short"
    few_citations = "Only one source supports this answer [shard_00001_00001]."
    rows = [
        {"task_id": "parse", "status": "completed", "output_text": "not json"},
        {
            "task_id": "policy",
            "status": "completed",
            "output_text": json.dumps([{"prompt": bad_prompt, "reference_answer": valid_answer()}]),
        },
        {
            "task_id": "citations",
            "status": "completed",
            "output_text": json.dumps([{"prompt": valid_prompt(), "reference_answer": few_citations}]),
        },
        {
            "task_id": "dup",
            "status": "completed",
            "output_text": json.dumps([{"prompt": valid_prompt(), "reference_answer": valid_answer()}]),
        },
    ]
    accepted, rejected, summary = topics.process_results(rows, candidates_per_episode=1, existing_prompts=[valid_prompt()])
    assert accepted == []
    assert [row["reason"] for row in rejected] == [
        "parse_error",
        "candidate_validation_error",
        "candidate_validation_error",
        "near_duplicate",
    ]
    assert summary["rejection_reasons"] == {
        "candidate_validation_error": 2,
        "near_duplicate": 1,
        "parse_error": 1,
    }


def test_category_task_and_parser() -> None:
    prompt = topics.render_category_prompt(
        examples=[{"domain": "Science", "prompt": "Explain climate adaptation planning."}],
        category_count=2,
    )
    assert "Create exactly 2 unique top-level topic categories" in prompt
    assert "Domain: Science" in prompt
    assert topics.parse_categories_text("1. Public health\n- public health\n2) Software systems\n") == [
        "Public health",
        "Software systems",
    ]


def test_parse_categories_requires_exact_count(tmp_path: Path) -> None:
    input_path = tmp_path / "results.jsonl"
    input_path.write_text(
        json.dumps({"task_id": "rr", "status": "completed", "output_text": "One\nTwo"}) + "\n",
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {
            "input_file": input_path,
            "output_file": tmp_path / "categories.txt",
            "summary_output": tmp_path / "summary.json",
            "category_count": 3,
        },
    )()
    with pytest.raises(SystemExit, match="expected 3 unique categories"):
        topics.parse_categories(args)


def test_topics_materialize_cli(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "tasks.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pi-trec",
            "topics",
            "materialize",
            "--episodes",
            "1",
            "--icl-source",
            "fixed",
            "--output-file",
            str(output_path),
        ],
    )
    main()
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["task_id"] == "karl_synth_episode_000001"
    assert row["metadata"]["pipeline"] == "karl_style_qa_synthesis"


def test_topics_generate_cli_with_fake_pi_uses_pine_search_prompt(tmp_path: Path, monkeypatch) -> None:
    fake_pi = tmp_path / "fake_pi.py"
    fake_pi.write_text(
        f"""#!/usr/bin/env python3
import json
import os
import pathlib
import sys
assert "--no-builtin-tools" in sys.argv
assert "--no-tools" not in sys.argv
assert os.environ["PI_SEARCH_EXTENSION_CONFIG"] == "{{}}"
extension_indexes = [i for i, arg in enumerate(sys.argv) if arg == "-e"]
assert len(extension_indexes) == 2
override = pathlib.Path(sys.argv[extension_indexes[1] + 1]).read_text(encoding="utf-8")
assert "retrieval agent operating inside Pi" in override
assert "read_document" in override
prompt_arg = sys.argv[-1]
assert prompt_arg.startswith("@")
assert prompt_arg.endswith("prompt.txt")
assert "Create exactly 1 new benchmark examples" in pathlib.Path(prompt_arg[1:]).read_text(encoding="utf-8")
print(json.dumps({{"type":"message_end","message":{{"role":"assistant","content":{json.dumps(json.dumps([{"prompt": valid_prompt(), "reference_answer": valid_answer()}]))}}}}}))
""",
        encoding="utf-8",
    )
    fake_pi.chmod(0o755)
    extension_dir = tmp_path / "extension"
    extension_dir.mkdir()
    extension_path = extension_dir / "pi_search.ts"
    extension_path.write_text("export default function(pi) {}", encoding="utf-8")
    input_path = tmp_path / "tasks.jsonl"
    output_path = tmp_path / "results.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "task_id": "karl_synth_episode_000001",
                "instruction": topics.render_synthesis_prompt(max_search_calls=50, candidates_per_episode=1, search_topk=20),
                "metadata": {"pipeline": "karl_style_qa_synthesis"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pi-trec",
            "topics",
            "generate",
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
            "--overwrite",
        ],
    )
    main()
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["evaluator"] == "topics-generate"
    assert row["status"] == "completed"
    assert row["metadata"]["pipeline"] == "karl_style_qa_synthesis"


def test_topics_generate_cli_writes_failed_output_and_resume_skips(tmp_path: Path, monkeypatch) -> None:
    fake_pi = tmp_path / "fake_pi.py"
    fake_pi.write_text(
        """#!/usr/bin/env python3
import sys
sys.exit(2)
""",
        encoding="utf-8",
    )
    fake_pi.chmod(0o755)
    input_path = tmp_path / "tasks.jsonl"
    output_path = tmp_path / "results.jsonl"
    failed_path = tmp_path / "failed.jsonl"
    input_path.write_text(
        '{"task_id":"t1","instruction":"prompt","metadata":{}}\n{"task_id":"t2","instruction":"prompt","metadata":{}}\n',
        encoding="utf-8",
    )
    output_path.write_text('{"task_id":"t1","status":"completed"}\n', encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pi-trec",
            "topics",
            "generate",
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
            "--resume",
        ],
    )
    main()
    failed = json.loads(failed_path.read_text(encoding="utf-8"))
    assert failed["task_id"] == "t2"
    assert failed["status"] == "failed"


def test_topics_parse_and_report_cli(tmp_path: Path, monkeypatch) -> None:
    results_path = tmp_path / "results.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "report.md"
    results_path.write_text(
        json.dumps(
            {
                "task_id": "karl_synth_episode_000001",
                "status": "completed",
                "output_text": json.dumps([{"prompt": valid_prompt(), "reference_answer": valid_answer()}]),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pi-trec",
            "topics",
            "parse",
            "--input-file",
            str(results_path),
            "--output-file",
            str(candidates_path),
            "--rejected-output",
            str(rejected_path),
            "--summary-output",
            str(summary_path),
            "--candidates-per-episode",
            "1",
        ],
    )
    main()
    assert json.loads(candidates_path.read_text(encoding="utf-8"))["source_task_id"] == "karl_synth_episode_000001"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pi-trec",
            "topics",
            "report",
            "--input-file",
            str(candidates_path),
            "--summary-input",
            str(summary_path),
            "--output-file",
            str(report_path),
        ],
    )
    main()
    assert "Accepted candidates: 1" in report_path.read_text(encoding="utf-8")
