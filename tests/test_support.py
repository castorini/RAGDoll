import asyncio
import json
from pathlib import Path

import yaml

from pi_trec.config import (
    SupportAssembleConfig,
    SupportJudgeConfig,
    SupportMetricRowsConfig,
    SupportMetricsConfig,
    SupportResolveReferencesConfig,
    SupportSummarizeConfig,
)
from pi_trec.support import (
    SUPPORT_EVAL_PROMPT,
    assemble,
    assemble_support_assignments,
    compute_metrics,
    iter_support_tasks,
    judge,
    metric_lines,
    parse_support_label,
    render_support_prompt,
    resolve_references,
    summarize,
    support_metric,
    write_metric_rows,
)
from pi_trec.support import resolve as resolve_mod

UPSTREAM = Path(__file__).resolve().parent / "fixtures" / "upstream"
SUPPORT_PROMPT_FIXTURE = UPSTREAM / "trec2024-rag" / "support_evaluation_codex_gpt5_5.yaml"


def _fake_agent(tmp_path: Path, output_text: str) -> Path:
    agent = tmp_path / "fake_pi.py"
    payload = json.dumps({"type": "message_end", "message": {"role": "assistant", "content": output_text}})
    agent.write_text(f"#!/usr/bin/env python3\nprint({json.dumps(payload)})\n", encoding="utf-8")
    agent.chmod(0o755)
    return agent


def test_support_prompt_matches_fixture() -> None:
    assert SUPPORT_EVAL_PROMPT == yaml.safe_load(SUPPORT_PROMPT_FIXTURE.read_text(encoding="utf-8"))["prefix_user"]


def test_support_prompt_includes_sentence_context() -> None:
    prompt = render_support_prompt(statement="s", citation="c", sentence_context="before **s** after")
    assert "Cited Passage: c" in prompt
    assert "Sentence: s" in prompt
    assert "Sentence Context: before **s** after" in prompt


def test_support_tasks_accept_direct_rows() -> None:
    tasks = iter_support_tasks([{"task_id": "t", "statement": "s", "citation": "c"}])
    assert tasks[0]["task_id"] == "t"
    assert "Sentence: s" in tasks[0]["instruction"]
    assert "Sentence Context: **s**" in tasks[0]["instruction"]


def test_support_tasks_accept_direct_rows_with_sentence_context() -> None:
    tasks = iter_support_tasks(
        [
            {
                "task_id": "t",
                "statement": "It helps.",
                "citation": "The source says peer support helps.",
                "sentence_context": "The response discusses peer support. It helps.",
            }
        ]
    )
    expected = "The response discusses peer support. **It helps.**"
    assert f"Sentence Context: {expected}" in tasks[0]["instruction"]
    assert tasks[0]["metadata"]["sentence_context"] == expected


def test_support_tasks_accept_resolved_trec_answer_rows() -> None:
    tasks = iter_support_tasks(
        [
            {
                "topic_id": "q1",
                "run_id": "r1",
                "references": ["d1"],
                "segments": {"d1": "citation text"},
                "answer": [{"text": "statement", "citations": [0]}],
            }
        ]
    )
    assert tasks[0]["task_id"] == "r1:q1:s0:c0"
    assert tasks[0]["metadata"]["citation"] == "citation text"
    assert tasks[0]["metadata"]["sentence_context"] == "**statement**"


def test_support_tasks_accept_official_metadata_with_segments() -> None:
    tasks = iter_support_tasks(
        [
            {
                "metadata": {"narrative_id": "14", "run_id": "r1"},
                "references": ["d1"],
                "segments": {"d1": "citation text"},
                "answer": [{"text": "statement", "citations": [0]}],
            }
        ]
    )
    assert tasks[0]["task_id"] == "r1:14:s0:c0"


def test_support_tasks_build_full_answer_context() -> None:
    tasks = iter_support_tasks(
        [
            {
                "topic_id": "q1",
                "run_id": "r1",
                "references": ["d1"],
                "segments": {"d1": "citation text"},
                "answer": [
                    {"text": "First sentence.", "citations": []},
                    {"text": "Second sentence.", "citations": [0]},
                    {"text": "Third sentence.", "citations": []},
                ],
            }
        ]
    )
    expected = "First sentence. **Second sentence.** Third sentence."
    assert tasks[0]["metadata"]["sentence_context"] == expected
    assert f"Sentence Context: {expected}" in tasks[0]["instruction"]


def test_parse_support_label() -> None:
    assert parse_support_label("Full Support") == "FS"
    assert parse_support_label("partial support") == "PS"
    assert parse_support_label("No Support") == "NS"
    assert parse_support_label("unknown") is None


def test_support_metric_matches_reference_arithmetic() -> None:
    metric = support_metric(
        {
            "narrative_id": "14",
            "run_id": "r1",
            "sentences": [
                {"citations": [{"support": "2"}]},
                {"citations": [{"support": "1"}]},
                {"citations": [{"support": "0"}, {"support": "2"}]},
                {"citations": [{"support": "-1"}]},
                {"citations": []},
            ],
        }
    )
    assert metric.topic_id == "14"
    assert metric.run_id == "r1"
    assert metric.weighted_precision_first_citation == 0.5
    assert metric.hard_precision == 1 / 3
    assert metric.weighted_recall_first_citation == 0.375
    assert metric.hard_recall == 0.25
    assert metric.weighted_precision_all_judged_citations == 2 / 3
    assert metric.weighted_recall_all_judged_citations == 0.5
    assert metric.sentences == 5


def test_support_metrics_writes_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "human.jsonl"
    output_path = tmp_path / "metrics.jsonl"
    input_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"narrative_id": "14", "run_id": "r1", "sentences": [{"citations": [{"support": "2"}]}]},
                {"topic_id": "15", "run_id": "r2", "sentences": [{"citations": []}]},
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    compute_metrics(SupportMetricsConfig(input_file=input_path, output_file=output_path))

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "topic_id": "14",
            "run_id": "r1",
            "weighted_precision_first_citation": 1.0,
            "weighted_recall_first_citation": 1.0,
            "weighted_precision_all_judged_citations": 1.0,
            "weighted_recall_all_judged_citations": 1.0,
            "hard_precision": 1.0,
            "hard_recall": 1.0,
            "sentences": 1,
        },
        {
            "topic_id": "15",
            "run_id": "r2",
            "weighted_precision_first_citation": 0.0,
            "weighted_recall_first_citation": 0.0,
            "weighted_precision_all_judged_citations": 0.0,
            "weighted_recall_all_judged_citations": 0.0,
            "hard_precision": 0.0,
            "hard_recall": 0.0,
            "sentences": 1,
        },
    ]


def test_metric_lines_group_by_run_metric_then_topic() -> None:
    rows = [
        {
            "topic_id": "72",
            "run_id": "activity-manufacturer",
            "weighted_precision_first_citation": 0.521,
            "weighted_recall_first_citation": 0.521,
            "weighted_precision_all_judged_citations": 0.6,
            "weighted_recall_all_judged_citations": 0.6,
        },
        {
            "topic_id": "14",
            "run_id": "activity-manufacturer",
            "weighted_precision_first_citation": 0.433333,
            "weighted_recall_first_citation": 0.433333,
            "weighted_precision_all_judged_citations": 0.5,
            "weighted_recall_all_judged_citations": 0.5,
        },
        {
            "topic_id": "200",
            "run_id": "angle-apartment",
            "weighted_precision_first_citation": 0.155,
            "weighted_recall_first_citation": 0.155,
            "weighted_precision_all_judged_citations": 0.2,
            "weighted_recall_all_judged_citations": 0.2,
        },
    ]
    assert metric_lines(rows) == [
        "activity-manufacturer 14 weighted_precision_first_citation 0.433333",
        "activity-manufacturer 72 weighted_precision_first_citation 0.521",
        "activity-manufacturer 14 weighted_recall_first_citation 0.433333",
        "activity-manufacturer 72 weighted_recall_first_citation 0.521",
        "activity-manufacturer 14 weighted_precision_all_judged_citations 0.5",
        "activity-manufacturer 72 weighted_precision_all_judged_citations 0.6",
        "activity-manufacturer 14 weighted_recall_all_judged_citations 0.5",
        "activity-manufacturer 72 weighted_recall_all_judged_citations 0.6",
        "angle-apartment 200 weighted_precision_first_citation 0.155",
        "angle-apartment 200 weighted_recall_first_citation 0.155",
        "angle-apartment 200 weighted_precision_all_judged_citations 0.2",
        "angle-apartment 200 weighted_recall_all_judged_citations 0.2",
    ]


def test_write_metric_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "metrics.jsonl"
    output_path = tmp_path / "metric-rows.txt"
    input_path.write_text(
        '{"topic_id":"14","run_id":"r1","weighted_precision_first_citation":0.25,"weighted_recall_first_citation":0.125,"weighted_precision_all_judged_citations":0.5,"weighted_recall_all_judged_citations":0.25,"hard_precision":0.0,"hard_recall":0.0,"sentences":2}\n',
        encoding="utf-8",
    )

    write_metric_rows(SupportMetricRowsConfig(input_file=input_path, output_file=output_path, metrics=["hard_precision"]))

    assert output_path.read_text(encoding="utf-8") == "r1 14 hard_precision 0\n"


def test_resolve_references_with_pyserini_http(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "answers.jsonl"
    output_path = tmp_path / "resolved.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "metadata": {"team_id": "team", "run_id": "r1", "narrative_id": "14"},
                "references": ["doc-a"],
                "answer": [{"text": "Answer sentence.", "citations": [0]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_read(config, request_body):
        assert config.pyserini_base_url == "http://pyserini"
        assert config.pyserini_index == "climbmix"
        assert request_body["docid"] == "doc-a"
        return {"found": True, "docid": "doc-a", "text": "Resolved passage."}

    monkeypatch.setattr(resolve_mod, "read_pyserini_document", fake_read)

    resolve_references(
        SupportResolveReferencesConfig(
            input_file=input_path,
            output_file=output_path,
            pyserini_api="http://pyserini",
            pyserini_index="climbmix",
        )
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["topic_id"] == "14"
    assert row["run_id"] == "r1"
    assert row["segments"] == {"doc-a": "Resolved passage."}


def test_resolve_references_with_local_pyserini_index(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "answers.jsonl"
    output_path = tmp_path / "resolved.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "metadata": {"run_id": "r1", "narrative_id": "14"},
                "references": ["doc-a"],
                "answer": [{"text": "Answer sentence.", "citations": [0]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeDoc:
        def raw(self) -> str:
            return json.dumps({"title": "Doc title", "segment": "Segment text."})

    class FakeSearcher:
        def doc(self, docid: str):
            assert docid == "doc-a"
            return FakeDoc()

    monkeypatch.setattr(resolve_mod, "_lucene_searcher", lambda index: FakeSearcher())

    resolve_references(
        SupportResolveReferencesConfig(
            input_file=input_path,
            output_file=output_path,
            pyserini_index=str(tmp_path / "index"),
        )
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["segments"] == {"doc-a": "Doc title: Segment text."}


def test_resolve_references_with_pyserini_prebuilt_index(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "answers.jsonl"
    output_path = tmp_path / "resolved.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "metadata": {"run_id": "r1", "narrative_id": "14"},
                "references": ["doc-a"],
                "answer": [{"text": "Answer sentence.", "citations": [0]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeDoc:
        def raw(self) -> str:
            return json.dumps({"contents": "Prebuilt passage."})

    class FakeSearcher:
        @classmethod
        def from_prebuilt_index(cls, index: str):
            assert index == "msmarco-v2.1-doc-segmented"
            return cls()

        def doc(self, docid: str):
            assert docid == "doc-a"
            return FakeDoc()

    monkeypatch.setattr(resolve_mod, "_lucene_searcher_cls", lambda: FakeSearcher)

    resolve_references(
        SupportResolveReferencesConfig(
            input_file=input_path,
            output_file=output_path,
            pyserini_index="msmarco-v2.1-doc-segmented",
        )
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["segments"] == {"doc-a": "Prebuilt passage."}


def test_resolve_references_rejects_invalid_citation_index(tmp_path: Path) -> None:
    input_path = tmp_path / "answers.jsonl"
    output_path = tmp_path / "resolved.jsonl"
    input_path.write_text(
        '{"metadata":{"run_id":"r1","narrative_id":"14"},"references":["doc-a"],"answer":[{"text":"s","citations":[2]}]}\n',
        encoding="utf-8",
    )

    try:
        resolve_references(
            SupportResolveReferencesConfig(
                input_file=input_path,
                output_file=output_path,
                pyserini_api="http://pyserini",
                pyserini_index="climbmix",
            )
        )
    except ValueError as exc:
        assert "outside references" in str(exc)
    else:
        raise AssertionError("expected invalid citation index to fail")


def test_assemble_support_assignments_preserves_answer_shape(tmp_path: Path) -> None:
    answers = tmp_path / "answers.jsonl"
    judgments = tmp_path / "judgments.parsed.jsonl"
    answers.write_text(
        json.dumps(
            {
                "metadata": {"run_id": "r1", "narrative_id": "14"},
                "references": ["doc-a", "doc-b"],
                "answer": [
                    {"text": "first", "citations": [0, 1]},
                    {"text": "uncited", "citations": []},
                    {"text": "missing judgment", "citations": [1]},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    judgments.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "status": "completed",
                    "support_label": "FS",
                    "metadata": {"run_id": "r1", "topic_id": "14", "sentence_index": 0, "citation_index": 0},
                },
                {
                    "status": "completed",
                    "support_label": "PS",
                    "metadata": {"run_id": "r1", "topic_id": "14", "sentence_index": 0, "citation_index": 1},
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = assemble_support_assignments(answers, [judgments])

    assert rows == [
        {
            "topic_id": "14",
            "narrative_id": "14",
            "run_id": "r1",
            "sentences": [
                {
                    "sentenceID": 0,
                    "text": "first",
                    "citations": [
                        {"citationID": 0, "reference": "doc-a", "support": "2"},
                        {"citationID": 1, "reference": "doc-b", "support": "1"},
                    ],
                },
                {"sentenceID": 1, "text": "uncited", "citations": []},
                {
                    "sentenceID": 2,
                    "text": "missing judgment",
                    "citations": [{"citationID": 0, "reference": "doc-b", "support": "-1"}],
                },
            ],
        }
    ]
    metric = support_metric(rows[0])
    assert metric.weighted_precision_first_citation == 1.0
    assert metric.weighted_recall_first_citation == 0.5
    assert metric.weighted_precision_all_judged_citations == 0.75
    assert metric.weighted_recall_all_judged_citations == 0.375


def test_assemble_writes_assignments(tmp_path: Path) -> None:
    answers = tmp_path / "r1.jsonl"
    judgments = tmp_path / "judgments.parsed.jsonl"
    output = tmp_path / "support_assignments.jsonl"
    answers.write_text('{"topic_id":"14","answer":[{"text":"s","citations":["d"]}]}\n', encoding="utf-8")
    judgments.write_text(
        '{"status":"completed","raw_output":"No Support","metadata":{"run_id":"r1","topic_id":"14","sentence_index":0,"citation_index":0}}\n',
        encoding="utf-8",
    )

    assemble(SupportAssembleConfig(answers_file=answers, judgments=[judgments], output_file=output, run_id="r1"))

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["run_id"] == "r1"
    assert row["sentences"][0]["citations"][0]["support"] == "0"


def test_assemble_support_assignments_accepts_responses_with_citation_map(tmp_path: Path) -> None:
    answers = tmp_path / "r1.jsonl"
    judgments = tmp_path / "judgments.parsed.jsonl"
    answers.write_text(
        json.dumps(
            {
                "metadata": {"run_id": "r1", "narrative_id": "14"},
                "responses": [
                    {"text": "s1", "citations": {"doc-a": 1.0}},
                    {"text": "s2", "citations": {"doc-b": 1.0, "doc-c": 0.5}},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    judgments.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "status": "completed",
                    "support_label": "FS",
                    "metadata": {"run_id": "r1", "topic_id": "14", "sentence_index": 0, "citation_index": 0},
                },
                {
                    "status": "completed",
                    "support_label": "NS",
                    "metadata": {"run_id": "r1", "topic_id": "14", "sentence_index": 1, "citation_index": 0},
                },
                {
                    "status": "completed",
                    "support_label": "PS",
                    "metadata": {"run_id": "r1", "topic_id": "14", "sentence_index": 1, "citation_index": 1},
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = assemble_support_assignments(answers, [judgments])

    assert rows[0]["sentences"] == [
        {"sentenceID": 0, "text": "s1", "citations": [{"citationID": 0, "reference": "doc-a", "support": "2"}]},
        {
            "sentenceID": 1,
            "text": "s2",
            "citations": [
                {"citationID": 0, "reference": "doc-b", "support": "0"},
                {"citationID": 1, "reference": "doc-c", "support": "1"},
            ],
        },
    ]
    metric = support_metric(rows[0])
    assert metric.weighted_precision_first_citation == 0.5
    assert metric.weighted_precision_all_judged_citations == 0.625


def test_summarize_writes_assignments_metrics_and_rows(tmp_path: Path) -> None:
    answers_dir = tmp_path / "answers"
    judgments_root = tmp_path / "judgments"
    output_dir = tmp_path / "out"
    answers_dir.mkdir()
    (judgments_root / "r1").mkdir(parents=True)
    (answers_dir / "r1").write_text(
        '{"topic_id":"14","answer":[{"text":"s","citations":["d"]},{"text":"u","citations":[]}]}\n',
        encoding="utf-8",
    )
    (judgments_root / "r1" / "judgments.parsed.jsonl").write_text(
        '{"status":"completed","support_label":"FS","metadata":{"run_id":"r1","topic_id":"14","sentence_index":0,"citation_index":0}}\n',
        encoding="utf-8",
    )

    summarize(SupportSummarizeConfig(answers_dir=answers_dir, judgments_root=judgments_root, output_dir=output_dir))

    assert (output_dir / "support_assignments.jsonl").exists()
    metrics = [json.loads(line) for line in (output_dir / "support_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert metrics[0]["weighted_recall_first_citation"] == 0.5
    assert (output_dir / "support_metric_rows.txt").read_text(encoding="utf-8") == (
        "r1 14 weighted_precision_first_citation 1\n"
        "r1 14 weighted_recall_first_citation 0.5\n"
        "r1 14 weighted_precision_all_judged_citations 1\n"
        "r1 14 weighted_recall_all_judged_citations 0.5\n"
    )


def test_support_judge_resume_skips_completed(tmp_path: Path) -> None:
    records = [
        {"task_id": "a", "statement": "s1", "citation": "c1"},
        {"task_id": "b", "statement": "s2", "citation": "c2"},
    ]
    input_path = tmp_path / "in.jsonl"
    input_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    output_path = tmp_path / "out.jsonl"
    output_path.write_text(
        json.dumps({"task_id": "a", "status": "completed", "statement": "s1", "citation": "c1", "support_label": "NS"}) + "\n",
        encoding="utf-8",
    )
    config = SupportJudgeConfig(
        input_file=input_path,
        output_file=output_path,
        resume=True,
        cache_dir=None,
        agent_binary=str(_fake_agent(tmp_path, "Full Support")),
        model="m",
        thinking="medium",
        agent_state_dir=tmp_path / "missing",
        max_concurrency=1,
    )
    asyncio.run(judge(config))
    grouped: dict[str, list[dict]] = {}
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            grouped.setdefault(row["task_id"], []).append(row)
    assert grouped["a"] == [{"task_id": "a", "status": "completed", "statement": "s1", "citation": "c1", "support_label": "NS"}]
    assert len(grouped["b"]) == 1
    assert grouped["b"][0]["support_label"] == "FS"
