import asyncio
import json
from pathlib import Path

import yaml

from pi_trec.config import SupportJudgeConfig, SupportMetricsConfig
from pi_trec.support import (
    SUPPORT_EVAL_PROMPT,
    compute_metrics,
    iter_support_tasks,
    judge,
    parse_support_label,
    render_support_prompt,
    support_metric,
)

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
                {"citations": [{"support": "0"}]},
                {"citations": [{"support": "-1"}]},
                {"citations": []},
            ],
        }
    )
    assert metric.topic_id == "14"
    assert metric.run_id == "r1"
    assert metric.weighted_precision == 0.5
    assert metric.hard_precision == 1 / 3
    assert metric.weighted_recall == 0.375
    assert metric.hard_recall == 0.25
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
            "weighted_precision": 1.0,
            "hard_precision": 1.0,
            "weighted_recall": 1.0,
            "hard_recall": 1.0,
            "sentences": 1,
        },
        {
            "topic_id": "15",
            "run_id": "r2",
            "weighted_precision": 0.0,
            "hard_precision": 0.0,
            "weighted_recall": 0.0,
            "hard_recall": 0.0,
            "sentences": 1,
        },
    ]


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
