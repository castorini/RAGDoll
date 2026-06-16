from pathlib import Path

from pi_trec.prompts import SUPPORT_EVAL_PROMPT, parse_support_label
from pi_trec.support import iter_support_tasks, render_support_prompt


def test_support_prompt_matches_source_script() -> None:
    source = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "upstream"
        / "trec2024-rag"
        / "support_evaluation_individual_gpt4o.py"
    )
    text = source.read_text(encoding="utf-8")
    expected = text.split('SUPPORT_EVAL_PROMPT = """', 1)[1].split('"""', 1)[0]
    assert SUPPORT_EVAL_PROMPT == expected
    assert render_support_prompt(statement="s", citation="c") == expected.format(statement="s", citation="c")


def test_support_tasks_accept_direct_rows() -> None:
    tasks = iter_support_tasks([{"task_id": "t", "statement": "s", "citation": "c"}])
    assert tasks[0]["task_id"] == "t"
    assert "Statement: s" in tasks[0]["instruction"]


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


def test_parse_support_label() -> None:
    assert parse_support_label("Full Support") == "FS"
    assert parse_support_label("partial support") == "PS"
    assert parse_support_label("No Support") == "NS"
    assert parse_support_label("unknown") is None

