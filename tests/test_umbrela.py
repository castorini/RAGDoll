import json
from pathlib import Path

from pi_trec.prompts import parse_umbrela_judgment, render_umbrela_prompt
from pi_trec.umbrela import iter_prompt_tasks


def _source_prompt(prompt_type: str) -> str:
    source = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "upstream"
        / "umbrela"
        / f"qrel_zeroshot_{prompt_type}.yaml"
    )
    text = source.read_text(encoding="utf-8")
    expected = text.split("prefix_user: |-\n", 1)[1]
    return "\n".join(line[2:] if line.startswith("  ") else line for line in expected.splitlines())


def test_umbrela_bing_prompt_matches_source_template() -> None:
    expected = _source_prompt("bing")
    assert render_umbrela_prompt(query="q", passage="p", prompt_type="bing") == expected.format(query="q", passage="p")


def test_umbrela_basic_prompt_matches_source_template() -> None:
    expected = _source_prompt("basic")
    assert render_umbrela_prompt(query="q", passage="p", prompt_type="basic") == expected.format(query="q", passage="p")


def test_umbrela_materializes_shared_query_candidate_schema() -> None:
    tasks = iter_prompt_tasks(
        [
            {
                "query": {"qid": "q1", "text": "what is python"},
                "candidates": [
                    "Python is a language.",
                    {"doc": {"docid": "d2", "segment": "Coffee is a drink."}},
                ],
            }
        ],
        prompt_type="basic",
    )
    assert len(tasks) == 2
    assert tasks[0]["metadata"]["qid"] == "q1"
    assert tasks[1]["metadata"]["docid"] == "d2"
    assert "Query: what is python" in tasks[0]["instruction"]


def test_parse_umbrela_judgment() -> None:
    assert parse_umbrela_judgment("##final score: 3") == 3
    assert parse_umbrela_judgment("The relevance category is 2") == 2
    assert parse_umbrela_judgment("nope") is None


def test_materialized_task_is_json_serializable() -> None:
    task = iter_prompt_tasks([{"query": "q", "candidates": ["p"]}], prompt_type="bing")[0]
    json.dumps(task)
