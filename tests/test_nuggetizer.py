from pathlib import Path

from pi_trec.nuggetizer import (
    NUGGET_AGENTIC_CREATOR_SYSTEM,
    NUGGET_ASSIGNER_SYSTEM,
    NUGGET_CREATOR_SYSTEM,
    NUGGET_SCORER_SYSTEM,
    create_context,
    direct_assign_inputs,
    iter_agentic_create_tasks,
    iter_assign_tasks,
    iter_create_inputs,
    iter_create_tasks,
    iter_window_bounds,
    normalize_nuggets,
    parse_label_list,
    render_agentic_create_prompt,
    render_assign_prompt,
    render_create_prompt,
    render_score_prompt,
)


def _yaml_part(path: Path, key: str) -> str:
    text = path.read_text(encoding="utf-8")
    marker = f"{key}: "
    if marker in text and f"{key}: |-" not in text:
        return text.split(marker, 1)[1].splitlines()[0].strip().strip('"')
    body = text.split(f"{key}: |-\n", 1)[1]
    return "\n".join(line[2:] if line.startswith("  ") else line for line in body.splitlines())


def test_nugget_create_prompt_matches_source_template() -> None:
    source = Path(__file__).resolve().parent / "fixtures" / "upstream" / "nuggetizer" / "creator_template.yaml"
    expected_system = _yaml_part(source, "system_message")
    expected_user = _yaml_part(source, "prefix_user").format(
        query="q",
        context="ctx",
        nuggets=[],
        nuggets_length=0,
        creator_max_nuggets=30,
    )
    expected_user = expected_user.replace(
        "(this is an iterative process).  Return", "(this is an iterative process). Return"
    )
    assert expected_system == NUGGET_CREATOR_SYSTEM
    assert render_create_prompt(query="q", context="ctx") == expected_user


def test_nugget_score_prompt_matches_source_template() -> None:
    source = Path(__file__).resolve().parent / "fixtures" / "upstream" / "nuggetizer" / "scorer_template.yaml"
    expected_system = _yaml_part(source, "system_message")
    expected_user = _yaml_part(source, "prefix_user").format(query="q", nuggets=["n"], num_nuggets=1)
    assert expected_system == NUGGET_SCORER_SYSTEM
    assert render_score_prompt(query="q", nuggets=["n"]) == expected_user


def test_nugget_assign_3grade_prompt_matches_source_template() -> None:
    source = Path(__file__).resolve().parent / "fixtures" / "upstream" / "nuggetizer" / "assigner_template.yaml"
    expected_system = _yaml_part(source, "system_message")
    expected_user = _yaml_part(source, "prefix_user").format(query="q", context="ctx", nuggets=["n"], num_nuggets=1)
    assert expected_system == NUGGET_ASSIGNER_SYSTEM
    assert render_assign_prompt(
        query="q", context="ctx", nuggets=["n"], assign_mode="support-grade-3"
    ) == expected_user


def test_nugget_assign_2grade_prompt_matches_source_template() -> None:
    source = Path(__file__).resolve().parent / "fixtures" / "upstream" / "nuggetizer" / "assigner_2grade_template.yaml"
    expected_system = _yaml_part(source, "system_message")
    expected_user = _yaml_part(source, "prefix_user").format(query="q", context="ctx", nuggets=["n"], num_nuggets=1)
    assert expected_system == NUGGET_ASSIGNER_SYSTEM
    assert render_assign_prompt(
        query="q", context="ctx", nuggets=["n"], assign_mode="support-grade-2"
    ) == expected_user


def test_create_tasks_accept_query_candidate_schema() -> None:
    tasks = iter_create_tasks(
        [{"query": {"qid": "q1", "text": "query"}, "candidates": [{"doc": {"segment": "passage"}}]}],
        creator_max_nuggets=30,
    )
    assert tasks[0]["task_id"] == "q1"
    assert tasks[0]["system_prompt"] == NUGGET_CREATOR_SYSTEM
    assert "[1] passage" in tasks[0]["metadata"]["context"]


def test_create_context_uses_reference_enumeration() -> None:
    assert create_context(["alpha", {"doc": {"segment": "beta"}}]) == "[1] alpha\n[2] beta"


def test_iter_window_bounds_matches_reference() -> None:
    assert iter_window_bounds(0, 10) == []
    assert iter_window_bounds(5, 10) == [(0, 5)]
    assert iter_window_bounds(25, 10) == [(0, 10), (10, 20), (20, 25)]


def test_iter_create_inputs_keep_raw_candidates_for_windowing() -> None:
    inputs = iter_create_inputs(
        [{"query": {"qid": "q1", "text": "query"}, "candidates": ["a", "b"], "nuggets": [{"text": "seed"}]}]
    )
    assert inputs[0]["task_id"] == "q1"
    assert inputs[0]["candidates"] == ["a", "b"]
    assert inputs[0]["initial_nuggets"] == ["seed"]


def test_agentic_create_prompt_accepts_initial_nuggets() -> None:
    prompt = render_agentic_create_prompt(
        query="what is python used for",
        nuggets=["web development"],
        creator_max_nuggets=5,
    )

    assert "Search Query: what is python used for" in prompt
    assert "Initial Nugget List: ['web development']" in prompt
    assert "at most 5 nuggets" in prompt
    assert "search and read_document tools" in prompt


def test_agentic_create_tasks_accept_query_and_initial_nuggets() -> None:
    tasks = iter_agentic_create_tasks(
        [{"query": {"qid": "q1", "text": "query"}, "nuggets": [{"text": "seed", "importance": "vital"}]}],
        creator_max_nuggets=7,
    )

    assert tasks[0]["task_id"] == "q1"
    assert tasks[0]["evaluator"] == "nugget-agentic-create"
    assert tasks[0]["system_prompt"] == NUGGET_AGENTIC_CREATOR_SYSTEM
    assert tasks[0]["metadata"]["initial_nuggets"] == ["seed"]
    assert "at most 7 nuggets" in tasks[0]["instruction"]


def test_direct_assign_inputs_accepts_direct_and_joined_shapes() -> None:
    direct = direct_assign_inputs({"query": "q", "context": "ctx", "nuggets": [{"text": "n"}]})
    assert direct[0]["nugget_texts"] == ["n"]
    joined = direct_assign_inputs(
        {
            "answer_record": {"topic_id": "q1", "topic": "q", "answer": [{"text": "answer"}]},
            "nugget_record": {"qid": "q1", "query": "q", "nuggets": [{"text": "n"}]},
        }
    )
    assert joined[0]["context"] == "answer"
    assert joined[0]["nuggets"] == [{"text": "n"}]


def test_assign_tasks_materialize() -> None:
    tasks = iter_assign_tasks(
        [{"task_id": "t", "query": "q", "context": "ctx", "nuggets": [{"text": "n"}], "nugget_texts": ["n"], "source": {}}],
        assign_mode="support-grade-3",
    )
    assert tasks[0]["evaluator"] == "nugget-assign"
    assert tasks[0]["system_prompt"] == NUGGET_ASSIGNER_SYSTEM
    assert "Labels:" in tasks[0]["instruction"]


def test_parse_label_list() -> None:
    assert parse_label_list("['support', 'not_support']") == ["support", "not_support"]
    assert parse_label_list("Labels: ['vital']") == ["vital"]
    assert parse_label_list("no list") is None


def test_parse_label_list_tolerates_fences_and_bare_words() -> None:
    assert parse_label_list("```python\n['vital', 'okay']\n```") == ["vital", "okay"]
    assert parse_label_list("[vital, okay]") == ["vital", "okay"]
    assert parse_label_list('["a", "b"]') == ["a", "b"]
    assert parse_label_list("[]") == []


def test_normalize_nuggets_preserves_importance() -> None:
    assert normalize_nuggets([{"text": "n", "importance": "vital"}, "m"]) == [
        {"text": "n", "importance": "vital"},
        {"text": "m"},
    ]
