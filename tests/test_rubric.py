import asyncio
import json
import math
from pathlib import Path

from pi_trec.config import RubricAuthorConfig, RubricEvalConfig, RubricGradeConfig, RubricScoreConfig
from pi_trec.rubric import (
    author,
    cell_score,
    compute_scores,
    direct_grade_inputs,
    finalize_rubric,
    grade,
    iter_author_inputs,
    iter_grade_payloads_from_files,
    normalize_criteria,
    parse_rubric,
    parse_verdict_list,
    render_author_prompt,
    render_grade_prompt,
    rubric_eval_pipeline,
    score_response,
    verdict_value,
)


def _fake_agent(tmp_path: Path, output_text: str) -> Path:
    agent = tmp_path / "fake_pi.py"
    payload = json.dumps({"type": "message_end", "message": {"role": "assistant", "content": output_text}})
    agent.write_text(f"#!/usr/bin/env python3\nprint({json.dumps(payload)})\n", encoding="utf-8")
    agent.chmod(0o755)
    return agent


def _stateful_agent(tmp_path: Path, outputs: list[str]) -> Path:
    """Fake Pi agent emitting ``outputs[i]`` on its i-th call (last value sticks)."""
    agent = tmp_path / "stateful_pi.py"
    counter = tmp_path / "calls.txt"
    payloads = [json.dumps({"type": "message_end", "message": {"role": "assistant", "content": text}}) for text in outputs]
    agent.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib\n"
        f"counter = pathlib.Path({json.dumps(str(counter))})\n"
        "n = int(counter.read_text()) if counter.exists() else 0\n"
        "counter.write_text(str(n + 1))\n"
        f"payloads = {json.dumps(payloads)}\n"
        "print(payloads[min(n, len(payloads) - 1)])\n",
        encoding="utf-8",
    )
    agent.chmod(0o755)
    return agent


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- Scoring math (ResearchRubrics Eq. 1, positive-only) --------------------


def test_score_response_normalized_weighted_sum() -> None:
    criteria = [
        {"weight": 5, "verdict": "satisfied"},
        {"weight": 2, "verdict": "partially_satisfied"},
        {"weight": 3, "verdict": "not_satisfied"},
    ]
    assert score_response(criteria) == (5 * 1.0 + 2 * 0.5 + 3 * 0.0) / (5 + 2 + 3)


def test_binary_collapses_partial_to_not_satisfied() -> None:
    criteria = [
        {"weight": 4, "verdict": "satisfied"},
        {"weight": 4, "verdict": "partially_satisfied"},
    ]
    assert score_response(criteria, binary=False) == (4 + 2) / 8
    assert score_response(criteria, binary=True) == (4 + 0) / 8


def test_verdict_value_and_empty_rubric() -> None:
    assert verdict_value("Partially Satisfied", binary=False) == 0.5
    assert verdict_value("partially satisfied", binary=True) == 0.0
    assert verdict_value("nonsense", binary=False) == 0.0
    assert math.isnan(score_response([]))  # no positive weight -> undefined


# --- Parsing & rubric finalization ------------------------------------------


def test_parse_rubric_extracts_authored_fields() -> None:
    text = """```json
    [
      {"text": "Conveys fact X", "type": "explicit", "source_nugget": "n0"},
      {"text": "Conveys fact Y", "type": "bogus", "source_nugget": ""}
    ]
    ```"""
    criteria = parse_rubric(text)
    assert criteria is not None and len(criteria) == 2
    assert criteria[0] == {"text": "Conveys fact X", "type": "explicit", "source_nugget": "n0"}
    assert criteria[1]["type"] == "explicit"  # unknown type defaults
    assert criteria[1]["source_nugget"] is None
    assert "weight" not in criteria[0]  # weight is assigned by finalize_rubric, not the parser


def test_finalize_rubric_assigns_weights_and_covers_vitals() -> None:
    nuggets = [
        {"text": "X", "importance": "vital", "id": "n0"},
        {"text": "Y", "importance": "okay", "id": "n1"},
        {"text": "Z", "importance": "vital", "id": "n2"},
    ]
    authored = [
        {"text": "Conveys X", "type": "explicit", "source_nugget": "n0"},
        {"text": "Conveys Y", "type": "explicit", "source_nugget": "n1"},
        {"text": "Synthesizes well", "type": "synthesis", "source_nugget": None},
    ]
    by_text = {c["text"]: c for c in finalize_rubric(authored, nuggets)}
    # weight/tier come from the source nugget's importance, never the model
    assert by_text["Conveys X"]["weight"] == 5 and by_text["Conveys X"]["tier"] == "mandatory"
    assert by_text["Conveys Y"]["weight"] == 2 and by_text["Conveys Y"]["tier"] == "optional"
    assert by_text["Synthesizes well"]["weight"] == 2  # topic-level -> optional
    # the uncovered vital n2 gets a mandatory criterion injected so the contract always holds
    injected = next(c for c in by_text.values() if c["source_nugget"] == "n2")
    assert injected["tier"] == "mandatory" and injected["weight"] == 5
    assert all(c["weight"] > 0 for c in by_text.values())  # criteria are positive-only


def test_normalize_criteria_drops_non_positive_weights() -> None:
    criteria = normalize_criteria(
        [
            {"text": "zero weight", "type": "explicit", "tier": "optional", "weight": 0},
            {"text": "negative weight", "type": "explicit", "tier": "optional", "weight": -3},
            {"text": "keep me", "type": "explicit", "tier": "mandatory", "weight": 4},
        ]
    )
    # reading a frozen/hand-edited rubric drops weight<=0 (criteria are positive-only)
    assert [c["text"] for c in criteria] == ["keep me"]


def test_render_prompts_have_no_negative_text() -> None:
    author_prompt = render_author_prompt(query="q", nuggets=[{"text": "X", "importance": "vital"}]).lower()
    grade_prompt = render_grade_prompt(query="q", answer="a", criteria=[{"text": "c", "weight": 5}]).lower()
    for blob in (author_prompt, grade_prompt):
        assert "negative" not in blob and "contradict" not in blob and "failure" not in blob


def test_format_criteria_numbers_each_line() -> None:
    from pi_trec.rubric.prompts import format_criteria

    assert format_criteria([{"text": "first"}, {"text": "second"}]) == "1. first\n2. second"


def test_parse_rubric_rejects_non_array() -> None:
    assert parse_rubric("no list here") is None
    assert parse_rubric("[]") is None


def test_parse_verdict_list_roundtrips() -> None:
    assert parse_verdict_list('["satisfied", "not_satisfied"]') == ["satisfied", "not_satisfied"]


# --- Stages: author inputs and the answer x rubric join ---------------------


def test_iter_author_inputs_from_nuggets() -> None:
    inputs = iter_author_inputs(
        [{"qid": "q1", "query": "what is python", "nuggets": [{"text": "a language", "importance": "vital"}, "is interpreted"]}]
    )
    assert len(inputs) == 1
    assert inputs[0]["qid"] == "q1"
    assert inputs[0]["nuggets"][0] == {"text": "a language", "importance": "vital", "id": "n0"}
    assert "what is python" in render_author_prompt(query=inputs[0]["query"], nuggets=inputs[0]["nuggets"])


def test_direct_grade_inputs_join_shape() -> None:
    payload = {
        "answer_record": {"run_id": "runA", "qid": "q1", "topic": "q?", "answer_text": "Python is a language."},
        "rubric_record": {"qid": "q1", "criteria": [{"text": "Conveys X", "weight": 5, "tier": "mandatory"}]},
    }
    items = direct_grade_inputs(payload)
    assert len(items) == 1
    assert items[0]["run_id"] == "runA"
    assert items[0]["qid"] == "q1"
    assert items[0]["criteria"][0]["weight"] == 5
    assert render_grade_prompt(query=items[0]["query"], answer=items[0]["context"], criteria=items[0]["criteria"])


def test_iter_grade_payloads_joins_by_qid(tmp_path: Path) -> None:
    answers = tmp_path / "answers.jsonl"
    answers.write_text(
        "\n".join(json.dumps(r) for r in [{"run_id": "runA", "qid": "q1", "answer_text": "x"}, {"run_id": "runA", "qid": "q9", "answer_text": "y"}]) + "\n",
        encoding="utf-8",
    )
    rubric = tmp_path / "rubric.jsonl"
    rubric.write_text(json.dumps({"qid": "q1", "criteria": []}) + "\n", encoding="utf-8")
    payloads = list(iter_grade_payloads_from_files(answers, rubric))
    assert len(payloads) == 1  # q9 has no rubric and is skipped
    assert payloads[0]["answer_record"]["qid"] == "q1"


# --- Flows with a fake Pi agent ---------------------------------------------


def test_author_writes_parsed_rubric(tmp_path: Path) -> None:
    nuggets = tmp_path / "nuggets.jsonl"
    nuggets.write_text(json.dumps({"qid": "q1", "query": "q?", "nuggets": [{"text": "X", "importance": "vital"}]}) + "\n", encoding="utf-8")
    rubric_json = json.dumps([{"text": "Conveys X", "type": "explicit", "source_nugget": "n0"}])
    config = RubricAuthorConfig(
        input_file=nuggets,
        output_file=tmp_path / "rubric.jsonl",
        cache_dir=None,
        agent_binary=str(_fake_agent(tmp_path, rubric_json)),
        model="m",
        thinking="medium",
        agent_state_dir=tmp_path / "missing",
        max_concurrency=1,
    )
    asyncio.run(author(config))
    rows = _rows(config.output_file)
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    # weight/tier are code-assigned from the vital nugget
    assert rows[0]["criteria"][0]["weight"] == 5 and rows[0]["criteria"][0]["tier"] == "mandatory"


def test_grade_writes_verdicts(tmp_path: Path) -> None:
    grade_inputs = tmp_path / "grade_inputs.jsonl"
    grade_inputs.write_text(
        json.dumps(
            {
                "query": "q?",
                "answer": "Python is a language.",
                "qid": "q1",
                "run_id": "runA",
                "criteria": [
                    {"text": "Conveys X", "weight": 5, "tier": "mandatory", "type": "explicit"},
                    {"text": "Conveys Y", "weight": 5, "tier": "optional", "type": "explicit"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = RubricGradeConfig(
        input_file=grade_inputs,
        output_file=tmp_path / "graded.jsonl",
        cache_dir=None,
        agent_binary=str(_fake_agent(tmp_path, '["satisfied", "not_satisfied"]')),
        model="m",
        thinking="medium",
        agent_state_dir=tmp_path / "missing",
        max_concurrency=1,
    )
    asyncio.run(grade(config))
    rows = _rows(config.output_file)
    assert len(rows) == 1
    assert [c["verdict"] for c in rows[0]["criteria"]] == ["satisfied", "not_satisfied"]
    assert cell_score(rows[0]).ternary_score == (5 + 0) / 10


def test_grade_resume_skips_completed(tmp_path: Path) -> None:
    grade_inputs = tmp_path / "grade_inputs.jsonl"
    payloads = [
        {"task_id": "runA:q1", "query": "q?", "answer": "a", "qid": "q1", "run_id": "runA", "criteria": [{"text": "c", "weight": 5, "tier": "mandatory", "type": "explicit"}]},
        {"task_id": "runA:q2", "query": "q?", "answer": "b", "qid": "q2", "run_id": "runA", "criteria": [{"text": "c", "weight": 5, "tier": "mandatory", "type": "explicit"}]},
    ]
    grade_inputs.write_text("\n".join(json.dumps(p) for p in payloads) + "\n", encoding="utf-8")
    output = tmp_path / "graded.jsonl"
    output.write_text(json.dumps({"task_id": "runA:q1", "status": "completed", "qid": "q1", "run_id": "runA", "criteria": []}) + "\n", encoding="utf-8")
    config = RubricGradeConfig(
        input_file=grade_inputs,
        output_file=output,
        resume=True,
        cache_dir=None,
        agent_binary=str(_fake_agent(tmp_path, '["satisfied"]')),
        model="m",
        thinking="medium",
        agent_state_dir=tmp_path / "missing",
        max_concurrency=1,
    )
    asyncio.run(grade(config))
    by_id = {r["task_id"]: r for r in _rows(output)}
    assert by_id["runA:q1"]["criteria"] == []  # untouched
    assert by_id["runA:q2"]["criteria"][0]["verdict"] == "satisfied"  # newly graded


# --- Retry busts the cache on a parse miss (cache on by default) -------------


def test_author_retry_busts_cache_on_parse_miss(tmp_path: Path) -> None:
    nuggets = tmp_path / "nuggets.jsonl"
    nuggets.write_text(json.dumps({"qid": "q1", "query": "q?", "nuggets": [{"text": "X", "importance": "vital"}]}) + "\n", encoding="utf-8")
    good = json.dumps([{"text": "Conveys X", "type": "explicit", "source_nugget": "n0"}])
    config = RubricAuthorConfig(
        input_file=nuggets,
        output_file=tmp_path / "rubric.jsonl",
        cache_dir=tmp_path / "cache",  # caching on: trial 0's unparseable miss is cached
        agent_binary=str(_stateful_agent(tmp_path, ["no rubric here", good])),
        model="m",
        thinking="medium",
        agent_state_dir=tmp_path / "missing",
        max_concurrency=1,
        max_trials=2,
    )
    asyncio.run(author(config))
    rows = _rows(config.output_file)
    # Without cache-busting, trial 1 re-reads the cached miss and authoring fails.
    assert len(rows) == 1 and rows[0]["status"] == "completed"
    assert rows[0]["criteria"][0]["text"] == "Conveys X"


def test_grade_retry_busts_cache_on_parse_miss(tmp_path: Path) -> None:
    grade_inputs = tmp_path / "grade_inputs.jsonl"
    grade_inputs.write_text(
        json.dumps(
            {
                "query": "q?",
                "answer": "Python is a language.",
                "qid": "q1",
                "run_id": "runA",
                "criteria": [{"text": "Conveys X", "weight": 5, "tier": "mandatory", "type": "explicit"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = RubricGradeConfig(
        input_file=grade_inputs,
        output_file=tmp_path / "graded.jsonl",
        cache_dir=tmp_path / "cache",
        agent_binary=str(_stateful_agent(tmp_path, ["not a parseable list", '["satisfied"]'])),
        model="m",
        thinking="medium",
        agent_state_dir=tmp_path / "missing",
        max_concurrency=1,
        max_trials=2,
    )
    asyncio.run(grade(config))
    rows = _rows(config.output_file)
    # The cached parse-miss would otherwise pin every retry to a "failed" verdict.
    assert rows[0]["criteria"][0]["verdict"] == "satisfied"


# --- Scoring outputs --------------------------------------------------------


def test_compute_scores_writes_csvs_and_correlation(tmp_path: Path) -> None:
    graded = tmp_path / "graded.jsonl"
    strong = {"text": "c", "weight": 5, "tier": "mandatory", "type": "explicit", "verdict": "satisfied"}
    weak = {"text": "c", "weight": 5, "tier": "mandatory", "type": "explicit", "verdict": "partially_satisfied"}
    graded.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"qid": "q1", "run_id": "runA", "criteria": [strong]},
                {"qid": "q1", "run_id": "runB", "criteria": [weak]},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "scores"
    compute_scores(RubricScoreConfig(input_file=graded, output_dir=out))
    for name in ("cell_scores.csv", "run_scores.csv", "category_failures.csv", "correlations.csv"):
        assert (out / name).exists()
    run_csv = (out / "run_scores.csv").read_text(encoding="utf-8")
    # runB's ternary (0.5) and binary (0.0) differ; runA is 1.0 / 1.0.
    assert "runB,1,0.5,0.0" in run_csv
    assert "runA,1,1.0,1.0" in run_csv


def test_rubric_eval_pipeline_with_fixed_rubric(tmp_path: Path) -> None:
    rubric = tmp_path / "rubric.jsonl"
    rubric.write_text(
        json.dumps(
            {
                "qid": "q1",
                "query": "q?",
                "criteria": [{"text": "Conveys X", "weight": 5, "tier": "mandatory", "type": "explicit"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    answers = tmp_path / "answers.jsonl"
    answers.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"run_id": "runA", "qid": "q1", "topic": "q?", "answer_text": "Python is a language."},
                {"run_id": "runB", "qid": "q1", "topic": "q?", "answer_text": "Coffee is a drink."},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = RubricEvalConfig(
        rubric_file=rubric,
        answers_file=answers,
        output_dir=tmp_path / "out",
        cache_dir=None,
        agent_binary=str(_fake_agent(tmp_path, '["satisfied"]')),  # one criterion per cell
        model="m",
        thinking="medium",
        agent_state_dir=tmp_path / "missing",
        max_concurrency=1,
    )
    config.validate()
    asyncio.run(rubric_eval_pipeline(config))
    out = config.output_dir
    assert {p.name for p in out.iterdir()} >= {"grade_inputs.jsonl", "graded.jsonl", "scores"}
    graded = _rows(out / "graded.jsonl")
    assert len(graded) == 2  # both runs graded against the fixed rubric
    assert all([c["weight"] for c in r["criteria"]] == [5] for r in graded)
    assert (out / "scores" / "run_scores.csv").exists()
