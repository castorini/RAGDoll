import csv
import json
import sys
from pathlib import Path

import pytest

from ragdoll.arena.metrics import fit_arena_ratings, leaderboard_rows, pairwise_rows
from ragdoll.arena.prompts import (
    parse_verdict,
    render_arena_prompt,
)
from ragdoll.arena.stages import (
    assistant_order,
    battles_for_system_degree,
    coverage_rows,
    iter_arena_tasks,
    load_answer_set,
    load_answer_sets,
    load_rubrics,
    sampled_pair_qids,
    sampled_topic_pairs,
)
from ragdoll.cli import main


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_load_answer_set_accepts_normalized_rows(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "a.jsonl",
        [{"run_id": "sys-a", "qid": "q1", "query": "question", "answer_text": "answer"}],
    )

    answer_set = load_answer_set(path)

    assert answer_set.run_id == "sys-a"
    assert answer_set.rows_by_qid["q1"].answer_text == "answer"


def test_load_answer_set_accepts_official_trec_rows_and_concatenates_sentences(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "a.jsonl",
        [
            {
                "metadata": {"run_id": "sys-a", "narrative_id": "q1", "narrative": "question"},
                "answer": [
                    {"text": "Sentence one.", "citations": [0]},
                    {"text": "Sentence two.", "citations": [1]},
                ],
                "references": ["DO_NOT_RENDER"],
            }
        ],
    )

    answer_set = load_answer_set(path)

    assert answer_set.rows_by_qid["q1"].answer_text == "Sentence one. Sentence two."


def test_load_answer_set_accepts_response_rows_and_concatenates_sentences(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "a.jsonl",
        [
            {
                "metadata": {"run_id": "sys-a", "narrative_id": 1, "narrative": "question"},
                "responses": [
                    {"text": "Sentence one.", "citations": [0]},
                    {"text": "Sentence two.", "citations": [1]},
                ],
                "references": ["DO_NOT_RENDER"],
            }
        ],
    )

    answer_set = load_answer_set(path)

    assert answer_set.rows_by_qid["1"].answer_text == "Sentence one. Sentence two."


def test_load_answer_set_accepts_blank_answers(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "a.jsonl",
        [
            {"run_id": "sys-a", "qid": "q1", "query": "question", "answer_text": ""},
            {"run_id": "sys-a", "qid": "q2", "query": "question", "answer": []},
            {"run_id": "sys-a", "qid": "q3", "query": "question", "answer": None},
        ],
    )

    answer_set = load_answer_set(path)

    assert answer_set.rows_by_qid["q1"].answer_text == ""
    assert answer_set.rows_by_qid["q2"].answer_text == ""
    assert answer_set.rows_by_qid["q3"].answer_text == ""


def test_arena_prompt_uses_answer_text_without_citations_or_references(tmp_path: Path) -> None:
    left = _write(
        tmp_path / "a.jsonl",
        [
            {
                "metadata": {"run_id": "sys-a", "narrative_id": "q1", "narrative": "question"},
                "answer": [{"text": "Left sentence.", "citations": ["DO_NOT_RENDER_CITATION"]}],
                "references": ["DO_NOT_RENDER_REFERENCE"],
            }
        ],
    )
    right = _write(
        tmp_path / "b.jsonl",
        [
            {
                "metadata": {"run_id": "sys-b", "narrative_id": "q1", "narrative": "question"},
                "answer": [{"text": "Right sentence.", "citations": ["DO_NOT_RENDER_CITATION"]}],
                "references": ["DO_NOT_RENDER_REFERENCE"],
            }
        ],
    )

    task = iter_arena_tasks(load_answer_sets([left, right]), seed=13)[0]

    assert "Left sentence." in task["instruction"]
    assert "Right sentence." in task["instruction"]
    assert "DO_NOT_RENDER_CITATION" not in task["instruction"]
    assert "DO_NOT_RENDER_REFERENCE" not in task["instruction"]
    assert task["metadata"]["prompt"] == "PAIRWISE_ANSWER_COMPARISON_NAIVE"


def test_arena_prompt_default_is_final_naive_prompt() -> None:
    prompt = render_arena_prompt(query="q", answer_a="a", answer_b="b")

    assert "[The Start of User's Question]\nq\n[The End of User's Question]" in prompt
    assert "Rubric" not in prompt


def test_iter_arena_tasks_can_use_topic_rubric_prompt(tmp_path: Path) -> None:
    left = _write(tmp_path / "a.jsonl", [{"run_id": "sys-a", "qid": "q1", "query": "q", "answer_text": "a"}])
    right = _write(tmp_path / "b.jsonl", [{"run_id": "sys-b", "qid": "q1", "query": "q", "answer_text": "b"}])
    rubrics = _write(
        tmp_path / "rubric.jsonl",
        [
            {
                "qid": "q1",
                "status": "completed",
                "criteria": [
                    {"text": "Important criterion", "tier": "mandatory", "weight": 5, "type": "explicit"},
                    {"text": "Tie breaker", "tier": "optional", "weight": 2, "type": "synthesis"},
                ],
            }
        ],
    )

    task = iter_arena_tasks(
        load_answer_sets([left, right]),
        seed=13,
        rubrics_by_qid=load_rubrics(rubrics),
        rubrics_source=str(rubrics),
    )[0]

    assert "Rubric" in task["instruction"]
    assert "1. [tier=mandatory, weight=5, criterion_type=explicit] Important criterion" in task["instruction"]
    assert "2. [tier=optional, weight=2, criterion_type=synthesis] Tie breaker" in task["instruction"]
    assert task["metadata"]["rubrics"] is True
    assert task["metadata"]["rubric_criteria_count"] == 2
    assert task["metadata"]["mandatory_criteria_count"] == 1
    assert task["metadata"]["rubrics_source"] == str(rubrics)
    assert task["metadata"]["prompt"] == "PAIRWISE_ANSWER_COMPARISON_W_RUBRICS"


def test_iter_arena_tasks_rejects_missing_topic_rubric(tmp_path: Path) -> None:
    left = _write(tmp_path / "a.jsonl", [{"run_id": "sys-a", "qid": "q1", "query": "q", "answer_text": "a"}])
    right = _write(tmp_path / "b.jsonl", [{"run_id": "sys-b", "qid": "q1", "query": "q", "answer_text": "b"}])
    rubrics = _write(
        tmp_path / "rubric.jsonl",
        [{"qid": "q2", "status": "completed", "criteria": [{"text": "Other criterion"}]}],
    )

    with pytest.raises(ValueError, match="missing rubrics"):
        iter_arena_tasks(load_answer_sets([left, right]), seed=13, rubrics_by_qid=load_rubrics(rubrics))


def test_pairing_uses_shared_qids_and_reports_coverage(tmp_path: Path) -> None:
    left = _write(
        tmp_path / "a.jsonl",
        [
            {"run_id": "sys-a", "qid": "q1", "query": "one", "answer_text": "a1"},
            {"run_id": "sys-a", "qid": "q2", "query": "two", "answer_text": "a2"},
        ],
    )
    right = _write(
        tmp_path / "b.jsonl",
        [{"run_id": "sys-b", "qid": "q1", "query": "one", "answer_text": "b1"}],
    )
    answer_sets = load_answer_sets([left, right])

    tasks = iter_arena_tasks(answer_sets, seed=13)
    coverage = coverage_rows(answer_sets)

    assert [task["metadata"]["qid"] for task in tasks] == ["q1"]
    assert "rubrics" not in tasks[0]["metadata"]
    assert coverage[0]["shared_topics"] == 1
    assert coverage[0]["run_a_only_topics"] == 1
    assert coverage[0]["run_b_only_topics"] == 0


def test_load_answer_set_rejects_duplicate_qids(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "a.jsonl",
        [
            {"run_id": "sys-a", "qid": "q1", "query": "q", "answer_text": "a"},
            {"run_id": "sys-a", "qid": "q1", "query": "q", "answer_text": "b"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate qid"):
        load_answer_set(path)


def test_load_answer_set_rejects_inconsistent_run_ids(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "a.jsonl",
        [
            {"run_id": "sys-a", "qid": "q1", "query": "q", "answer_text": "a"},
            {"run_id": "sys-b", "qid": "q2", "query": "q", "answer_text": "b"},
        ],
    )

    with pytest.raises(ValueError, match="inconsistent run_id"):
        load_answer_set(path)


def test_iter_arena_tasks_rejects_query_mismatch(tmp_path: Path) -> None:
    left = _write(tmp_path / "a.jsonl", [{"run_id": "sys-a", "qid": "q1", "query": "left", "answer_text": "a"}])
    right = _write(tmp_path / "b.jsonl", [{"run_id": "sys-b", "qid": "q1", "query": "right", "answer_text": "b"}])

    with pytest.raises(ValueError, match="query mismatch"):
        iter_arena_tasks(load_answer_sets([left, right]), seed=13)


def test_assistant_order_is_deterministic_and_recorded(tmp_path: Path) -> None:
    left = _write(tmp_path / "a.jsonl", [{"run_id": "sys-a", "qid": "q1", "query": "q", "answer_text": "a"}])
    right = _write(tmp_path / "b.jsonl", [{"run_id": "sys-b", "qid": "q1", "query": "q", "answer_text": "b"}])

    task = iter_arena_tasks(load_answer_sets([left, right]), seed=13)[0]

    assert assistant_order(13, "sys-a", "sys-b", "q1") == (
        task["metadata"]["assistant_a_run_id"],
        task["metadata"]["assistant_b_run_id"],
    )


def test_sampled_pair_qids_is_deterministic_and_pair_order_invariant() -> None:
    qids = ["q1", "q2", "q3", "q4"]

    first = sampled_pair_qids(qids, run_a="sys-a", run_b="sys-b", k=2, seed=7)
    second = sampled_pair_qids(qids, run_a="sys-b", run_b="sys-a", k=2, seed=7)

    assert first == second
    assert len(first) == 2
    assert first == sorted(first)
    assert set(first) <= set(qids)


def test_iter_arena_tasks_samples_shared_qids_per_pair(tmp_path: Path) -> None:
    left = _write(
        tmp_path / "a.jsonl",
        [
            {"run_id": "sys-a", "qid": "q1", "query": "q1", "answer_text": "a1"},
            {"run_id": "sys-a", "qid": "q2", "query": "q2", "answer_text": "a2"},
            {"run_id": "sys-a", "qid": "q3", "query": "q3", "answer_text": "a3"},
        ],
    )
    right = _write(
        tmp_path / "b.jsonl",
        [
            {"run_id": "sys-b", "qid": "q1", "query": "q1", "answer_text": "b1"},
            {"run_id": "sys-b", "qid": "q2", "query": "q2", "answer_text": "b2"},
            {"run_id": "sys-b", "qid": "q3", "query": "q3", "answer_text": "b3"},
        ],
    )

    tasks = iter_arena_tasks(load_answer_sets([left, right]), seed=13, sample_topics_per_pair=2, sampling_seed=99)

    assert len(tasks) == 2
    assert [task["metadata"]["qid"] for task in tasks] == sampled_pair_qids(
        ["q1", "q2", "q3"],
        run_a="sys-a",
        run_b="sys-b",
        k=2,
        seed=99,
    )


def test_sampled_topic_pairs_supports_degree_one_matching() -> None:
    pairs = [(f"s{i}", f"s{j}") for i in range(5) for j in range(i + 1, 5)]

    sampled = sampled_topic_pairs(pairs, qid="q1", battles_per_topic=3, seed=17)

    assert len(sampled) == 3
    assert len({run_id for pair in sampled for run_id in pair}) == 5


def test_iter_arena_tasks_samples_battles_per_system_per_topic(tmp_path: Path) -> None:
    paths = []
    for run_id in ["sys-a", "sys-b", "sys-c", "sys-d"]:
        paths.append(
            _write(
                tmp_path / f"{run_id}.jsonl",
                [
                    {"run_id": run_id, "qid": "q1", "query": "q1", "answer_text": run_id},
                    {"run_id": run_id, "qid": "q2", "query": "q2", "answer_text": run_id},
                ],
            )
        )

    tasks = iter_arena_tasks(
        load_answer_sets(paths),
        seed=13,
        sample_battles_per_system_per_topic=1,
        sampling_seed=99,
    )

    assert battles_for_system_degree(4, 1) == 2
    assert len(tasks) == 4
    assert {task["metadata"]["qid"] for task in tasks} == {"q1", "q2"}
    assert all(task["metadata"]["pair"][0] != task["metadata"]["pair"][1] for task in tasks)


def test_parse_verdict_requires_single_label() -> None:
    assert parse_verdict("[[A]]") == "A"
    assert parse_verdict(" [[B]]\n") == "B"
    assert parse_verdict("[[Tie]]") == "Tie"
    assert parse_verdict("[[Tie (Both Bad)]]") == "Tie (Both Bad)"
    assert parse_verdict("I choose [[A]]") is None
    assert parse_verdict("[[C]]") is None
    assert parse_verdict("[[Tie (bothbad)]]") is None
    assert parse_verdict("[[Both Good]]") is None
    assert parse_verdict("[[Both Bad]]") is None


def test_tie_verdicts_count_as_ties() -> None:
    judgments = [
        {"status": "completed", "pair": ["a", "b"], "judge_verdict": "Tie", "preferred_run_id": None},
        {"status": "completed", "pair": ["a", "b"], "judge_verdict": "Tie (Both Bad)", "preferred_run_id": None},
    ]
    coverage = [{"run_a": "a", "run_b": "b", "shared_topics": 2}]

    pairwise = pairwise_rows(judgments, coverage)
    scores = fit_arena_ratings(judgments, ["a", "b"])
    leaderboard = leaderboard_rows(judgments, ["a", "b"])

    assert pairwise[0]["valid_judgments"] == 2
    assert pairwise[0]["ties"] == 2
    assert pairwise[0]["run_a_preference_rate"] == 0.5
    assert scores == {"a": 1000.0, "b": 1000.0}
    assert {row["run_id"]: row["ties"] for row in leaderboard} == {"a": 2, "b": 2}


def test_pairwise_and_arena_ranking() -> None:
    judgments = [
        {"status": "completed", "pair": ["a", "b"], "judge_verdict": "A", "preferred_run_id": "a"},
        {"status": "completed", "pair": ["a", "b"], "judge_verdict": "Tie", "preferred_run_id": None},
        {"status": "completed", "pair": ["a", "c"], "judge_verdict": "A", "preferred_run_id": "a"},
        {"status": "completed", "pair": ["b", "c"], "judge_verdict": "A", "preferred_run_id": "b"},
    ]
    coverage = [
        {"run_a": "a", "run_b": "b", "shared_topics": 2},
        {"run_a": "a", "run_b": "c", "shared_topics": 1},
        {"run_a": "b", "run_b": "c", "shared_topics": 1},
    ]

    pairwise = pairwise_rows(judgments, coverage)
    scores = fit_arena_ratings(judgments, ["a", "b", "c"])
    leaderboard = leaderboard_rows(judgments, ["a", "b", "c"])

    assert pairwise[0]["run_a_preference_rate"] == 0.75
    assert scores["a"] > scores["b"] > scores["c"]
    assert round(sum(scores.values()) / len(scores), 6) == 1000.0
    assert [row["run_id"] for row in leaderboard] == ["a", "b", "c"]
    assert "arena_score" in leaderboard[0]


def test_materialize_arena_cli(tmp_path: Path, monkeypatch) -> None:
    left = _write(tmp_path / "a.jsonl", [{"run_id": "sys-a", "qid": "q1", "query": "q", "answer_text": "a"}])
    right = _write(tmp_path / "b.jsonl", [{"run_id": "sys-b", "qid": "q1", "query": "q", "answer_text": "b"}])
    output = tmp_path / "tasks.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ragdoll",
            "materialize",
            "arena",
            "--answers",
            str(left),
            "--answers",
            str(right),
            "--output-file",
            str(output),
        ],
    )

    main()

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["evaluator"] == "arena"
    assert row["metadata"]["pair"] == ["sys-a", "sys-b"]


def test_materialize_arena_cli_accepts_answers_dir(tmp_path: Path, monkeypatch) -> None:
    answers_dir = tmp_path / "answers"
    answers_dir.mkdir()
    _write(answers_dir / "b.jsonl", [{"run_id": "sys-b", "qid": "q1", "query": "q", "answer_text": "b"}])
    _write(answers_dir / "a.jsonl", [{"run_id": "sys-a", "qid": "q1", "query": "q", "answer_text": "a"}])
    output = tmp_path / "tasks.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ragdoll",
            "materialize",
            "arena",
            "--answers-dir",
            str(answers_dir),
            "--output-file",
            str(output),
        ],
    )

    main()

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["metadata"]["pair"] == ["sys-a", "sys-b"]


def test_materialize_arena_cli_samples_topics_per_pair(tmp_path: Path, monkeypatch) -> None:
    left = _write(
        tmp_path / "a.jsonl",
        [
            {"run_id": "sys-a", "qid": "q1", "query": "q1", "answer_text": "a1"},
            {"run_id": "sys-a", "qid": "q2", "query": "q2", "answer_text": "a2"},
            {"run_id": "sys-a", "qid": "q3", "query": "q3", "answer_text": "a3"},
        ],
    )
    right = _write(
        tmp_path / "b.jsonl",
        [
            {"run_id": "sys-b", "qid": "q1", "query": "q1", "answer_text": "b1"},
            {"run_id": "sys-b", "qid": "q2", "query": "q2", "answer_text": "b2"},
            {"run_id": "sys-b", "qid": "q3", "query": "q3", "answer_text": "b3"},
        ],
    )
    output = tmp_path / "tasks.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ragdoll",
            "materialize",
            "arena",
            "--answers",
            str(left),
            "--answers",
            str(right),
            "--output-file",
            str(output),
            "--sample-topics-per-pair",
            "1",
            "--sampling-seed",
            "5",
        ],
    )

    main()

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["metadata"]["qid"] in {"q1", "q2", "q3"}


def test_materialize_arena_cli_samples_battles_per_system_per_topic(tmp_path: Path, monkeypatch) -> None:
    paths = []
    for run_id in ["sys-a", "sys-b", "sys-c", "sys-d"]:
        paths.append(
            _write(
                tmp_path / f"{run_id}.jsonl",
                [
                    {"run_id": run_id, "qid": "q1", "query": "q1", "answer_text": run_id},
                    {"run_id": run_id, "qid": "q2", "query": "q2", "answer_text": run_id},
                ],
            )
        )
    output = tmp_path / "tasks.jsonl"
    argv = ["ragdoll", "materialize", "arena", "--output-file", str(output)]
    for path in paths:
        argv.extend(["--answers", str(path)])
    argv.extend(["--sample-battles-per-system-per-topic", "1", "--sampling-seed", "5"])
    monkeypatch.setattr(sys, "argv", argv)

    main()

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4


def test_materialize_arena_cli_accepts_rubric_file(tmp_path: Path, monkeypatch) -> None:
    left = _write(tmp_path / "a.jsonl", [{"run_id": "sys-a", "qid": "q1", "query": "q", "answer_text": "a"}])
    right = _write(tmp_path / "b.jsonl", [{"run_id": "sys-b", "qid": "q1", "query": "q", "answer_text": "b"}])
    rubrics = _write(
        tmp_path / "rubric.jsonl",
        [
            {
                "qid": "q1",
                "status": "completed",
                "criteria": [{"text": "Important criterion", "tier": "mandatory", "weight": 5}],
            }
        ],
    )
    output = tmp_path / "tasks.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ragdoll",
            "materialize",
            "arena",
            "--answers",
            str(left),
            "--answers",
            str(right),
            "--output-file",
            str(output),
            "--rubric-file",
            str(rubrics),
        ],
    )

    main()

    row = json.loads(output.read_text(encoding="utf-8"))
    assert "Important criterion" in row["instruction"]
    assert row["metadata"]["rubrics"] is True
    assert row["metadata"]["prompt"] == "PAIRWISE_ANSWER_COMPARISON_W_RUBRICS"
    assert row["metadata"]["rubric_criteria_count"] == 1


def test_arena_compare_all_cli_with_fake_pi(tmp_path: Path, monkeypatch) -> None:
    fake_pi = tmp_path / "fake_pi.py"
    fake_pi.write_text(
        """#!/usr/bin/env python3
import json
print(json.dumps({"type":"message_end","message":{"role":"assistant","content":"[[A]]"}}))
""",
        encoding="utf-8",
    )
    fake_pi.chmod(0o755)
    left = _write(
        tmp_path / "a.jsonl",
        [
            {"run_id": "sys-a", "qid": "q1", "query": "q1", "answer_text": "a1"},
            {"run_id": "sys-a", "qid": "q2", "query": "q2", "answer_text": "a2"},
        ],
    )
    right = _write(
        tmp_path / "b.jsonl",
        [
            {"run_id": "sys-b", "qid": "q1", "query": "q1", "answer_text": "b1"},
            {"run_id": "sys-b", "qid": "q2", "query": "q2", "answer_text": "b2"},
        ],
    )
    output_dir = tmp_path / "arena"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ragdoll",
            "arena",
            "compare-all",
            "--answers",
            str(left),
            "--answers",
            str(right),
            "--output-dir",
            str(output_dir),
            "--agent-binary",
            str(fake_pi),
            "--agent-state-dir",
            str(tmp_path / "missing"),
            "--model",
            "judge-model",
            "--thinking",
            "medium",
            "--max-concurrency",
            "1",
            "--no-cache",
            "--overwrite",
        ],
    )

    main()

    judgments = [json.loads(line) for line in (output_dir / "judgments.jsonl").read_text(encoding="utf-8").splitlines()]
    leaderboard = list(csv.DictReader((output_dir / "leaderboard.csv").open(encoding="utf-8")))
    assert len(judgments) == 2
    assert all(row["judge_verdict"] == "A" for row in judgments)
    assert all(row["model"] == "judge-model" for row in judgments)
    assert len(leaderboard) == 2
    assert (output_dir / "tasks.jsonl").exists()
    tasks = [json.loads(line) for line in (output_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all("Rubric" not in task["instruction"] for task in tasks)
    assert all(task["metadata"]["prompt"] == "PAIRWISE_ANSWER_COMPARISON_NAIVE" for task in tasks)
    assert (output_dir / "pairwise.csv").exists()
    assert (output_dir / "coverage.csv").exists()


def test_arena_compare_all_cli_with_rubric_file_and_fake_pi(tmp_path: Path, monkeypatch) -> None:
    fake_pi = tmp_path / "fake_pi.py"
    fake_pi.write_text(
        """#!/usr/bin/env python3
import json
print(json.dumps({"type":"message_end","message":{"role":"assistant","content":"[[Tie]]"}}))
""",
        encoding="utf-8",
    )
    fake_pi.chmod(0o755)
    left = _write(tmp_path / "a.jsonl", [{"run_id": "sys-a", "qid": "q1", "query": "q1", "answer_text": "a1"}])
    right = _write(tmp_path / "b.jsonl", [{"run_id": "sys-b", "qid": "q1", "query": "q1", "answer_text": "b1"}])
    rubrics = _write(
        tmp_path / "rubric.jsonl",
        [{"qid": "q1", "status": "completed", "criteria": [{"text": "Important criterion", "tier": "mandatory", "weight": 5}]}],
    )
    output_dir = tmp_path / "arena"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ragdoll",
            "arena",
            "compare-all",
            "--answers",
            str(left),
            "--answers",
            str(right),
            "--output-dir",
            str(output_dir),
            "--agent-binary",
            str(fake_pi),
            "--agent-state-dir",
            str(tmp_path / "missing"),
            "--rubric-file",
            str(rubrics),
            "--max-concurrency",
            "1",
            "--no-cache",
            "--overwrite",
        ],
    )

    main()

    row = json.loads((output_dir / "judgments.jsonl").read_text(encoding="utf-8"))
    task = json.loads((output_dir / "tasks.jsonl").read_text(encoding="utf-8"))
    assert row["judge_verdict"] == "Tie"
    assert "Important criterion" in task["instruction"]
    assert task["metadata"]["prompt"] == "PAIRWISE_ANSWER_COMPARISON_W_RUBRICS"


def test_arena_compare_all_cli_accepts_answers_dir(tmp_path: Path, monkeypatch) -> None:
    fake_pi = tmp_path / "fake_pi.py"
    fake_pi.write_text(
        """#!/usr/bin/env python3
import json
print(json.dumps({"type":"message_end","message":{"role":"assistant","content":"[[Tie]]"}}))
""",
        encoding="utf-8",
    )
    fake_pi.chmod(0o755)
    answers_dir = tmp_path / "answers"
    answers_dir.mkdir()
    _write(answers_dir / "a.jsonl", [{"run_id": "sys-a", "qid": "q1", "query": "q1", "answer_text": "a1"}])
    _write(answers_dir / "b.jsonl", [{"run_id": "sys-b", "qid": "q1", "query": "q1", "answer_text": "b1"}])
    output_dir = tmp_path / "arena"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ragdoll",
            "arena",
            "compare-all",
            "--answers-dir",
            str(answers_dir),
            "--output-dir",
            str(output_dir),
            "--agent-binary",
            str(fake_pi),
            "--agent-state-dir",
            str(tmp_path / "missing"),
            "--max-concurrency",
            "1",
            "--no-cache",
            "--overwrite",
        ],
    )

    main()

    row = json.loads((output_dir / "judgments.jsonl").read_text(encoding="utf-8"))
    assert row["judge_verdict"] == "Tie"
