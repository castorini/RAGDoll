import json
from pathlib import Path

from ragdoll.config import MaterializeAssignInputsConfig
from ragdoll.trec_io import (
    iter_assign_payloads_from_files,
    load_nuggets_by_qid,
    materialize_assign_inputs,
    read_trec_answers,
)


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_load_nuggets_by_qid(tmp_path: Path) -> None:
    nuggets = _write(tmp_path / "n.jsonl", [{"qid": "q1", "nuggets": [{"text": "a"}]}])
    indexed = load_nuggets_by_qid(nuggets)
    assert set(indexed) == {"q1"}


def test_iter_assign_payloads_joins_and_skips_missing(tmp_path: Path) -> None:
    answers = _write(
        tmp_path / "a.jsonl",
        [
            {"run_id": "rA", "qid": "q1", "answer_text": "x"},
            {"run_id": "rA", "qid": "q9", "answer_text": "orphan"},
        ],
    )
    nuggets = _write(tmp_path / "n.jsonl", [{"qid": "q1", "nuggets": [{"text": "a"}]}])
    payloads = list(iter_assign_payloads_from_files(answers, nuggets))
    assert len(payloads) == 1
    assert payloads[0]["answer_record"]["qid"] == "q1"
    assert payloads[0]["nugget_record"]["nuggets"] == [{"text": "a"}]


def test_materialize_assign_inputs(tmp_path: Path) -> None:
    answers = _write(tmp_path / "a.jsonl", [{"run_id": "rA", "qid": "q1", "answer_text": "x"}])
    nuggets = _write(tmp_path / "n.jsonl", [{"qid": "q1", "nuggets": [{"text": "a"}]}])
    out = tmp_path / "joined.jsonl"
    materialize_assign_inputs(
        MaterializeAssignInputsConfig(answers_file=answers, nuggets_file=nuggets, output_file=out)
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0].keys() >= {"answer_record", "nugget_record"}


def test_read_trec_answers_normalizes_response(tmp_path: Path) -> None:
    run = _write(
        tmp_path / "run.jsonl",
        [{"run_id": "team", "topic_id": "q1", "topic": "what", "response": [{"text": "hello", "citations": [0]}]}],
    )
    rows = list(read_trec_answers(run))
    assert rows[0]["run_id"] == "team"
    assert rows[0]["qid"] == "q1"
    assert rows[0]["answer_text"] == "hello"
    assert rows[0]["answer"][0]["citations"] == [0]


def test_read_trec_answers_explicit_run_id_overrides_row_value(tmp_path: Path) -> None:
    run = _write(
        tmp_path / "run.jsonl",
        [{"run_id": "short", "topic_id": "q1", "topic": "what", "answer": [{"text": "hello"}]}],
    )

    rows = list(read_trec_answers(run, run_id="org.official-run"))

    assert rows[0]["run_id"] == "org.official-run"
