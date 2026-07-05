from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from ragdoll.arena.prompts import render_arena_prompt
from ragdoll.jsonl import read_jsonl


@dataclass(frozen=True)
class AnswerRecord:
    run_id: str
    qid: str
    query: str
    answer_text: str
    source: dict[str, Any]


@dataclass(frozen=True)
class AnswerSet:
    run_id: str
    path: Path
    rows_by_qid: dict[str, AnswerRecord]

    @property
    def qids(self) -> set[str]:
        return set(self.rows_by_qid)


@dataclass(frozen=True)
class NuggetRecord:
    qid: str
    nuggets: list[dict[str, str]]
    source: dict[str, Any]


@dataclass(frozen=True)
class RubricRecord:
    qid: str
    query: str
    criteria: list[dict[str, Any]]
    source: dict[str, Any]


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("metadata") if isinstance(row.get("metadata"), dict) else {}


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int):
            return str(value)
    return ""


def qid_of(row: dict[str, Any]) -> str:
    metadata = _metadata(row)
    return _first_text(row.get("qid"), row.get("topic_id"), row.get("query_id"), metadata.get("narrative_id"))


def run_id_of(row: dict[str, Any]) -> str:
    metadata = _metadata(row)
    return _first_text(row.get("run_id"), metadata.get("run_id"))


def query_of(row: dict[str, Any]) -> str:
    metadata = _metadata(row)
    return _first_text(row.get("query"), row.get("topic"), metadata.get("narrative"), metadata.get("title"))


def _answer_sentence_text(answer: Any) -> list[str]:
    if isinstance(answer, str):
        return [answer.strip()] if answer.strip() else []
    if not isinstance(answer, list):
        return []
    texts: list[str] = []
    for item in answer:
        if isinstance(item, dict):
            text = item.get("text")
        else:
            text = item
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts


def answer_text_of(row: dict[str, Any]) -> str | None:
    answer_text = row.get("answer_text")
    if isinstance(answer_text, str):
        return answer_text.strip()
    if "answer" in row:
        return " ".join(_answer_sentence_text(row.get("answer"))).strip()
    if "responses" in row:
        return " ".join(_answer_sentence_text(row.get("responses"))).strip()
    return None


def load_answer_set(path: Path) -> AnswerSet:
    rows_by_qid: dict[str, AnswerRecord] = {}
    run_id: str | None = None
    for row in read_jsonl(path):
        row_run_id = run_id_of(row)
        qid = qid_of(row)
        query = query_of(row)
        answer_text = answer_text_of(row)
        if not row_run_id:
            raise ValueError(f"{path}: missing run_id or metadata.run_id")
        if not qid:
            raise ValueError(f"{path}: missing qid/topic_id/metadata.narrative_id")
        if not query:
            raise ValueError(f"{path}:{qid}: missing query/topic/metadata narrative/title")
        if answer_text is None:
            raise ValueError(f"{path}:{qid}: missing answer_text or answer[].text")
        if run_id is None:
            run_id = row_run_id
        elif row_run_id != run_id:
            raise ValueError(f"{path}: inconsistent run_id values: {run_id!r} and {row_run_id!r}")
        if qid in rows_by_qid:
            raise ValueError(f"{path}: duplicate qid {qid!r}")
        rows_by_qid[qid] = AnswerRecord(
            run_id=row_run_id,
            qid=qid,
            query=query,
            answer_text=answer_text,
            source=row,
        )
    if run_id is None:
        raise ValueError(f"{path}: no answer rows found")
    return AnswerSet(run_id=run_id, path=path, rows_by_qid=rows_by_qid)


def load_answer_sets(paths: list[Path]) -> list[AnswerSet]:
    sets = [load_answer_set(path) for path in paths]
    seen: set[str] = set()
    for answer_set in sets:
        if answer_set.run_id in seen:
            raise ValueError(f"duplicate run_id across answer files: {answer_set.run_id!r}")
        seen.add(answer_set.run_id)
    return sets


def _nugget_qid(row: dict[str, Any]) -> str:
    metadata = _metadata(row)
    return _first_text(row.get("qid"), row.get("topic_id"), row.get("query_id"), metadata.get("narrative_id"))


def _rubric_qid(row: dict[str, Any]) -> str:
    metadata = _metadata(row)
    return _first_text(
        row.get("qid"),
        row.get("topic_id"),
        row.get("query_id"),
        row.get("task_id"),
        metadata.get("narrative_id"),
    )


def _normalize_nugget(item: Any) -> dict[str, str] | None:
    if isinstance(item, str):
        text = item.strip()
        return {"text": text, "importance": ""} if text else None
    if not isinstance(item, dict):
        return None
    text = str(item.get("text", "")).strip()
    if not text:
        return None
    normalized = {"text": text, "importance": str(item.get("importance", "")).strip()}
    mapped_sub_narrative = str(item.get("mapped_sub_narrative", "")).strip()
    if mapped_sub_narrative:
        normalized["mapped_sub_narrative"] = mapped_sub_narrative
    return normalized


def load_nuggets(path: Path) -> dict[str, NuggetRecord]:
    records: dict[str, NuggetRecord] = {}
    for row in read_jsonl(path):
        qid = _nugget_qid(row)
        if not qid:
            raise ValueError(f"{path}: missing qid/topic_id for nugget row")
        if qid in records:
            raise ValueError(f"{path}: duplicate nugget qid {qid!r}")
        nuggets_value = row.get("nuggets")
        if not isinstance(nuggets_value, list):
            raise ValueError(f"{path}:{qid}: missing nuggets list")
        nuggets = [nugget for nugget in (_normalize_nugget(item) for item in nuggets_value) if nugget is not None]
        if not nuggets:
            raise ValueError(f"{path}:{qid}: no usable nuggets")
        records[qid] = NuggetRecord(qid=qid, nuggets=nuggets, source=row)
    if not records:
        raise ValueError(f"{path}: no nugget rows found")
    return records


def format_nuggets_for_prompt(nugget_record: NuggetRecord) -> str:
    lines: list[str] = []
    for index, nugget in enumerate(nugget_record.nuggets, start=1):
        importance = nugget.get("importance") or "unlabeled"
        lines.append(f"{index}. [{importance}] {nugget['text']}")
    return "\n".join(lines)


def _normalize_rubric_criterion(item: Any) -> dict[str, Any] | None:
    if isinstance(item, str):
        text = item.strip()
        return {"text": text} if text else None
    if not isinstance(item, dict):
        return None
    text = str(item.get("text", "")).strip()
    if not text:
        return None
    normalized: dict[str, Any] = {"text": text}
    for field in ("tier", "type"):
        value = item.get(field)
        if value is not None and str(value).strip():
            normalized[field] = str(value).strip()
    if item.get("source_nugget") is not None and str(item.get("source_nugget")).strip():
        normalized["source_nugget"] = str(item.get("source_nugget")).strip()
    if item.get("weight") is not None:
        normalized["weight"] = item.get("weight")
    return normalized


def load_rubrics(path: Path) -> dict[str, RubricRecord]:
    records: dict[str, RubricRecord] = {}
    for row in read_jsonl(path):
        qid = _rubric_qid(row)
        if not qid:
            raise ValueError(f"{path}: missing qid/topic_id/task_id for rubric row")
        if qid in records:
            raise ValueError(f"{path}: duplicate rubric qid {qid!r}")
        status = row.get("status")
        if status is not None and status != "completed":
            raise ValueError(f"{path}:{qid}: rubric row status is {status!r}")
        criteria_value = row.get("criteria")
        if not isinstance(criteria_value, list):
            raise ValueError(f"{path}:{qid}: missing criteria list")
        criteria = [
            criterion
            for criterion in (_normalize_rubric_criterion(item) for item in criteria_value)
            if criterion is not None
        ]
        if not criteria:
            raise ValueError(f"{path}:{qid}: no usable rubric criteria")
        records[qid] = RubricRecord(qid=qid, query=query_of(row), criteria=criteria, source=row)
    if not records:
        raise ValueError(f"{path}: no rubric rows found")
    return records


def format_rubric_for_prompt(rubric_record: RubricRecord) -> str:
    lines: list[str] = []
    for index, criterion in enumerate(rubric_record.criteria, start=1):
        labels = []
        tier = str(criterion.get("tier", "")).strip()
        weight = criterion.get("weight")
        criterion_type = str(criterion.get("type", "")).strip()
        if tier:
            labels.append(tier)
        if weight is not None:
            labels.append(f"weight={weight}")
        if criterion_type:
            labels.append(criterion_type)
        label = f" [{', '.join(labels)}]" if labels else ""
        lines.append(f"{index}.{label} {criterion['text']}")
    return "\n".join(lines)


def assistant_order(seed: int, run_a: str, run_b: str, qid: str) -> tuple[str, str]:
    first, second = sorted((run_a, run_b))
    key = f"{seed}\x00{first}\x00{second}\x00{qid}".encode()
    bit = hashlib.sha256(key).digest()[0] % 2
    return (run_a, run_b) if bit == 0 else (run_b, run_a)


def _task_id(run_a: str, run_b: str, qid: str) -> str:
    return f"arena:{run_a}:{run_b}:{qid}"


def sampled_pair_qids(qids: list[str], *, run_a: str, run_b: str, k: int | None, seed: int) -> list[str]:
    if k is None or len(qids) <= k:
        return qids
    if k <= 0:
        raise ValueError("sample_topics_per_pair must be positive")
    first, second = sorted((run_a, run_b))
    key = f"{seed}\x00{first}\x00{second}".encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
    return sorted(rng.sample(qids, k))


def battles_for_system_degree(n_systems: int, battles_per_system: float | None) -> int | None:
    if battles_per_system is None:
        return None
    if battles_per_system <= 0:
        raise ValueError("sample_battles_per_system_per_topic must be positive")
    return math.ceil(n_systems * battles_per_system / 2)


def sampled_topic_pairs(
    pairs: list[tuple[str, str]],
    *,
    qid: str,
    battles_per_topic: int | None,
    seed: int,
) -> list[tuple[str, str]]:
    if battles_per_topic is None or len(pairs) <= battles_per_topic:
        return pairs
    if battles_per_topic <= 0:
        raise ValueError("sample_battles_per_topic must be positive")

    available = set(pairs)
    systems = sorted({run_id for pair in pairs for run_id in pair})
    key = f"{seed}\x00{qid}\x00{battles_per_topic}".encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
    rng.shuffle(systems)

    sampled: set[tuple[str, str]] = set()
    if battles_per_topic >= len(systems):
        for index, run_a in enumerate(systems):
            run_b = systems[(index + 1) % len(systems)]
            sampled.add(tuple(sorted((run_a, run_b))))
    elif battles_per_topic >= math.ceil(len(systems) / 2):
        for index in range(0, len(systems) - 1, 2):
            sampled.add(tuple(sorted((systems[index], systems[index + 1]))))
        if len(systems) % 2:
            sampled.add(tuple(sorted((systems[-1], systems[0]))))

    sampled &= available
    remaining = [pair for pair in pairs if pair not in sampled]
    needed = battles_per_topic - len(sampled)
    sampled.update(rng.sample(remaining, needed))
    return sorted(sampled)


def _arena_task(
    left: AnswerSet,
    right: AnswerSet,
    qid: str,
    *,
    seed: int,
    rubrics: bool = False,
    nuggets_by_qid: dict[str, NuggetRecord] | None = None,
    nuggets_source: str | None = None,
    rubrics_by_qid: dict[str, RubricRecord] | None = None,
    rubrics_source: str | None = None,
    prompt_variant: str = "default",
) -> dict[str, Any]:
    if nuggets_by_qid is not None and rubrics_by_qid is not None:
        raise ValueError("arena task generation accepts either nuggets_by_qid or rubrics_by_qid, not both")
    left_answer = left.rows_by_qid[qid]
    right_answer = right.rows_by_qid[qid]
    if left_answer.query != right_answer.query:
        raise ValueError(f"query mismatch for qid {qid!r} between {left.run_id!r} and {right.run_id!r}")
    assistant_a_run_id, assistant_b_run_id = assistant_order(seed, left.run_id, right.run_id, qid)
    by_run_id = {left.run_id: left_answer, right.run_id: right_answer}
    answer_a = by_run_id[assistant_a_run_id]
    answer_b = by_run_id[assistant_b_run_id]
    metadata = {
        "pair": [left.run_id, right.run_id],
        "qid": qid,
        "query": left_answer.query,
        "assistant_a_run_id": assistant_a_run_id,
        "assistant_b_run_id": assistant_b_run_id,
    }
    if prompt_variant != "default":
        metadata["prompt_variant"] = prompt_variant
    if rubrics:
        metadata["rubrics"] = True
    nuggets_prompt = None
    rubric_prompt = None
    if nuggets_by_qid is not None:
        nugget_record = nuggets_by_qid.get(qid)
        if nugget_record is None:
            raise ValueError(f"missing nuggets for qid {qid!r}")
        nuggets_prompt = format_nuggets_for_prompt(nugget_record)
        metadata.update(
            {
                "nuggets": True,
                "nuggets_qid": qid,
                "nuggets_count": len(nugget_record.nuggets),
                "vital_nuggets_count": sum(
                    1 for nugget in nugget_record.nuggets if nugget.get("importance", "").lower() == "vital"
                ),
            }
        )
        if nuggets_source is not None:
            metadata["nuggets_source"] = nuggets_source
    if rubrics_by_qid is not None:
        rubric_record = rubrics_by_qid.get(qid)
        if rubric_record is None:
            raise ValueError(f"missing rubrics for qid {qid!r}")
        rubric_prompt = format_rubric_for_prompt(rubric_record)
        metadata.update(
            {
                "rubrics": True,
                "rubric_qid": qid,
                "rubric_criteria_count": len(rubric_record.criteria),
                "mandatory_criteria_count": sum(
                    1 for criterion in rubric_record.criteria if str(criterion.get("tier", "")).lower() == "mandatory"
                ),
            }
        )
        if rubrics_source is not None:
            metadata["rubrics_source"] = rubrics_source
    return {
        "task_id": _task_id(left.run_id, right.run_id, qid),
        "evaluator": "arena",
        "system_prompt": "",
        "instruction": render_arena_prompt(
            query=left_answer.query,
            answer_a=answer_a.answer_text,
            answer_b=answer_b.answer_text,
            rubrics=rubrics,
            nuggets=nuggets_prompt,
            rubric=rubric_prompt,
            prompt_variant=prompt_variant,
        ),
        "metadata": metadata,
    }


def iter_arena_tasks(
    answer_sets: list[AnswerSet],
    *,
    seed: int,
    sample_topics_per_pair: int | None = None,
    sample_battles_per_topic: int | None = None,
    sample_battles_per_system_per_topic: float | None = None,
    sampling_seed: int | None = None,
    rubrics: bool = False,
    nuggets_by_qid: dict[str, NuggetRecord] | None = None,
    nuggets_source: str | None = None,
    rubrics_by_qid: dict[str, RubricRecord] | None = None,
    rubrics_source: str | None = None,
    prompt_variant: str = "default",
) -> list[dict[str, Any]]:
    if nuggets_by_qid is not None and rubrics_by_qid is not None:
        raise ValueError("arena task generation accepts either nuggets_by_qid or rubrics_by_qid, not both")
    tasks: list[dict[str, Any]] = []
    sample_seed = seed if sampling_seed is None else sampling_seed
    by_run_id = {answer_set.run_id: answer_set for answer_set in answer_sets}
    if sample_battles_per_topic is not None or sample_battles_per_system_per_topic is not None:
        qids = sorted({qid for answer_set in answer_sets for qid in answer_set.qids})
        for qid in qids:
            pairs = [
                (left.run_id, right.run_id)
                for left, right in combinations(answer_sets, 2)
                if qid in left.rows_by_qid and qid in right.rows_by_qid
            ]
            battles_per_topic = sample_battles_per_topic
            if sample_battles_per_system_per_topic is not None:
                n_systems = len({run_id for pair in pairs for run_id in pair})
                battles_per_topic = battles_for_system_degree(n_systems, sample_battles_per_system_per_topic)
            for run_a, run_b in sampled_topic_pairs(
                pairs,
                qid=qid,
                battles_per_topic=battles_per_topic,
                seed=sample_seed,
            ):
                tasks.append(
                    _arena_task(
                        by_run_id[run_a],
                        by_run_id[run_b],
                        qid,
                        seed=seed,
                        rubrics=rubrics,
                        nuggets_by_qid=nuggets_by_qid,
                        nuggets_source=nuggets_source,
                        rubrics_by_qid=rubrics_by_qid,
                        rubrics_source=rubrics_source,
                        prompt_variant=prompt_variant,
                    )
                )
        return tasks

    for left, right in combinations(answer_sets, 2):
        shared_qids = sorted(left.qids & right.qids)
        shared_qids = sampled_pair_qids(
            shared_qids,
            run_a=left.run_id,
            run_b=right.run_id,
            k=sample_topics_per_pair,
            seed=sample_seed,
        )
        for qid in shared_qids:
            tasks.append(
                _arena_task(
                    left,
                    right,
                    qid,
                    seed=seed,
                    rubrics=rubrics,
                    nuggets_by_qid=nuggets_by_qid,
                    nuggets_source=nuggets_source,
                    rubrics_by_qid=rubrics_by_qid,
                    rubrics_source=rubrics_source,
                    prompt_variant=prompt_variant,
                )
            )
    return tasks


def coverage_rows(answer_sets: list[AnswerSet]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in combinations(answer_sets, 2):
        shared = left.qids & right.qids
        rows.append(
            {
                "run_a": left.run_id,
                "run_b": right.run_id,
                "run_a_topics": len(left.qids),
                "run_b_topics": len(right.qids),
                "shared_topics": len(shared),
                "run_a_only_topics": len(left.qids - right.qids),
                "run_b_only_topics": len(right.qids - left.qids),
            }
        )
    return rows
