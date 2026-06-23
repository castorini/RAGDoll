"""Byte-identity provenance for every prompt copied from castorini.

Upstream prompts captured 2026-06-15 at these pinned commits, vendored under
tests/fixtures/upstream/. Those files must stay byte-identical to upstream, so
the capture metadata lives here (and in that dir's README), never inside them:
  castorini/umbrela      e2c4e4126bf7b568d9230008b465a58445cabc30  qrel_zeroshot_{basic,bing}.yaml
  castorini/nuggetizer   d00dc3eb791f4f8f49547ce3ca4f60a63b08cfd7  {creator,scorer,assigner,assigner_2grade}_template.yaml
  lintool/trec2024-rag   89ce8cc5082d38377395d5b1f67cda64540fdcad  support_evaluation_individual_gpt4o.py  (private repo)

Upstream render references:
  umbrela    src/umbrela/prompts/template_loader.py  (PromptTemplate.render)
  nuggetizer src/nuggetizer/prompts/service.py        (create_*_messages)
  support    support_eval/code/..._gpt4o.py           (AbstractEvaluation.__call__)
"""

from __future__ import annotations

import ast
from pathlib import Path
from string import Formatter

import pytest
import yaml

from pi_trec.nuggetizer import (
    NUGGET_ASSIGNER_2GRADE_USER,
    NUGGET_ASSIGNER_SYSTEM,
    NUGGET_ASSIGNER_USER,
    NUGGET_CREATOR_SYSTEM,
    NUGGET_CREATOR_USER,
    NUGGET_SCORER_SYSTEM,
    NUGGET_SCORER_USER,
    create_context,
    direct_assign_inputs,
    iter_assign_tasks,
    iter_create_tasks,
    render_assign_prompt,
    render_create_prompt,
    render_score_prompt,
)
from pi_trec.support import SUPPORT_EVAL_PROMPT, iter_support_tasks, render_support_prompt
from pi_trec.umbrela import (
    UMBRELA_ZERO_BASIC,
    UMBRELA_ZERO_BING,
    iter_prompt_tasks,
    render_umbrela_prompt,
)

UPSTREAM = Path(__file__).resolve().parent / "fixtures" / "upstream"


def _yaml_template(rel: str) -> tuple[str, str]:
    data = yaml.safe_load((UPSTREAM / rel).read_text(encoding="utf-8"))
    return str(data["system_message"]), str(data["prefix_user"])


def _support_prompt_literal() -> str:
    source = (UPSTREAM / "trec2024-rag" / "support_evaluation_individual_gpt4o.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SUPPORT_EVAL_PROMPT" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("SUPPORT_EVAL_PROMPT not found in upstream script")


def _castorini_umbrela(prompt_type: str, *, query: str, passage: str) -> str:
    _, prefix_user = _yaml_template(f"umbrela/qrel_zeroshot_{prompt_type}.yaml")
    return prefix_user.format(examples="", query=query, passage=passage)


def _castorini_context(segments: list[str]) -> str:
    return "\n".join(f"[{index + 1}] {segment}" for index, segment in enumerate(segments))


def _castorini_create(*, query: str, segments: list[str], nuggets: list[str], creator_max_nuggets: int) -> str:
    _, prefix_user = _yaml_template("nuggetizer/creator_template.yaml")
    return prefix_user.format(
        query=query,
        context=_castorini_context(segments),
        nuggets=nuggets,
        nuggets_length=len(nuggets),
        creator_max_nuggets=creator_max_nuggets,
    )


def _castorini_score(*, query: str, nuggets: list[str]) -> str:
    _, prefix_user = _yaml_template("nuggetizer/scorer_template.yaml")
    return prefix_user.format(query=query, nuggets=nuggets, num_nuggets=len(nuggets))


def _castorini_assign(*, template: str, query: str, context: str, nuggets: list[str]) -> str:
    _, prefix_user = _yaml_template(f"nuggetizer/{template}")
    return prefix_user.format(query=query, context=context, nuggets=nuggets, num_nuggets=len(nuggets))


def _castorini_support(*, statement: str, citation: str) -> str:
    return _support_prompt_literal().format(statement=statement, citation=citation)


TEXTS = [
    "what is python used for",
    "café résumé — “smart quotes” and naïve façade 日本語",
    "value with {curly} braces and a {query}-like token",
    "",
    "multi\nline value  with trailing space ",
]
NUGGET_LISTS = [
    [],
    ["web development"],
    ["coffee originated in Ethiopia", "spread to Arabia"],
    ["O'Brien's results", 'they said "hi"', "mix ' and \" quotes", "café {token}"],
]
SEGMENT_LISTS = [
    ["a single passage segment"],
    ["first segment", "second segment", "third with trailing space "],
    ["café résumé", "日本語\nsecond line"],
]


@pytest.mark.parametrize("prompt_type", ["basic", "bing"])
@pytest.mark.parametrize("passage", TEXTS)
@pytest.mark.parametrize("query", TEXTS)
def test_umbrela_prompt_byte_identical(prompt_type: str, query: str, passage: str) -> None:
    assert render_umbrela_prompt(
        query=query, passage=passage, prompt_type=prompt_type) == _castorini_umbrela(
        prompt_type, query=query, passage=passage
    )


@pytest.mark.parametrize("prompt_type", ["basic", "bing"])
def test_umbrela_zeroshot_template_fields(prompt_type: str) -> None:
    _, prefix_user = _yaml_template(f"umbrela/qrel_zeroshot_{prompt_type}.yaml")
    fields = {name for _, name, _, _ in Formatter().parse(prefix_user) if name}
    assert fields == {"query", "passage"}


def test_umbrela_template_constants_match_upstream() -> None:
    assert UMBRELA_ZERO_BASIC == _yaml_template("umbrela/qrel_zeroshot_basic.yaml")[1]
    assert UMBRELA_ZERO_BING == _yaml_template("umbrela/qrel_zeroshot_bing.yaml")[1]


def test_umbrela_materialized_instruction_byte_identical() -> None:
    query, passage = TEXTS[1], TEXTS[2]
    tasks = iter_prompt_tasks(
        [{"query": {"qid": "q1", "text": query}, "candidates": [{"doc": {"docid": "d1", "segment": passage}}]}],
        prompt_type="bing",
    )
    assert tasks[0]["instruction"] == _castorini_umbrela("bing", query=query, passage=passage)


@pytest.mark.parametrize("segments", SEGMENT_LISTS)
def test_create_context_matches_castorini(segments: list[str]) -> None:
    assert create_context(segments) == _castorini_context(segments)


@pytest.mark.parametrize("nuggets", NUGGET_LISTS)
@pytest.mark.parametrize("segments", SEGMENT_LISTS)
@pytest.mark.parametrize("query", TEXTS)
@pytest.mark.parametrize("creator_max_nuggets", [30, 10, 5])
def test_nugget_create_prompt_byte_identical(
    creator_max_nuggets: int, query: str, segments: list[str], nuggets: list[str]
) -> None:
    ours = render_create_prompt(
        query=query, context=create_context(segments), nuggets=nuggets, creator_max_nuggets=creator_max_nuggets
    )
    assert ours == _castorini_create(
        query=query, segments=segments, nuggets=nuggets, creator_max_nuggets=creator_max_nuggets
    )


def test_nugget_create_materialized_instruction_byte_identical() -> None:
    query, segments, nuggets = "history of coffee", SEGMENT_LISTS[1], NUGGET_LISTS[3]
    tasks = iter_create_tasks(
        [{"query": {"qid": "q1", "text": query}, "candidates": segments, "nuggets": [{"text": n} for n in nuggets]}],
        creator_max_nuggets=10,
    )
    assert tasks[0]["system_prompt"] == NUGGET_CREATOR_SYSTEM
    assert tasks[0]["instruction"] == _castorini_create(
        query=query, segments=segments, nuggets=nuggets, creator_max_nuggets=10
    )


@pytest.mark.parametrize("nuggets", NUGGET_LISTS)
@pytest.mark.parametrize("query", TEXTS)
def test_nugget_score_prompt_byte_identical(query: str, nuggets: list[str]) -> None:
    assert render_score_prompt(query=query, nuggets=nuggets) == _castorini_score(query=query, nuggets=nuggets)


@pytest.mark.parametrize(
    ("assign_mode", "template"),
    [("support-grade-3", "assigner_template.yaml"), ("support-grade-2", "assigner_2grade_template.yaml")],
)
@pytest.mark.parametrize("nuggets", NUGGET_LISTS)
@pytest.mark.parametrize("context", TEXTS)
@pytest.mark.parametrize("query", TEXTS)
def test_nugget_assign_prompt_byte_identical(
    query: str, context: str, nuggets: list[str], assign_mode: str, template: str
) -> None:
    ours = render_assign_prompt(query=query, context=context, nuggets=nuggets, assign_mode=assign_mode)
    assert ours == _castorini_assign(template=template, query=query, context=context, nuggets=nuggets)


@pytest.mark.parametrize(
    ("assign_mode", "template"),
    [("support-grade-3", "assigner_template.yaml"), ("support-grade-2", "assigner_2grade_template.yaml")],
)
def test_nugget_assign_materialized_instruction_byte_identical(assign_mode: str, template: str) -> None:
    query, context, nuggets = "quote handling", "a supporting passage", NUGGET_LISTS[3]
    inputs = direct_assign_inputs({"query": query, "context": context, "nuggets": [{"text": n} for n in nuggets]})
    tasks = iter_assign_tasks(inputs, assign_mode=assign_mode)
    assert tasks[0]["system_prompt"] == NUGGET_ASSIGNER_SYSTEM
    assert tasks[0]["instruction"] == _castorini_assign(
        template=template, query=query, context=context, nuggets=nuggets
    )


def test_nugget_user_templates_match_upstream() -> None:
    assert NUGGET_CREATOR_USER == _yaml_template("nuggetizer/creator_template.yaml")[1]
    assert NUGGET_SCORER_USER == _yaml_template("nuggetizer/scorer_template.yaml")[1]
    assert NUGGET_ASSIGNER_USER == _yaml_template("nuggetizer/assigner_template.yaml")[1]
    assert NUGGET_ASSIGNER_2GRADE_USER == _yaml_template("nuggetizer/assigner_2grade_template.yaml")[1]


def test_nugget_system_messages_match_upstream() -> None:
    assert NUGGET_CREATOR_SYSTEM == _yaml_template("nuggetizer/creator_template.yaml")[0]
    assert NUGGET_SCORER_SYSTEM == _yaml_template("nuggetizer/scorer_template.yaml")[0]
    assert NUGGET_ASSIGNER_SYSTEM == _yaml_template("nuggetizer/assigner_template.yaml")[0]
    assert NUGGET_ASSIGNER_SYSTEM == _yaml_template("nuggetizer/assigner_2grade_template.yaml")[0]


@pytest.mark.parametrize("citation", TEXTS)
@pytest.mark.parametrize("statement", TEXTS)
def test_support_prompt_byte_identical(statement: str, citation: str) -> None:
    assert render_support_prompt(statement=statement, citation=citation) == _castorini_support(
        statement=statement, citation=citation
    )


def test_support_prompt_constant_matches_upstream_literal() -> None:
    assert SUPPORT_EVAL_PROMPT == _support_prompt_literal()


def test_support_materialized_instruction_byte_identical() -> None:
    statement, citation = TEXTS[1], TEXTS[4]
    tasks = iter_support_tasks([{"task_id": "t", "statement": statement, "citation": citation}])
    assert tasks[0]["instruction"] == _castorini_support(statement=statement, citation=citation)
