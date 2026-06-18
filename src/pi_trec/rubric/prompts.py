from __future__ import annotations

import json
from typing import Any

from pi_trec.nuggetizer.prompts import clean_response, parse_label_list

CRITERION_TYPES = ("explicit", "implicit", "synthesis", "references", "communication", "instruction")
TIERS = ("mandatory", "optional")
VERDICTS = ("satisfied", "partially_satisfied", "not_satisfied")

MANDATORY_WEIGHT = 5
OPTIONAL_WEIGHT = 2


RUBRIC_AUTHOR_SYSTEM = "\
You are RubricAuthorLLM, an evaluation designer that turns a list of scored information nuggets \
into a weighted, reference-independent rubric for grading long-form research answers."


RUBRIC_AUTHOR_USER = """\
You are given a search query and a list of atomic nuggets, each tagged with an id and \
already labeled vital or okay. Author a rubric: a list of criteria that \
an ideal answer to the query should satisfy, each scored against the full answer later. \
Do not grade anything now; only write the criteria.

Criterion types:
- explicit: a point the query directly asks for.
- implicit: a point a well-informed reader expects even if not directly asked.
- synthesis: connecting or reconciling information across nuggets, not a single fact.
- references: appropriate, specific, relevant citation or evidence usage.
- communication: clarity, organization, and tone for the intended reader.
- instruction: adherence to an explicit instruction or constraint in the query.

Importance and weighting are assigned automatically from each nugget's label; do not include them.

Construction rules:
- Each VITAL nugget -> one criterion of type explicit or implicit, asserting the answer conveys that fact.
- Each OKAY nugget -> one criterion asserting the answer conveys that fact.
- Add 1-3 topic-level criteria covering synthesis, references, and communication.
- Set source_nugget to the id (e.g. n0) of the nugget a criterion derives from, or null for topic-level criteria.

Search Query: {query}
Nuggets (id -- text -- importance):
{nuggets}

Return ONLY a JSON array. Each element is an object with keys: text, type, source_nugget. Do not explain.
Rubric:"""


RUBRIC_GRADER_SYSTEM = "You are RubricGraderLLM, \
an intelligent judge that assigns a ternary verdict to each rubric criterion based on a full answer."


RUBRIC_GRADER_USER = """\
Based on the query and the full answer, grade each of the {num_criteria} criteria \
with exactly one verdict from: satisfied, partially_satisfied, not_satisfied.

A criterion is satisfied when the answer states the point clearly and correctly; \
partially_satisfied when the answer gestures at it but is vague, hedged, or incomplete so \
a reader gets only part of it; not_satisfied when it is missing or stated wrongly.

Search Query: {query}
Answer:
{answer}
Criteria:
{criteria}

Return the list of verdicts in a Pythonic list format (type: List[str]), in the same order as the criteria, \
one per criterion. Only return the list. Do not explain.
Verdicts:"""


def nugget_id(nugget: dict[str, Any], index: int) -> str:
    return str(nugget.get("id") or f"n{index}")


def format_nuggets(nuggets: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- {nugget_id(n, i)} -- {n['text']} -- {n.get('importance', 'okay')}"
        for i, n in enumerate(nuggets)
    )


def format_criteria(criteria: list[dict[str, Any]]) -> str:
    return "\n".join(f"{index}. {c['text']}" for index, c in enumerate(criteria, start=1))


def render_author_prompt(*, query: str, nuggets: list[dict[str, Any]]) -> str:
    return RUBRIC_AUTHOR_USER.format(query=query, nuggets=format_nuggets(nuggets))


def render_grade_prompt(*, query: str, answer: str, criteria: list[dict[str, Any]]) -> str:
    return RUBRIC_GRADER_USER.format(
        query=query,
        answer=answer,
        criteria=format_criteria(criteria),
        num_criteria=len(criteria)
    )


def _coerce_authored(item: Any) -> dict[str, Any] | None:
    """Normalize one freshly-authored criterion (text/type/source only)."""
    if not isinstance(item, dict):
        return None
    text = str(item.get("text", "")).strip()
    if not text:
        return None
    ctype = str(item.get("type", "explicit")).strip().lower()
    ctype = ctype if ctype in CRITERION_TYPES else "explicit"
    source = item.get("source_nugget")
    return {"text": text, "type": ctype, "source_nugget": str(source) if source not in (None, "") else None}


def _coerce_criterion(item: Any) -> dict[str, Any] | None:
    """Normalize one criterion from a frozen/hand-edited rubric file, dropping malformed entries."""
    if not isinstance(item, dict):
        return None
    text = str(item.get("text", "")).strip()
    if not text:
        return None
    tier = str(item.get("tier", "optional")).strip().lower()
    tier = tier if tier in TIERS else "optional"
    ctype = str(item.get("type", "explicit")).strip().lower()
    ctype = ctype if ctype in CRITERION_TYPES else "explicit"
    try:
        weight = int(round(float(item.get("weight", 0))))
    except (TypeError, ValueError):
        return None
    weight = max(0, min(5, weight))
    if weight <= 0:
        return None
    source = item.get("source_nugget")
    return {
        "text": text,
        "type": ctype,
        "tier": tier,
        "weight": weight,
        "source_nugget": str(source) if source not in (None, "") else None,
    }


def parse_rubric(text: str) -> list[dict[str, Any]] | None:
    """Extract a freshly-authored JSON array of criteria from a model response."""
    cleaned = clean_response(text)
    open_index = cleaned.find("[")
    close_index = cleaned.rfind("]")
    if open_index == -1 or close_index == -1 or close_index <= open_index:
        return None
    try:
        raw = json.loads(cleaned[open_index : close_index + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, list):
        return None
    criteria = [c for c in (_coerce_authored(item) for item in raw) if c is not None]
    return criteria or None


def finalize_rubric(criteria: list[dict[str, Any]], nuggets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign tier/weight from the source nuggets and guarantee vital coverage."""
    importance_by_id = {
        nugget_id(nugget, index): str(nugget.get("importance", "okay")).strip().lower()
        for index, nugget in enumerate(nuggets)
    }
    finalized: list[dict[str, Any]] = []
    covered_vitals: set[str] = set()
    for criterion in criteria:
        source = criterion.get("source_nugget")
        tier = "mandatory" if importance_by_id.get(source) == "vital" else "optional"
        finalized.append({**criterion, "tier": tier, "weight": MANDATORY_WEIGHT if tier == "mandatory" else OPTIONAL_WEIGHT})
        if tier == "mandatory":
            covered_vitals.add(str(source))
    for index, nugget in enumerate(nuggets):
        nid = nugget_id(nugget, index)
        if importance_by_id[nid] == "vital" and nid not in covered_vitals:
            finalized.append(
                {
                    "text": f"The answer conveys: {nugget['text']}",
                    "type": "explicit",
                    "tier": "mandatory",
                    "weight": MANDATORY_WEIGHT,
                    "source_nugget": nid,
                }
            )
    return finalized


def normalize_criteria(value: Any) -> list[dict[str, Any]]:
    """Coerce a list of criterion objects from a frozen/hand-edited rubric file."""
    if not isinstance(value, list):
        return []
    return [c for c in (_coerce_criterion(item) for item in value) if c is not None]


def parse_verdict_list(text: str) -> list[str] | None:
    """Parse a Pythonic list of verdict strings (reuses the nugget list parser)."""
    return parse_label_list(text)
