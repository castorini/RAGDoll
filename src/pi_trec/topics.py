"""KARL-style topic generation through the Pi local-agent runner."""

from __future__ import annotations

import asyncio
import json
import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pi_trec.config import (
    DEFAULT_CANDIDATES_PER_EPISODE,
    DEFAULT_CATEGORY_COUNT,
    DEFAULT_EPISODES,
    DEFAULT_ICL_EXAMPLES,
    DEFAULT_ICL_SEED,
    DEFAULT_INFORMAL_STYLE_PROBABILITY,
    DEFAULT_MAX_SEARCH_CALLS,
    DEFAULT_MIN_SEARCH_CALLS_PER_CANDIDATE,
    DEFAULT_MIN_UNIQUE_CITED_DOCIDS,
    DEFAULT_SEARCH_TOPK,
    DEFAULT_TOPIC_CATEGORY_SEED,
    TopicsCategoryTaskConfig,
    TopicsGenerateConfig,
    TopicsMaterializeConfig,
    TopicsParseCategoriesConfig,
    TopicsParseConfig,
    TopicsReportConfig,
)
from pi_trec.jsonl import append_jsonl, read_jsonl, write_jsonl
from pi_trec.prompts import PI_SEARCH_SYSTEM_PROMPT
from pi_trec.runner import run_prompt, select_rows

# The DEFAULT_* values above are re-exported from pi_trec.config (single source
# of truth) so existing imports and the CLI keep working unchanged.
REFERENCE_ANSWER_PLACEHOLDER = "<comprehensive grounded reference answer with sentence-level raw-document-id citations>"

JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
CITATION_RE = re.compile(r"\[([^\[\]]+)\]")
DOCID_RE = re.compile(r"(?:docid:\s*)?(shard_\d+_\d+)", re.IGNORECASE)
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")
BIBLIOGRAPHY_RE = re.compile(r"(?im)^\s*(references|bibliography)\s*:")
TOKEN_RE = re.compile(r"[a-z0-9]+")
FORBIDDEN_PROMPT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bretrieved documents?\b", re.IGNORECASE),
    re.compile(r"\bsearch tools?\b", re.IGNORECASE),
    re.compile(r"\brubrics?\b", re.IGNORECASE),
    re.compile(r"\bcitations?\b", re.IGNORECASE),
    re.compile(r"\breferences?\b", re.IGNORECASE),
    re.compile(r"\bnugget\s+(?:criteria|rubrics?|evaluation)\b", re.IGNORECASE),
    re.compile(r"\b(?:benchmark|evaluation)\s+(?:criteria|rubrics?|dataset|datasets)\b", re.IGNORECASE),
    re.compile(r"\bpyserini\b", re.IGNORECASE),
    re.compile(r"\bclimbmix\b", re.IGNORECASE),
    re.compile(r"\bresearchrubrics\b", re.IGNORECASE),
    re.compile(r"\btrec\s+rag\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class IclExample:
    prompt: str
    sample_id: str | None = None
    domain: str | None = None


@dataclass(frozen=True)
class Candidate:
    prompt: str
    reference_answer: str


ICL_EXAMPLES: tuple[IclExample, ...] = (
    IclExample(
        prompt=(
            "I want to create a plan for July 4, 2025, i.e., Independence Day in Washington DC. "
            "I would like an itinerary of all the things to do and all the activities that are "
            "planned for Independence Day. Create a plan for the whole day and also extend it to "
            "the weekend, if required. Provide some reviews or explain why one should visit the "
            "place or engage in the activity. Add any additional information that is required."
        ),
    ),
    IclExample(
        prompt=(
            "Write a synthesis report on the applications of AI in drug discovery for a technical "
            "audience unfamiliar with biology. It should cover the main applications of AI in every "
            "stage of the drug discovery process, the latest technological advancements, challenges, "
            "and current adoption in the real world."
        ),
    ),
    IclExample(
        prompt=(
            "I am a software engineer at a small startup, trying to scale our product to around "
            "1M users / day from around the world. The service is a social media app similar to "
            "twitter, where users can write a message and follow others. Right now we have a simple "
            "web interface with a local mySQL database. Please write a technical report on the "
            "transition to more scaled software and provide recommendations for anything you "
            "consider a necessary change."
        ),
    ),
    IclExample(
        prompt=(
            "I am a high schooler in Cupertino. It is AP season and I'm trying to study for my "
            "tests. Can you give me 10 recommendations for places that open until at least 10pm "
            "that are good for studying (fast wifi, sufficient table space to do practice tests, "
            "good study environment, etc)? I would prefer not having to spend money on drinks/food "
            "in cafes but have a $10 budget if I must, and absolutely refuse to spend money on "
            "parking/fees. I am only open to places that are at most at 20 minute drive from me."
        ),
    ),
)


def render_icl_examples(icl_examples: tuple[IclExample, ...] = ICL_EXAMPLES) -> str:
    blocks: list[str] = []
    for index, example in enumerate(icl_examples, start=1):
        blocks.append(
            "\n".join(
                [
                    f"Example {index}:",
                    "{",
                    f'  "prompt": {json.dumps(example.prompt)},',
                    f'  "reference_answer": {json.dumps(REFERENCE_ANSWER_PLACEHOLDER)}',
                    "}",
                ]
            )
        )
    return "\n\n".join(blocks)


def render_synthesis_prompt(
    *,
    max_search_calls: int,
    candidates_per_episode: int,
    search_topk: int,
    min_unique_cited_docids: int = DEFAULT_MIN_UNIQUE_CITED_DOCIDS,
    min_search_calls_per_candidate: int = DEFAULT_MIN_SEARCH_CALLS_PER_CANDIDATE,
    informal_user_prompt_style: bool = False,
    target_topic_category: str | None = None,
    icl_examples: tuple[IclExample, ...] = ICL_EXAMPLES,
) -> str:
    informal_style_line = (
        "- Write the generated user prompt in a more informal, personal, "
        "or imperfectly structured voice when the topic allows it; it should feel "
        "like a real person asking for help, not a polished institutional memo.\n"
        if informal_user_prompt_style
        else ""
    )
    topic_category_line = (
        "Create the new benchmark example within this broad topical category: "
        f"{target_topic_category}. Treat it as a starting point for topic selection, "
        "not wording to copy.\n\n"
        if target_topic_category
        else ""
    )
    return f"""You are a benchmark data synthesis agent creating grounded long-form research tasks.

You will be given a small set of in-context examples showing the desired format and style. Follow their style, but create new topics that are semantically distinct from the examples. Do not reuse the example topics, document identifiers, entities, wording, or structure too closely.

{topic_category_line}\
You may make up to {max_search_calls} search calls while creating these examples. Search calls return up to {search_topk} results.

Create exactly {candidates_per_episode} new benchmark examples.

For each final benchmark example, perform at least {min_search_calls_per_candidate} distinct search calls about that example's topic before writing its reference answer. Distinct search calls should use meaningfully different query strings for that topic. Do not count result-page reads as search calls.

If the first page of search results does not expose enough distinct sources or source contexts, you may use result-page reads to browse deeper ranks before issuing another search or choosing documents to read.

Each example must contain:
1. A realistic single-turn user prompt.
2. A grounded reference answer with sentence-level citations using raw document ids.

The user prompt should:
- Ask for a long-form deliverable such as a report, memo, plan, recommendation, comparative analysis, technical overview, or synthesis.
- Read like a realistic request from a person with a concrete information need.
- Include specific context, audience, constraints, or desired coverage.
- Usually be about 30-240 words.
{informal_style_line}- Require combining information from multiple sources in the available corpus.
- Be answerable from the available corpus.
- Be semantically distinct from the in-context examples.
- Not include the reference answer.
- Not mention retrieved documents, search tools, rubrics, nuggets, benchmarks, citations, references, or evaluation.

The reference answer should:
- Answer the prompt directly.
- Be grounded only in information found while completing this task.
- Synthesize evidence from at least {min_unique_cited_docids} distinct source documents across the full answer.
- Prefer 10-15 distinct source documents when the available evidence supports the topic.
- Choose topics with enough available evidence for broad source coverage.
- Be 500-1,000 words.
- Include concrete facts, relationships, tradeoffs, examples, and limitations where supported.
- Avoid unsupported claims.
- State uncertainty when available information is incomplete or conflicting.
- Put citations at the end of the sentence they support.
- Prefer sentence-level citation coverage, especially for concrete factual claims.
- It is acceptable for a high-level thesis, recommendation, transition, or summary sentence to be uncited when nearby cited sentences support the point.
- Use raw document ids directly in square brackets.
- Use at most 3 document ids per sentence; this is a sentence-level cap, not an answer-level cap.
- Ensure every factual sentence has at least one citation.
- Do not pad with weak or irrelevant citations just to increase the distinct source count.
- Do not include a bibliography or references section.

Citation examples:
- One source: [shard_01159_55212]
- Multiple sources: [shard_01159_55212; shard_03388_49566]

Create diversity across the {candidates_per_episode} examples:
- Use different domains or subdomains when possible.
- Vary task type, audience, and user context.
- Avoid prompts that are small variations of the same topic.

Return exactly one JSON array with {candidates_per_episode} objects. Do not include commentary outside the JSON.

Schema:
[
  {{
    "prompt": "...",
    "reference_answer": "..."
  }}
]

In-context examples:

{render_icl_examples(icl_examples)}"""


def load_researchrubrics_examples(path: Path) -> tuple[IclExample, ...]:
    examples: list[IclExample] = []
    for row in read_jsonl(path):
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        sample_id = row.get("sample_id")
        domain = row.get("domain")
        examples.append(
            IclExample(
                prompt=prompt.strip(),
                sample_id=str(sample_id) if sample_id is not None else None,
                domain=str(domain) if domain is not None else None,
            )
        )
    if not examples:
        raise ValueError(f"{path}: no ResearchRubrics prompts found")
    return tuple(examples)


def load_topic_categories(path: Path) -> tuple[str, ...]:
    categories = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if not categories:
        raise ValueError(f"{path}: no topic categories found")
    return categories


def sample_icl_examples(*, pool: tuple[IclExample, ...], episode_index: int, count: int, seed: int) -> tuple[IclExample, ...]:
    rng = random.Random(f"{seed}:{episode_index}")
    return tuple(rng.choice(pool) for _ in range(count))


def sample_topic_category(*, categories: tuple[str, ...], episode_index: int, seed: int) -> str:
    rng = random.Random(f"{seed}:topic-category:{episode_index}")
    return rng.choice(categories)


def sample_informal_user_prompt_style(*, episode_index: int, seed: int, probability: float) -> bool:
    rng = random.Random(f"{seed}:informal-style:{episode_index}")
    return rng.random() < probability


def icl_metadata(examples: tuple[IclExample, ...]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for example in examples:
        row = {"prompt": example.prompt}
        if example.sample_id is not None:
            row["sample_id"] = example.sample_id
        if example.domain is not None:
            row["domain"] = example.domain
        rows.append(row)
    return rows


def build_task(
    *,
    episode_index: int,
    candidates_per_episode: int,
    max_search_calls: int,
    search_topk: int,
    min_unique_cited_docids: int,
    min_search_calls_per_candidate: int,
    informal_user_prompt_style: bool,
    informal_style_probability: float,
    target_topic_category: str | None,
    topic_category_seed: int | None,
    icl_examples: tuple[IclExample, ...],
    icl_source: str,
    icl_seed: int,
) -> dict[str, Any]:
    task_id = f"karl_synth_episode_{episode_index:06d}"
    metadata: dict[str, Any] = {
        "pipeline": "karl_style_qa_synthesis",
        "episode_index": episode_index,
        "max_search_calls": max_search_calls,
        "search_topk": search_topk,
        "candidates_requested": candidates_per_episode,
        "min_unique_cited_docids": min_unique_cited_docids,
        "min_search_calls_per_candidate": min_search_calls_per_candidate,
        "informal_user_prompt_style": informal_user_prompt_style,
        "informal_style_probability": informal_style_probability,
        "icl_source": icl_source,
        "icl_seed": icl_seed,
        "icl_examples": icl_metadata(icl_examples),
    }
    if target_topic_category is not None:
        metadata["target_topic_category"] = target_topic_category
        metadata["topic_category_seed"] = topic_category_seed
    return {
        "task_id": task_id,
        "instruction": render_synthesis_prompt(
            max_search_calls=max_search_calls,
            candidates_per_episode=candidates_per_episode,
            search_topk=search_topk,
            min_unique_cited_docids=min_unique_cited_docids,
            min_search_calls_per_candidate=min_search_calls_per_candidate,
            informal_user_prompt_style=informal_user_prompt_style,
            target_topic_category=target_topic_category,
            icl_examples=icl_examples,
        ),
        "input_text": "",
        "metadata": metadata,
    }


def iter_tasks(
    *,
    episodes: int,
    candidates_per_episode: int,
    max_search_calls: int,
    search_topk: int,
    min_unique_cited_docids: int,
    min_search_calls_per_candidate: int,
    informal_style_probability: float,
    topic_categories: tuple[str, ...] | None,
    topic_category_seed: int,
    icl_pool: tuple[IclExample, ...],
    icl_source: str,
    icl_examples_per_episode: int,
    icl_seed: int,
) -> list[dict[str, Any]]:
    required_search_calls = candidates_per_episode * min_search_calls_per_candidate
    if required_search_calls > max_search_calls:
        raise ValueError(
            "candidates_per_episode times min_search_calls_per_candidate "
            f"requires {required_search_calls} searches, above max_search_calls {max_search_calls}"
        )
    return [
        build_task(
            episode_index=episode_index,
            candidates_per_episode=candidates_per_episode,
            max_search_calls=max_search_calls,
            search_topk=search_topk,
            min_unique_cited_docids=min_unique_cited_docids,
            min_search_calls_per_candidate=min_search_calls_per_candidate,
            informal_user_prompt_style=sample_informal_user_prompt_style(
                episode_index=episode_index,
                seed=icl_seed,
                probability=informal_style_probability,
            ),
            informal_style_probability=informal_style_probability,
            target_topic_category=(
                sample_topic_category(categories=topic_categories, episode_index=episode_index, seed=topic_category_seed)
                if topic_categories is not None
                else None
            ),
            topic_category_seed=topic_category_seed if topic_categories is not None else None,
            icl_examples=sample_icl_examples(pool=icl_pool, episode_index=episode_index, count=icl_examples_per_episode, seed=icl_seed),
            icl_source=icl_source,
            icl_seed=icl_seed,
        )
        for episode_index in range(1, episodes + 1)
    ]


def materialize(config: TopicsMaterializeConfig) -> None:
    _require_positive("episodes", config.episodes)
    _require_positive("candidates-per-episode", config.candidates_per_episode)
    _require_positive("max-search-calls", config.max_search_calls)
    _require_positive("search-topk", config.search_topk)
    _require_positive("min-unique-cited-docids", config.min_unique_cited_docids)
    _require_positive("min-search-calls-per-candidate", config.min_search_calls_per_candidate)
    _require_positive("icl-examples", config.icl_examples)
    _require_probability("informal-style-probability", config.informal_style_probability)
    if config.icl_source == "researchrubrics" and config.researchrubrics_path is None:
        raise SystemExit("--researchrubrics-path is required when --icl-source=researchrubrics")
    icl_pool = load_researchrubrics_examples(config.researchrubrics_path) if config.icl_source == "researchrubrics" else ICL_EXAMPLES
    topic_categories = load_topic_categories(config.topic_categories) if config.topic_categories else None
    tasks = iter_tasks(
        episodes=config.episodes,
        candidates_per_episode=config.candidates_per_episode,
        max_search_calls=config.max_search_calls,
        search_topk=config.search_topk,
        min_unique_cited_docids=config.min_unique_cited_docids,
        min_search_calls_per_candidate=config.min_search_calls_per_candidate,
        informal_style_probability=config.informal_style_probability,
        topic_categories=topic_categories,
        topic_category_seed=config.topic_category_seed,
        icl_pool=icl_pool,
        icl_source=config.icl_source,
        icl_examples_per_episode=config.icl_examples,
        icl_seed=config.icl_seed,
    )
    count = write_jsonl(config.output_file, tasks)
    print(f"wrote={count} output={config.output_file}")


async def generate(config: TopicsGenerateConfig) -> None:
    if config.overwrite:
        if config.output_file.exists():
            config.output_file.unlink()
        if config.failed_output and config.failed_output.exists():
            config.failed_output.unlink()
    rows = select_rows(
        list(read_jsonl(config.input_file)),
        output=config.output_file,
        resume=config.resume,
        overwrite=config.overwrite,
        shuffle=config.shuffle,
        seed=config.seed,
        limit=config.limit,
    )
    system_prompt = config.system_prompt
    if config.extension_path is not None and system_prompt == "":
        system_prompt = PI_SEARCH_SYSTEM_PROMPT
    agent_config = replace(config.local_agent_config(), system_prompt=system_prompt)
    raw_events_dir = config.raw_events_dir or config.output_file.parent / "raw-events" / config.output_file.stem
    semaphore = asyncio.Semaphore(max(1, config.max_concurrency))

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            row_config = replace(agent_config, system_prompt=str(row.get("system_prompt", agent_config.system_prompt)))
            return await run_prompt(
                task_id=str(row["task_id"]),
                evaluator="topics-generate",
                instruction=str(row["instruction"]),
                raw_events_dir=raw_events_dir,
                config=row_config,
                metadata=row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
            )

    pending = [asyncio.create_task(one(row)) for row in rows]
    for future in asyncio.as_completed(pending):
        result = await future
        if result["status"] == "completed":
            append_jsonl(config.output_file, result)
        elif config.failed_output:
            append_jsonl(config.failed_output, result)
        print(f"{result['status']} task_id={result['task_id']}", flush=True)
    print(f"processed={len(rows)} output={config.output_file} raw_events_dir={raw_events_dir}")


def parse(config: TopicsParseConfig) -> None:
    if config.candidates_per_episode <= 0:
        raise SystemExit("--candidates-per-episode must be a positive integer")
    existing_prompts = [] if config.skip_existing_dedup else load_existing_prompts(config.existing_prompt_file)
    accepted, rejected, summary = process_results(
        list(read_jsonl(config.input_file)),
        candidates_per_episode=config.candidates_per_episode,
        existing_prompts=existing_prompts,
    )
    write_jsonl(config.output_file, accepted)
    write_jsonl(config.rejected_output, rejected)
    config.summary_output.parent.mkdir(parents=True, exist_ok=True)
    config.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"accepted={len(accepted)} rejected={len(rejected)} output={config.output_file} summary={config.summary_output}")


def report(config: TopicsReportConfig) -> None:
    rows = list(read_jsonl(config.input_file))
    summary = load_summary(config.summary_input)
    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    config.output_file.write_text(render_report(rows, summary), encoding="utf-8")
    print(f"wrote report={config.output_file}")


def _strip_json_fence(text: str) -> str:
    match = JSON_FENCE_RE.match(text)
    return match.group(1) if match else text


def parse_output_json(output_text: str) -> Any:
    stripped = _strip_json_fence(output_text.strip())
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON output: {exc}") from exc


def parse_candidate_array(output_text: str, *, expected_count: int) -> list[Candidate]:
    value = parse_output_json(output_text)
    if not isinstance(value, list):
        raise ValueError("output must be a JSON array")
    if len(value) != expected_count:
        raise ValueError(f"expected {expected_count} candidates, found {len(value)}")
    candidates: list[Candidate] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"candidate {index} must be a JSON object")
        prompt = item.get("prompt")
        reference_answer = item.get("reference_answer")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"candidate {index} has an empty prompt")
        if not isinstance(reference_answer, str) or not reference_answer.strip():
            raise ValueError(f"candidate {index} has an empty reference_answer")
        candidates.append(Candidate(prompt=prompt.strip(), reference_answer=normalize_citation_groups(reference_answer.strip())))
    return candidates


def normalize_citation_groups(text: str, *, max_docids_per_group: int = 3) -> str:
    def replace(match: re.Match[str]) -> str:
        docids = [docid for docid in DOCID_RE.findall(match.group(1))]
        if not docids:
            return match.group(0)
        return "[" + "; ".join(docids[:max_docids_per_group]) + "]"

    return CITATION_RE.sub(replace, text)


def split_cited_docids(group_text: str) -> list[str]:
    docids = [docid for docid in DOCID_RE.findall(group_text)]
    if not docids or any(not docid for docid in docids):
        raise ValueError("citation group contains an empty document id")
    docids = docids[:3]
    for docid in docids:
        if not re.fullmatch(r"shard_\d+_\d+", docid):
            raise ValueError(f"invalid citation document id: {docid}")
    return docids


def citation_groups(text: str) -> list[list[str]]:
    groups: list[list[str]] = []
    for match in CITATION_RE.finditer(text):
        if DOCID_RE.search(match.group(1)):
            groups.append(split_cited_docids(match.group(1)))
    return groups


def factual_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for match in SENTENCE_RE.finditer(text):
        sentence = match.group(0).strip()
        words = re.findall(r"[A-Za-z0-9]+", sentence)
        if len(words) >= 4:
            sentences.append(sentence)
    return sentences


def unique_cited_docids(text: str) -> set[str]:
    docids: set[str] = set()
    for group in citation_groups(text):
        docids.update(group)
    return docids


def uncited_factual_sentences(text: str) -> list[str]:
    return [sentence for sentence in factual_sentences(text) if not CITATION_RE.findall(sentence)]


def validate_reference_answer_citations(reference_answer: str, *, min_unique_docids: int = DEFAULT_MIN_UNIQUE_CITED_DOCIDS) -> list[str]:
    errors: list[str] = []
    if BIBLIOGRAPHY_RE.search(reference_answer):
        errors.append("reference_answer must not include a references or bibliography section")
    try:
        cited_docids = unique_cited_docids(reference_answer)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        if len(cited_docids) < min_unique_docids:
            errors.append(f"reference_answer cites {len(cited_docids)} unique document ids, below minimum {min_unique_docids}")
    return errors


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def validate_prompt_policy(prompt: str, *, min_words: int = 30, max_words: int = 240) -> list[str]:
    errors: list[str] = []
    words = word_count(prompt)
    if words < min_words:
        errors.append(f"prompt has {words} words, below minimum {min_words}")
    if words > max_words:
        errors.append(f"prompt has {words} words, above maximum {max_words}")
    if re.search(r"(?im)^\s*(user|assistant|system)\s*:", prompt):
        errors.append("prompt must be a single-turn user request without role labels")
    if "reference_answer" in prompt or "reference answer" in prompt.lower():
        errors.append("prompt must not include or request the hidden reference answer")
    for pattern in FORBIDDEN_PROMPT_PATTERNS:
        if pattern.search(prompt):
            errors.append(f"prompt contains forbidden term matching {pattern.pattern}")
    return errors


def normalized_prompt(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def prompt_token_set(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def token_jaccard(left: str, right: str) -> float:
    left_tokens = prompt_token_set(left)
    right_tokens = prompt_token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def duplicate_reason(prompt: str, existing_prompts: list[str], *, jaccard_threshold: float = 0.85, prefix_words: int = 16) -> str | None:
    normalized = normalized_prompt(prompt)
    prompt_prefix = " ".join(normalized.split()[:prefix_words])
    for existing in existing_prompts:
        existing_normalized = normalized_prompt(existing)
        if normalized == existing_normalized:
            return "exact_duplicate"
        if prompt_prefix and prompt_prefix == " ".join(existing_normalized.split()[:prefix_words]):
            return "near_duplicate_prefix"
        if token_jaccard(prompt, existing) >= jaccard_threshold:
            return "near_duplicate_token_overlap"
    return None


def candidate_errors(candidate: Candidate) -> list[str]:
    return [*validate_prompt_policy(candidate.prompt), *validate_reference_answer_citations(candidate.reference_answer)]


def accepted_row(*, task_id: str, candidate_index: int, candidate: Candidate) -> dict[str, Any]:
    return {
        "candidate_id": f"{task_id}_{candidate_index:02d}",
        "prompt": candidate.prompt,
        "reference_answer": candidate.reference_answer,
        "source_task_id": task_id,
    }


def rejected_row(*, task_id: str, reason: str, errors: list[str], candidate_index: int | None = None, candidate: Candidate | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"source_task_id": task_id, "reason": reason, "errors": errors}
    if candidate_index is not None:
        row["candidate_index"] = candidate_index
    if candidate is not None:
        row["prompt"] = candidate.prompt
        row["reference_answer"] = candidate.reference_answer
    return row


def process_results(
    rows: list[dict[str, Any]],
    *,
    candidates_per_episode: int,
    existing_prompts: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    seen_prompts = list(existing_prompts or [])
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            reason_counts["missing_task_id"] += 1
            rejected.append(rejected_row(task_id="", reason="missing_task_id", errors=["task_id is required"]))
            continue
        if row.get("status") != "completed":
            reason_counts["non_completed_result"] += 1
            rejected.append(rejected_row(task_id=task_id, reason="non_completed_result", errors=[str(row.get("error") or "status is not completed")]))
            continue
        try:
            candidates = parse_candidate_array(str(row.get("output_text") or ""), expected_count=candidates_per_episode)
        except ValueError as exc:
            reason_counts["parse_error"] += 1
            rejected.append(rejected_row(task_id=task_id, reason="parse_error", errors=[str(exc)]))
            continue
        for candidate_index, candidate in enumerate(candidates):
            errors = candidate_errors(candidate)
            if errors:
                reason_counts["candidate_validation_error"] += 1
                rejected.append(
                    rejected_row(
                        task_id=task_id,
                        candidate_index=candidate_index,
                        candidate=candidate,
                        reason="candidate_validation_error",
                        errors=errors,
                    )
                )
                continue
            dedup_reason = duplicate_reason(candidate.prompt, seen_prompts)
            if dedup_reason is not None:
                reason_counts["near_duplicate"] += 1
                rejected.append(
                    rejected_row(
                        task_id=task_id,
                        candidate_index=candidate_index,
                        candidate=candidate,
                        reason="near_duplicate",
                        errors=[dedup_reason],
                    )
                )
                continue
            accepted.append(accepted_row(task_id=task_id, candidate_index=candidate_index, candidate=candidate))
            seen_prompts.append(candidate.prompt)
    summary = {
        "pine_results": len(rows),
        "accepted_candidates": len(accepted),
        "existing_prompts_loaded": len(existing_prompts or []),
        "rejected_rows": len(rejected),
        "rejection_reasons": dict(sorted(reason_counts.items())),
    }
    return accepted, rejected, summary


def load_existing_prompts(paths: list[Path] | None) -> list[str]:
    prompts: list[str] = []
    for path in paths or []:
        if path.suffix == ".jsonl":
            for row in read_jsonl(path):
                for key in ("prompt", "title", "query"):
                    value = row.get(key)
                    if isinstance(value, str) and value.strip():
                        prompts.append(value.strip())
                        break
        else:
            prompts.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return prompts


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def describe(values: list[int]) -> str:
    if not values:
        return "n/a"
    return f"min={min(values)}, mean={statistics.mean(values):.1f}, median={statistics.median(values):.1f}, max={max(values)}"


def duplicate_prompt_hashes(rows: list[dict[str, Any]]) -> int:
    seen: set[str] = set()
    duplicates = 0
    for row in rows:
        normalized = normalized_prompt(str(row.get("prompt") or ""))
        if normalized in seen:
            duplicates += 1
        else:
            seen.add(normalized)
    return duplicates


def render_report(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    prompt_lengths = [word_count(str(row.get("prompt") or "")) for row in rows]
    answer_lengths = [word_count(str(row.get("reference_answer") or "")) for row in rows]
    citation_group_counts = [len(citation_groups(str(row.get("reference_answer") or ""))) for row in rows]
    cited_doc_counts = [len(unique_cited_docids(str(row.get("reference_answer") or ""))) for row in rows]
    factual_sentence_counts = [len(factual_sentences(str(row.get("reference_answer") or ""))) for row in rows]
    uncited_sentence_counts = [len(uncited_factual_sentences(str(row.get("reference_answer") or ""))) for row in rows]
    lines = [
        "# KARL Synthesis Candidate Report",
        "",
        f"- Accepted candidates: {len(rows)}",
        f"- Rejected rows: {summary.get('rejected_rows', 'n/a')}",
        f"- Prompt words: {describe(prompt_lengths)}",
        f"- Reference-answer words: {describe(answer_lengths)}",
        f"- Citation groups per answer: {describe(citation_group_counts)}",
        f"- Unique cited docids per answer: {describe(cited_doc_counts)}",
        f"- Sentence-like factual units per answer: {describe(factual_sentence_counts)}",
        f"- Uncited sentence-like factual units per answer: {describe(uncited_sentence_counts)}",
        f"- Duplicate normalized prompts: {duplicate_prompt_hashes(rows)}",
    ]
    reasons = summary.get("rejection_reasons")
    if isinstance(reasons, dict) and reasons:
        lines.extend(["", "## Rejection Reasons", ""])
        for reason, count in sorted(reasons.items()):
            lines.append(f"- {reason}: {count}")
    return "\n".join(lines) + "\n"


def render_category_prompt(*, examples: list[dict[str, str]], category_count: int) -> str:
    blocks = []
    for index, example in enumerate(examples, start=1):
        blocks.append("\n".join([f"Input query {index}", f"Domain: {example['domain']}", f"Prompt: {example['prompt']}"]))
    return f"""You are creating a broad topic-category inventory for long-form research query generation.

You will be given ResearchRubrics-style input queries. Use them only to understand the range of possible user needs, domains, deliverable types, audiences, and topical breadth.

Create exactly {category_count} unique top-level topic categories.

Category requirements:
- Write one category per line.
- Use natural category phrases; vary the phrase length and wording instead of making every line the same 2-3 word noun-phrase pattern.
- Make categories broad enough to guide many future topics, not labels for just one input query.
- Make categories specific enough to prevent all future topics from collapsing into the same evidence-rich public-policy area.
- Prefer reusable category families over implementation details, subcomponents, named examples, or narrow constraints from a single prompt.
- Keep categories meaningfully distinct from each other; avoid near-duplicates, sibling variants, and multiple categories that only split one original prompt into small pieces.
- Cover the diversity of the input queries across technical, scientific, consumer, historical, business, current-events, philosophical, creative, and practical-help topics.
- Do not copy named entities, dates, document identifiers, or exact wording from the input queries.
- Do not include numbering, bullets, headings, citations, explanations, or blank lines.

Input queries:

{chr(10).join(blocks)}
"""


def category_task_row(*, instruction: str, category_count: int, input_count: int) -> dict[str, Any]:
    return {
        "task_id": "rr_top_level_categories_step0",
        "instruction": instruction,
        "input_text": "",
        "metadata": {
            "pipeline": "rr_top_level_category_generation",
            "category_count": category_count,
            "researchrubrics_prompts": input_count,
        },
    }


def load_researchrubrics_prompts(path: Path) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    for row in read_jsonl(path):
        domain = row.get("domain")
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        prompts.append({"domain": domain.strip() if isinstance(domain, str) and domain.strip() else "Unknown", "prompt": prompt.strip()})
    if not prompts:
        raise ValueError(f"{path}: no ResearchRubrics prompts found")
    return prompts


def category_task(config: TopicsCategoryTaskConfig) -> None:
    _require_positive("category-count", config.category_count)
    examples = load_researchrubrics_prompts(config.researchrubrics_path)
    task = category_task_row(
        instruction=render_category_prompt(examples=examples, category_count=config.category_count),
        category_count=config.category_count,
        input_count=len(examples),
    )
    count = write_jsonl(config.output_file, [task])
    print(f"wrote={count} output={config.output_file} researchrubrics_prompts={len(examples)}")


def normalize_category(line: str) -> str:
    category = line.strip()
    category = re.sub(r"^\s*(?:[-*]+|\d+[\).\:-]+)\s*", "", category)
    category = re.sub(r"\s+", " ", category)
    return category.strip(" \t\r\n\"'")


def parse_categories_text(text: str) -> list[str]:
    categories: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        category = normalize_category(line)
        if not category:
            continue
        key = category.casefold()
        if key in seen:
            continue
        seen.add(key)
        categories.append(category)
    return categories


def completed_output_text(rows: list[dict[str, Any]]) -> str:
    completed = [row for row in rows if row.get("status") == "completed"]
    if len(completed) != 1:
        raise ValueError(f"expected exactly one completed Pine result, found {len(completed)}")
    output_text = completed[0].get("output_text")
    if not isinstance(output_text, str) or not output_text.strip():
        raise ValueError("completed Pine result has empty output_text")
    return output_text


def parse_categories(config: TopicsParseCategoriesConfig) -> None:
    _require_positive("category-count", config.category_count)
    categories = parse_categories_text(completed_output_text(list(read_jsonl(config.input_file))))
    if len(categories) != config.category_count:
        raise SystemExit(f"expected {config.category_count} unique categories, parsed {len(categories)}")
    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    config.output_file.write_text("\n".join(categories) + "\n", encoding="utf-8")
    summary = {"input": str(config.input_file), "output": str(config.output_file), "categories": len(categories)}
    config.summary_output.parent.mkdir(parents=True, exist_ok=True)
    config.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote={len(categories)} output={config.output_file} summary={config.summary_output}")


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise SystemExit(f"--{name} must be a positive integer")


def _require_probability(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise SystemExit(f"--{name} must be between 0 and 1")
