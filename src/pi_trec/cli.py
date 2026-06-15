"""Command-line interface for pi-trec."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from pi_trec.runner import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_THINKING,
    DEFAULT_TIMEOUT_SECONDS,
    run_task_rows,
)
from pi_trec import nuggetizer, pyserini_wrapper, support, topics, umbrela


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run internal prompt tasks through Pi.")
    run_subparsers = run.add_subparsers(dest="run_command", required=True)
    local_agent = run_subparsers.add_parser("local-agent", help="Run task JSONL through a local Pi agent.")
    add_runner_args(local_agent)
    local_agent.set_defaults(func=lambda args: asyncio.run(run_task_rows(args)))

    serve = subparsers.add_parser("serve", help="Serve helper endpoints for Pi RAG evaluation runs.")
    serve_subparsers = serve.add_subparsers(dest="serve_command", required=True)
    pyserini = serve_subparsers.add_parser(
        "pyserini-wrapper",
        help="Wrap a Pyserini HTTP endpoint as the pi-search http-json backend contract.",
    )
    pyserini.add_argument("--pyserini-base-url", required=True)
    pyserini.add_argument("--pyserini-index", required=True)
    pyserini.add_argument("--host", default="127.0.0.1")
    pyserini.add_argument("--port", type=int, default=8091)
    pyserini.add_argument("--backend-id", default="pyserini-http")
    pyserini.add_argument("--default-limit", type=int, default=10)
    pyserini.add_argument("--max-page-size", type=int, default=100)
    pyserini.add_argument("--read-limit", type=int, default=200)
    pyserini.add_argument("--search-word-limit", type=int, default=512)
    pyserini.add_argument("--read-word-limit", type=int, default=4096)
    pyserini.add_argument("--token-env", default="PYSERINI_API_TOKEN")
    pyserini.add_argument("--print-config", action="store_true")
    pyserini.set_defaults(func=pyserini_wrapper.serve_pyserini_wrapper)

    materialize = subparsers.add_parser("materialize", help="Materialize evaluator prompts without running Pi.")
    materialize_subparsers = materialize.add_subparsers(dest="materialize_command", required=True)
    materialize_umbrela = materialize_subparsers.add_parser("umbrela", help="Materialize UMBRELA prompt tasks.")
    materialize_umbrela.add_argument("--input-file", type=Path, required=True)
    materialize_umbrela.add_argument("--output-file", type=Path, required=True)
    materialize_umbrela.add_argument("--prompt-type", choices=["bing", "basic"], default="bing")
    materialize_umbrela.set_defaults(func=umbrela.materialize)
    materialize_nugget_create = materialize_subparsers.add_parser("nugget-create", help="Materialize Nuggetizer create prompts.")
    materialize_nugget_create.add_argument("--input-file", type=Path, required=True)
    materialize_nugget_create.add_argument("--output-file", type=Path, required=True)
    materialize_nugget_create.add_argument("--max-nuggets", type=int, default=30)
    materialize_nugget_create.set_defaults(func=nuggetizer.materialize_create)
    materialize_nugget_agentic_create = materialize_subparsers.add_parser(
        "nugget-agentic-create",
        help="Materialize Nuggetizer agentic create prompts.",
    )
    materialize_nugget_agentic_create.add_argument("--input-file", type=Path, required=True)
    materialize_nugget_agentic_create.add_argument("--output-file", type=Path, required=True)
    materialize_nugget_agentic_create.add_argument("--max-nuggets", type=int, default=30)
    materialize_nugget_agentic_create.set_defaults(func=nuggetizer.materialize_agentic_create)
    materialize_nugget_score = materialize_subparsers.add_parser("nugget-score", help="Materialize Nuggetizer score prompts.")
    materialize_nugget_score.add_argument("--input-file", type=Path, required=True)
    materialize_nugget_score.add_argument("--output-file", type=Path, required=True)
    materialize_nugget_score.set_defaults(func=nuggetizer.materialize_score)
    materialize_nugget_assign = materialize_subparsers.add_parser("nugget-assign", help="Materialize Nuggetizer assign prompts.")
    add_assign_input_args(materialize_nugget_assign)
    materialize_nugget_assign.add_argument("--output-file", type=Path, required=True)
    materialize_nugget_assign.add_argument("--assign-mode", choices=["support-grade-3", "support-grade-2"], default="support-grade-3")
    materialize_nugget_assign.set_defaults(func=nuggetizer.materialize_assign)
    materialize_support = materialize_subparsers.add_parser("support", help="Materialize support-evaluation prompts.")
    materialize_support.add_argument("--input-file", type=Path, required=True)
    materialize_support.add_argument("--output-file", type=Path, required=True)
    materialize_support.set_defaults(func=support.materialize)

    umbrela_parser = subparsers.add_parser("umbrela", help="Run UMBRELA-compatible relevance judging.")
    umbrela_subparsers = umbrela_parser.add_subparsers(dest="umbrela_command", required=True)
    umbrela_judge = umbrela_subparsers.add_parser("judge", help="Judge query-candidate relevance through Pi.")
    add_runner_args(umbrela_judge)
    umbrela_judge.add_argument("--prompt-type", choices=["bing", "basic"], default="bing")
    umbrela_judge.add_argument("--include-trace", action="store_true")
    umbrela_judge.add_argument("--redact-prompts", action="store_true")
    umbrela_judge.set_defaults(func=lambda args: asyncio.run(umbrela.judge(args)))

    nuggetizer_parser = subparsers.add_parser("nuggetizer", help="Run Nuggetizer-compatible prompts through Pi.")
    nuggetizer_subparsers = nuggetizer_parser.add_subparsers(dest="nuggetizer_command", required=True)
    nugget_create = nuggetizer_subparsers.add_parser("create", help="Create and score nuggets through Pi.")
    add_runner_args(nugget_create)
    nugget_create.add_argument("--max-nuggets", type=int, default=30)
    nugget_create.add_argument("--include-trace", action="store_true")
    nugget_create.set_defaults(func=lambda args: asyncio.run(nuggetizer.create(args)))
    nugget_agentic_create = nuggetizer_subparsers.add_parser(
        "agentic-create",
        help="Create and score nuggets with Pi search/read-document tools.",
    )
    add_runner_args(nugget_agentic_create)
    nugget_agentic_create.add_argument("--max-nuggets", type=int, default=30)
    nugget_agentic_create.add_argument("--include-trace", action="store_true")
    nugget_agentic_create.set_defaults(func=lambda args: asyncio.run(nuggetizer.agentic_create(args)))
    nugget_assign = nuggetizer_subparsers.add_parser("assign", help="Assign nuggets through Pi.")
    add_runner_args(nugget_assign, include_input_file=False)
    add_assign_input_args(nugget_assign)
    nugget_assign.add_argument("--assign-mode", choices=["support-grade-3", "support-grade-2"], default="support-grade-3")
    nugget_assign.add_argument("--include-trace", action="store_true")
    nugget_assign.set_defaults(func=lambda args: asyncio.run(nuggetizer.assign(args)))

    support_parser = subparsers.add_parser("support", help="Run support evaluation through Pi.")
    support_subparsers = support_parser.add_subparsers(dest="support_command", required=True)
    support_judge = support_subparsers.add_parser("judge", help="Judge statement-citation support through Pi.")
    add_runner_args(support_judge)
    support_judge.add_argument("--include-prompt", action="store_true")
    support_judge.set_defaults(func=lambda args: asyncio.run(support.judge(args)))

    topics_parser = subparsers.add_parser("topics", help="Run KARL-style topic generation through Pi.")
    topics_subparsers = topics_parser.add_subparsers(dest="topics_command", required=True)

    topics_materialize = topics_subparsers.add_parser("materialize", help="Materialize Pine-compatible topic-generation tasks.")
    add_topics_materialize_args(topics_materialize)
    topics_materialize.set_defaults(func=topics.materialize)

    topics_generate = topics_subparsers.add_parser("generate", help="Run topic-generation tasks through Pi.")
    add_runner_args(topics_generate)
    topics_generate.set_defaults(func=lambda args: asyncio.run(topics.generate(args)))

    topics_parse = topics_subparsers.add_parser("parse", help="Parse topic-generation results into candidates.")
    topics_parse.add_argument("--input-file", type=Path, required=True)
    topics_parse.add_argument("--output-file", type=Path, required=True)
    topics_parse.add_argument("--rejected-output", type=Path, required=True)
    topics_parse.add_argument("--summary-output", type=Path, required=True)
    topics_parse.add_argument("--candidates-per-episode", type=int, default=topics.DEFAULT_CANDIDATES_PER_EPISODE)
    topics_parse.add_argument("--existing-prompt-file", action="append", default=[], type=Path)
    topics_parse.add_argument("--skip-existing-dedup", action="store_true")
    topics_parse.set_defaults(func=topics.parse)

    topics_report = topics_subparsers.add_parser("report", help="Report topic-generation candidate statistics.")
    topics_report.add_argument("--input-file", type=Path, required=True)
    topics_report.add_argument("--summary-input", type=Path, required=True)
    topics_report.add_argument("--output-file", type=Path, required=True)
    topics_report.set_defaults(func=topics.report)

    topics_category_task = topics_subparsers.add_parser("category-task", help="Build the ResearchRubrics category task.")
    topics_category_task.add_argument("--researchrubrics-path", type=Path, required=True)
    topics_category_task.add_argument("--output-file", type=Path, required=True)
    topics_category_task.add_argument("--category-count", type=int, default=topics.DEFAULT_CATEGORY_COUNT)
    topics_category_task.set_defaults(func=topics.category_task)

    topics_parse_categories = topics_subparsers.add_parser("parse-categories", help="Parse ResearchRubrics category results.")
    topics_parse_categories.add_argument("--input-file", type=Path, required=True)
    topics_parse_categories.add_argument("--output-file", type=Path, required=True)
    topics_parse_categories.add_argument("--summary-output", type=Path, required=True)
    topics_parse_categories.add_argument("--category-count", type=int, default=topics.DEFAULT_CATEGORY_COUNT)
    topics_parse_categories.set_defaults(func=topics.parse_categories)
    return parser


def add_runner_args(parser: argparse.ArgumentParser, *, include_input_file: bool = True) -> None:
    if include_input_file:
        parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--failed-output", type=Path)
    parser.add_argument("--raw-events-dir", type=Path)
    parser.add_argument("--agent-binary", default="pi")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thinking", default=DEFAULT_THINKING)
    parser.add_argument(
        "--system-prompt",
        default="",
        help="Exact Pi system prompt. Defaults to empty string to avoid Pi's coding-assistant default prompt.",
    )
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--agent-state-dir", type=Path)
    parser.add_argument("--extension-path", type=Path, help="Optional Pi extension path to load with -e.")
    parser.add_argument("--extension-cwd", type=Path, help="Working directory for the Pi extension process.")
    parser.add_argument(
        "--extension-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        type=parse_key_value,
        help="Environment variable passed to the Pi extension process. May be repeated.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=13)


def add_assign_input_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-file", type=Path)
    group.add_argument("--input-json")


def add_topics_materialize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episodes", type=int, default=topics.DEFAULT_EPISODES)
    parser.add_argument("--candidates-per-episode", type=int, default=topics.DEFAULT_CANDIDATES_PER_EPISODE)
    parser.add_argument("--max-search-calls", type=int, default=topics.DEFAULT_MAX_SEARCH_CALLS)
    parser.add_argument("--search-topk", type=int, default=topics.DEFAULT_SEARCH_TOPK)
    parser.add_argument("--min-unique-cited-docids", type=int, default=topics.DEFAULT_MIN_UNIQUE_CITED_DOCIDS)
    parser.add_argument("--min-search-calls-per-candidate", type=int, default=topics.DEFAULT_MIN_SEARCH_CALLS_PER_CANDIDATE)
    parser.add_argument("--icl-source", choices=["researchrubrics", "fixed"], default="researchrubrics")
    parser.add_argument("--icl-examples", type=int, default=topics.DEFAULT_ICL_EXAMPLES)
    parser.add_argument("--icl-seed", type=int, default=topics.DEFAULT_ICL_SEED)
    parser.add_argument("--informal-style-probability", type=float, default=topics.DEFAULT_INFORMAL_STYLE_PROBABILITY)
    parser.add_argument("--researchrubrics-path", type=Path)
    parser.add_argument("--topic-categories", type=Path)
    parser.add_argument("--topic-category-seed", type=int, default=topics.DEFAULT_TOPIC_CATEGORY_SEED)
    parser.add_argument("--output-file", type=Path, required=True)


def parse_key_value(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("expected KEY=VALUE")
    key, value = text.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError("environment variable key must not be empty")
    return key, value


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
