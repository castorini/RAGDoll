"""Command-line interface for pi-trec.

Every subcommand builds a typed config object (see :mod:`pi_trec.config`). Values
come from dataclass defaults, then an optional ``--config`` YAML file, then any
explicit CLI flags (later sources win). Flags use ``argparse.SUPPRESS`` defaults
so the parsed namespace contains only what the user actually passed, which keeps
the YAML/CLI merge unambiguous.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import inspect
from argparse import SUPPRESS
from pathlib import Path

from pi_trec import nuggetizer, pyserini_wrapper, support, topics, umbrela
from pi_trec.config import (
    BaseConfig,
    LocalAgentRunConfig,
    MaterializeNuggetAgenticCreateConfig,
    MaterializeNuggetAssignConfig,
    MaterializeNuggetCreateConfig,
    MaterializeNuggetScoreConfig,
    MaterializeSupportConfig,
    MaterializeUmbrelaConfig,
    NuggetAgenticCreateConfig,
    NuggetAssignConfig,
    NuggetCreateConfig,
    PyseriniServeConfig,
    SupportJudgeConfig,
    TopicsCategoryTaskConfig,
    TopicsGenerateConfig,
    TopicsMaterializeConfig,
    TopicsParseCategoriesConfig,
    TopicsParseConfig,
    TopicsReportConfig,
    UmbrelaJudgeConfig,
    load_config_file,
)
from pi_trec.runner import run_task_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run internal prompt tasks through Pi.")
    run_subparsers = run.add_subparsers(dest="run_command", required=True)
    local_agent = run_subparsers.add_parser("local-agent", help="Run task JSONL through a local Pi agent.")
    add_runner_args(local_agent)
    finish(local_agent, config_cls=LocalAgentRunConfig, handler=run_task_rows)

    serve = subparsers.add_parser("serve", help="Serve helper endpoints for Pi RAG evaluation runs.")
    serve_subparsers = serve.add_subparsers(dest="serve_command", required=True)
    pyserini = serve_subparsers.add_parser(
        "pyserini-wrapper",
        help="Wrap a Pyserini HTTP endpoint as the pi-search http-json backend contract.",
    )
    pyserini.add_argument("--pyserini-base-url", default=SUPPRESS)
    pyserini.add_argument("--pyserini-index", default=SUPPRESS)
    pyserini.add_argument("--host", default=SUPPRESS)
    pyserini.add_argument("--port", type=int, default=SUPPRESS)
    pyserini.add_argument("--backend-id", default=SUPPRESS)
    pyserini.add_argument("--default-limit", type=int, default=SUPPRESS)
    pyserini.add_argument("--max-page-size", type=int, default=SUPPRESS)
    pyserini.add_argument("--read-limit", type=int, default=SUPPRESS)
    pyserini.add_argument("--search-word-limit", type=int, default=SUPPRESS)
    pyserini.add_argument("--read-word-limit", type=int, default=SUPPRESS)
    pyserini.add_argument("--token-env", default=SUPPRESS)
    pyserini.add_argument("--print-config", action="store_true", default=SUPPRESS)
    finish(pyserini, config_cls=PyseriniServeConfig, handler=pyserini_wrapper.serve_pyserini_wrapper)

    materialize = subparsers.add_parser("materialize", help="Materialize evaluator prompts without running Pi.")
    materialize_subparsers = materialize.add_subparsers(dest="materialize_command", required=True)
    materialize_umbrela = materialize_subparsers.add_parser("umbrela", help="Materialize UMBRELA prompt tasks.")
    materialize_umbrela.add_argument("--input-file", type=Path, default=SUPPRESS)
    materialize_umbrela.add_argument("--output-file", type=Path, default=SUPPRESS)
    materialize_umbrela.add_argument("--prompt-type", choices=["bing", "basic"], default=SUPPRESS)
    finish(materialize_umbrela, config_cls=MaterializeUmbrelaConfig, handler=umbrela.materialize)
    materialize_nugget_create = materialize_subparsers.add_parser("nugget-create", help="Materialize Nuggetizer create prompts.")
    materialize_nugget_create.add_argument("--input-file", type=Path, default=SUPPRESS)
    materialize_nugget_create.add_argument("--output-file", type=Path, default=SUPPRESS)
    materialize_nugget_create.add_argument("--max-nuggets", type=int, default=SUPPRESS)
    finish(materialize_nugget_create, config_cls=MaterializeNuggetCreateConfig, handler=nuggetizer.materialize_create)
    materialize_nugget_agentic_create = materialize_subparsers.add_parser(
        "nugget-agentic-create",
        help="Materialize Nuggetizer agentic create prompts.",
    )
    materialize_nugget_agentic_create.add_argument("--input-file", type=Path, default=SUPPRESS)
    materialize_nugget_agentic_create.add_argument("--output-file", type=Path, default=SUPPRESS)
    materialize_nugget_agentic_create.add_argument("--max-nuggets", type=int, default=SUPPRESS)
    finish(
        materialize_nugget_agentic_create,
        config_cls=MaterializeNuggetAgenticCreateConfig,
        handler=nuggetizer.materialize_agentic_create,
    )
    materialize_nugget_score = materialize_subparsers.add_parser("nugget-score", help="Materialize Nuggetizer score prompts.")
    materialize_nugget_score.add_argument("--input-file", type=Path, default=SUPPRESS)
    materialize_nugget_score.add_argument("--output-file", type=Path, default=SUPPRESS)
    finish(materialize_nugget_score, config_cls=MaterializeNuggetScoreConfig, handler=nuggetizer.materialize_score)
    materialize_nugget_assign = materialize_subparsers.add_parser("nugget-assign", help="Materialize Nuggetizer assign prompts.")
    add_assign_input_args(materialize_nugget_assign)
    materialize_nugget_assign.add_argument("--output-file", type=Path, default=SUPPRESS)
    materialize_nugget_assign.add_argument("--assign-mode", choices=["support-grade-3", "support-grade-2"], default=SUPPRESS)
    finish(materialize_nugget_assign, config_cls=MaterializeNuggetAssignConfig, handler=nuggetizer.materialize_assign)
    materialize_support = materialize_subparsers.add_parser("support", help="Materialize support-evaluation prompts.")
    materialize_support.add_argument("--input-file", type=Path, default=SUPPRESS)
    materialize_support.add_argument("--output-file", type=Path, default=SUPPRESS)
    finish(materialize_support, config_cls=MaterializeSupportConfig, handler=support.materialize)

    umbrela_parser = subparsers.add_parser("umbrela", help="Run UMBRELA-compatible relevance judging.")
    umbrela_subparsers = umbrela_parser.add_subparsers(dest="umbrela_command", required=True)
    umbrela_judge = umbrela_subparsers.add_parser("judge", help="Judge query-candidate relevance through Pi.")
    add_runner_args(umbrela_judge)
    umbrela_judge.add_argument("--prompt-type", choices=["bing", "basic"], default=SUPPRESS)
    umbrela_judge.add_argument("--include-trace", action="store_true", default=SUPPRESS)
    umbrela_judge.add_argument("--redact-prompts", action="store_true", default=SUPPRESS)
    finish(umbrela_judge, config_cls=UmbrelaJudgeConfig, handler=umbrela.judge)

    nuggetizer_parser = subparsers.add_parser("nuggetizer", help="Run Nuggetizer-compatible prompts through Pi.")
    nuggetizer_subparsers = nuggetizer_parser.add_subparsers(dest="nuggetizer_command", required=True)
    nugget_create = nuggetizer_subparsers.add_parser("create", help="Create and score nuggets through Pi.")
    add_runner_args(nugget_create)
    nugget_create.add_argument("--max-nuggets", type=int, default=SUPPRESS)
    add_nugget_window_args(nugget_create)
    nugget_create.add_argument("--include-trace", action="store_true", default=SUPPRESS)
    finish(nugget_create, config_cls=NuggetCreateConfig, handler=nuggetizer.create)
    nugget_agentic_create = nuggetizer_subparsers.add_parser(
        "agentic-create",
        help="Create and score nuggets with Pi search/read-document tools.",
    )
    add_runner_args(nugget_agentic_create)
    nugget_agentic_create.add_argument("--max-nuggets", type=int, default=SUPPRESS)
    add_nugget_window_args(nugget_agentic_create)
    nugget_agentic_create.add_argument("--include-trace", action="store_true", default=SUPPRESS)
    finish(nugget_agentic_create, config_cls=NuggetAgenticCreateConfig, handler=nuggetizer.agentic_create)
    nugget_assign = nuggetizer_subparsers.add_parser("assign", help="Assign nuggets through Pi.")
    add_runner_args(nugget_assign, include_input_file=False)
    add_assign_input_args(nugget_assign)
    nugget_assign.add_argument("--assign-mode", choices=["support-grade-3", "support-grade-2"], default=SUPPRESS)
    add_nugget_window_args(nugget_assign)
    nugget_assign.add_argument("--include-trace", action="store_true", default=SUPPRESS)
    finish(nugget_assign, config_cls=NuggetAssignConfig, handler=nuggetizer.assign)

    support_parser = subparsers.add_parser("support", help="Run support evaluation through Pi.")
    support_subparsers = support_parser.add_subparsers(dest="support_command", required=True)
    support_judge = support_subparsers.add_parser("judge", help="Judge statement-citation support through Pi.")
    add_runner_args(support_judge)
    support_judge.add_argument("--include-prompt", action="store_true", default=SUPPRESS)
    finish(support_judge, config_cls=SupportJudgeConfig, handler=support.judge)

    topics_parser = subparsers.add_parser("topics", help="Run KARL-style topic generation through Pi.")
    topics_subparsers = topics_parser.add_subparsers(dest="topics_command", required=True)

    topics_materialize = topics_subparsers.add_parser("materialize", help="Materialize Pine-compatible topic-generation tasks.")
    add_topics_materialize_args(topics_materialize)
    finish(topics_materialize, config_cls=TopicsMaterializeConfig, handler=topics.materialize)

    topics_generate = topics_subparsers.add_parser("generate", help="Run topic-generation tasks through Pi.")
    add_runner_args(topics_generate)
    finish(topics_generate, config_cls=TopicsGenerateConfig, handler=topics.generate)

    topics_parse = topics_subparsers.add_parser("parse", help="Parse topic-generation results into candidates.")
    topics_parse.add_argument("--input-file", type=Path, default=SUPPRESS)
    topics_parse.add_argument("--output-file", type=Path, default=SUPPRESS)
    topics_parse.add_argument("--rejected-output", type=Path, default=SUPPRESS)
    topics_parse.add_argument("--summary-output", type=Path, default=SUPPRESS)
    topics_parse.add_argument("--candidates-per-episode", type=int, default=SUPPRESS)
    topics_parse.add_argument("--existing-prompt-file", action="append", default=SUPPRESS, type=Path)
    topics_parse.add_argument("--skip-existing-dedup", action="store_true", default=SUPPRESS)
    finish(topics_parse, config_cls=TopicsParseConfig, handler=topics.parse)

    topics_report = topics_subparsers.add_parser("report", help="Report topic-generation candidate statistics.")
    topics_report.add_argument("--input-file", type=Path, default=SUPPRESS)
    topics_report.add_argument("--summary-input", type=Path, default=SUPPRESS)
    topics_report.add_argument("--output-file", type=Path, default=SUPPRESS)
    finish(topics_report, config_cls=TopicsReportConfig, handler=topics.report)

    topics_category_task = topics_subparsers.add_parser("category-task", help="Build the ResearchRubrics category task.")
    topics_category_task.add_argument("--researchrubrics-path", type=Path, default=SUPPRESS)
    topics_category_task.add_argument("--output-file", type=Path, default=SUPPRESS)
    topics_category_task.add_argument("--category-count", type=int, default=SUPPRESS)
    finish(topics_category_task, config_cls=TopicsCategoryTaskConfig, handler=topics.category_task)

    topics_parse_categories = topics_subparsers.add_parser("parse-categories", help="Parse ResearchRubrics category results.")
    topics_parse_categories.add_argument("--input-file", type=Path, default=SUPPRESS)
    topics_parse_categories.add_argument("--output-file", type=Path, default=SUPPRESS)
    topics_parse_categories.add_argument("--summary-output", type=Path, default=SUPPRESS)
    topics_parse_categories.add_argument("--category-count", type=int, default=SUPPRESS)
    finish(topics_parse_categories, config_cls=TopicsParseCategoriesConfig, handler=topics.parse_categories)
    return parser


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML config file of field-name/value pairs. Explicit CLI flags override its values.",
    )


def finish(parser: argparse.ArgumentParser, *, config_cls: type[BaseConfig], handler) -> None:
    """Attach the shared ``--config`` flag and bind the config class + handler."""
    add_config_arg(parser)
    parser.set_defaults(config_cls=config_cls, handler=handler)


def add_runner_args(parser: argparse.ArgumentParser, *, include_input_file: bool = True) -> None:
    if include_input_file:
        parser.add_argument("--input-file", type=Path, default=SUPPRESS)
    parser.add_argument("--output-file", type=Path, default=SUPPRESS)
    parser.add_argument("--failed-output", type=Path, default=SUPPRESS)
    parser.add_argument("--raw-events-dir", type=Path, default=SUPPRESS)
    parser.add_argument("--agent-binary", default=SUPPRESS)
    parser.add_argument("--provider", default=SUPPRESS)
    parser.add_argument("--model", default=SUPPRESS)
    parser.add_argument("--thinking", default=SUPPRESS)
    parser.add_argument(
        "--system-prompt",
        default=SUPPRESS,
        help="Exact Pi system prompt. Defaults to empty string to avoid Pi's coding-assistant default prompt.",
    )
    parser.add_argument("--max-concurrency", type=int, default=SUPPRESS)
    parser.add_argument("--timeout-seconds", type=float, default=SUPPRESS)
    parser.add_argument("--agent-state-dir", type=Path, default=SUPPRESS)
    parser.add_argument("--extension-path", type=Path, default=SUPPRESS, help="Optional Pi extension path to load with -e.")
    parser.add_argument("--extension-cwd", type=Path, default=SUPPRESS, help="Working directory for the Pi extension process.")
    parser.add_argument(
        "--extension-env",
        action="append",
        default=SUPPRESS,
        metavar="KEY=VALUE",
        type=parse_key_value,
        help="Environment variable passed to the Pi extension process. May be repeated.",
    )
    parser.add_argument("--resume", action="store_true", default=SUPPRESS)
    parser.add_argument("--overwrite", action="store_true", default=SUPPRESS)
    parser.add_argument("--limit", type=int, default=SUPPRESS)
    parser.add_argument("--shuffle", action="store_true", default=SUPPRESS)
    parser.add_argument("--seed", type=int, default=SUPPRESS)


def add_nugget_window_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--window-size",
        type=int,
        default=SUPPRESS,
        help="Items per LLM call: documents for create, nuggets for score/assign (reference windowing).",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=SUPPRESS,
        help="Maximum attempts per window before falling back on a parse miss.",
    )


def add_assign_input_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input-file", type=Path, default=SUPPRESS)
    group.add_argument("--input-json", default=SUPPRESS)


def add_topics_materialize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episodes", type=int, default=SUPPRESS)
    parser.add_argument("--candidates-per-episode", type=int, default=SUPPRESS)
    parser.add_argument("--max-search-calls", type=int, default=SUPPRESS)
    parser.add_argument("--search-topk", type=int, default=SUPPRESS)
    parser.add_argument("--min-unique-cited-docids", type=int, default=SUPPRESS)
    parser.add_argument("--min-search-calls-per-candidate", type=int, default=SUPPRESS)
    parser.add_argument("--icl-source", choices=["researchrubrics", "fixed"], default=SUPPRESS)
    parser.add_argument("--icl-examples", type=int, default=SUPPRESS)
    parser.add_argument("--icl-seed", type=int, default=SUPPRESS)
    parser.add_argument("--informal-style-probability", type=float, default=SUPPRESS)
    parser.add_argument("--researchrubrics-path", type=Path, default=SUPPRESS)
    parser.add_argument("--topic-categories", type=Path, default=SUPPRESS)
    parser.add_argument("--topic-category-seed", type=int, default=SUPPRESS)
    parser.add_argument("--output-file", type=Path, default=SUPPRESS)


def parse_key_value(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("expected KEY=VALUE")
    key, value = text.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError("environment variable key must not be empty")
    return key, value


def build_config(args: argparse.Namespace) -> BaseConfig:
    """Merge dataclass defaults, an optional YAML file, and explicit CLI flags."""
    config_cls: type[BaseConfig] = args.config_cls
    file_data = load_config_file(args.config) if getattr(args, "config", None) else None
    field_names = {field.name for field in dataclasses.fields(config_cls)}
    cli_overrides = {key: value for key, value in vars(args).items() if key in field_names}
    return config_cls.from_sources(file_data=file_data, cli_overrides=cli_overrides)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = build_config(args)
    config.validate()
    handler = args.handler
    if inspect.iscoroutinefunction(handler):
        asyncio.run(handler(config))
    else:
        handler(config)


if __name__ == "__main__":
    main()
