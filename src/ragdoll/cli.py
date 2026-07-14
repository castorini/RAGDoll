"""Command-line interface for ragdoll."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import inspect
import logging
from argparse import SUPPRESS
from pathlib import Path

from ragdoll import (
    arena,
    castorini_wrapper,
    nuggetizer,
    rubric,
    support,
    trec_io,
    umbrela,
)
from ragdoll import (
    cost as cost_mod,
)
from ragdoll import (
    doctor as doctor_mod,
)
from ragdoll import (
    validate as validate_mod,
)
from ragdoll.config import (
    ArenaCompareAllConfig,
    BaseConfig,
    CastoriniServeConfig,
    CostConfig,
    DoctorConfig,
    LocalAgentRunConfig,
    MaterializeArenaConfig,
    MaterializeAssignInputsConfig,
    MaterializeNuggetAgenticCreateConfig,
    MaterializeNuggetAssignConfig,
    MaterializeNuggetCreateConfig,
    MaterializeNuggetScoreConfig,
    MaterializeSupportConfig,
    MaterializeTrecAnswersConfig,
    MaterializeUmbrelaConfig,
    NuggetAgenticCreateConfig,
    NuggetAssignConfig,
    NuggetCreateConfig,
    NuggetEvalConfig,
    NuggetMetricsConfig,
    RubricAuthorConfig,
    RubricEvalConfig,
    RubricGradeConfig,
    RubricScoreConfig,
    SupportAssembleConfig,
    SupportJudgeConfig,
    SupportMetricRowsConfig,
    SupportMetricsConfig,
    SupportResolveReferencesConfig,
    SupportSummarizeConfig,
    UmbrelaJudgeConfig,
    ValidateConfig,
    VisualizeAlignmentConfig,
    load_config_file,
    load_env_file,
)
from ragdoll.runner import run_task_rows
from ragdoll.visual import alignment as visual_alignment


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
    castorini = serve_subparsers.add_parser(
        "castorini-wrapper",
        help="Wrap a search API endpoint as the pi-search http-json backend contract.",
    )
    castorini.add_argument("--api-base-url", default=SUPPRESS)
    castorini.add_argument("--index", default=SUPPRESS)
    castorini.add_argument("--host", default=SUPPRESS)
    castorini.add_argument("--port", type=int, default=SUPPRESS)
    castorini.add_argument("--backend-id", default=SUPPRESS)
    castorini.add_argument("--default-limit", type=int, default=SUPPRESS)
    castorini.add_argument("--max-page-size", type=int, default=SUPPRESS)
    castorini.add_argument("--read-limit", type=int, default=SUPPRESS)
    castorini.add_argument("--search-word-limit", type=int, default=SUPPRESS)
    castorini.add_argument("--read-word-limit", type=int, default=SUPPRESS)
    castorini.add_argument("--token-env", default=SUPPRESS)
    castorini.add_argument("--print-config", action="store_true", default=SUPPRESS)
    finish(castorini, config_cls=CastoriniServeConfig, handler=castorini_wrapper.serve_castorini_wrapper)

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
    materialize_trec_answers = materialize_subparsers.add_parser(
        "trec-answers", help="Convert a TREC RAG answer run into the answers JSONL schema."
    )
    materialize_trec_answers.add_argument("--input-file", type=Path, default=SUPPRESS)
    materialize_trec_answers.add_argument("--output-file", type=Path, default=SUPPRESS)
    materialize_trec_answers.add_argument("--run-id", default=SUPPRESS)
    finish(materialize_trec_answers, config_cls=MaterializeTrecAnswersConfig, handler=trec_io.materialize_trec_answers)
    materialize_assign_inputs = materialize_subparsers.add_parser(
        "assign-inputs", help="Join answers + nuggets (by qid) into assign payloads."
    )
    materialize_assign_inputs.add_argument("--answers-file", type=Path, default=SUPPRESS)
    materialize_assign_inputs.add_argument("--nuggets-file", type=Path, default=SUPPRESS)
    materialize_assign_inputs.add_argument("--output-file", type=Path, default=SUPPRESS)
    finish(materialize_assign_inputs, config_cls=MaterializeAssignInputsConfig, handler=trec_io.materialize_assign_inputs)
    materialize_arena = materialize_subparsers.add_parser("arena", help="Materialize arena system-comparison prompts.")
    materialize_arena.add_argument("--answers", type=Path, action="append", default=SUPPRESS)
    materialize_arena.add_argument("--answers-dir", type=Path, default=SUPPRESS)
    materialize_arena.add_argument("--output-file", type=Path, default=SUPPRESS)
    materialize_arena.add_argument("--seed", type=int, default=SUPPRESS)
    materialize_arena.add_argument("--sample-topics-per-pair", type=int, default=SUPPRESS)
    materialize_arena.add_argument("--sample-battles-per-topic", type=int, default=SUPPRESS)
    materialize_arena.add_argument("--sample-battles-per-system-per-topic", type=float, default=SUPPRESS)
    materialize_arena.add_argument("--sampling-seed", type=int, default=SUPPRESS)
    materialize_arena.add_argument("--rubric-file", "--rubrics-file", dest="rubrics_file", type=Path, default=SUPPRESS)
    finish(materialize_arena, config_cls=MaterializeArenaConfig, handler=arena.materialize)

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
    nugget_metrics = nuggetizer_subparsers.add_parser(
        "metrics", help="Score an assignment file (coverage + Kendall tau vs a reference)."
    )
    nugget_metrics.add_argument("--input-file", type=Path, default=SUPPRESS)
    nugget_metrics.add_argument("--output-dir", type=Path, default=SUPPRESS)
    nugget_metrics.add_argument("--reference", type=Path, default=SUPPRESS, help="Gold/reference assignment for correlation.")
    add_logging_args(nugget_metrics)
    finish(nugget_metrics, config_cls=NuggetMetricsConfig, handler=nuggetizer.compute_metrics)
    nugget_eval = nuggetizer_subparsers.add_parser(
        "eval", help="End-to-end: create (optional) -> assign -> metrics in one run."
    )
    add_runner_args(nugget_eval, include_input_file=False)
    nugget_eval.add_argument("--create-input", type=Path, default=SUPPRESS, help="Candidates to CREATE nuggets from (E2E).")
    nugget_eval.add_argument("--nuggets-file", type=Path, default=SUPPRESS, help="Fixed nuggets to assign (assignment-only).")
    nugget_eval.add_argument("--answers-file", type=Path, default=SUPPRESS, help="Answers (topic x run) to assign.")
    nugget_eval.add_argument("--gold", type=Path, default=SUPPRESS, help="Reference assignment for correlation.")
    nugget_eval.add_argument("--output-dir", type=Path, default=SUPPRESS)
    nugget_eval.add_argument("--assign-mode", choices=["support-grade-3", "support-grade-2"], default=SUPPRESS)
    nugget_eval.add_argument("--max-nuggets", type=int, default=SUPPRESS)
    add_nugget_window_args(nugget_eval)
    nugget_eval.add_argument("--include-trace", action="store_true", default=SUPPRESS)
    finish(nugget_eval, config_cls=NuggetEvalConfig, handler=nuggetizer.eval_pipeline)

    rubric_parser = subparsers.add_parser("rubric", help="Run ResearchRubrics-style evaluation through Pi.")
    rubric_subparsers = rubric_parser.add_subparsers(dest="rubric_command", required=True)
    rubric_author = rubric_subparsers.add_parser("author", help="Author a weighted rubric from a nuggets file.")
    add_runner_args(rubric_author)
    add_nugget_window_args(rubric_author)
    rubric_author.add_argument("--include-trace", action="store_true", default=SUPPRESS)
    finish(rubric_author, config_cls=RubricAuthorConfig, handler=rubric.author)
    rubric_grade = rubric_subparsers.add_parser("grade", help="Ternary-grade answers against a rubric through Pi.")
    add_runner_args(rubric_grade, include_input_file=False)
    add_assign_input_args(rubric_grade)
    add_nugget_window_args(rubric_grade)
    rubric_grade.add_argument("--include-trace", action="store_true", default=SUPPRESS)
    finish(rubric_grade, config_cls=RubricGradeConfig, handler=rubric.grade)
    rubric_score = rubric_subparsers.add_parser(
        "score", help="Score a graded rubric file (ternary + binary, Kendall tau vs a reference)."
    )
    rubric_score.add_argument("--input-file", type=Path, default=SUPPRESS)
    rubric_score.add_argument("--output-dir", type=Path, default=SUPPRESS)
    rubric_score.add_argument("--reference", type=Path, default=SUPPRESS, help="Gold/reference graded file for correlation.")
    add_logging_args(rubric_score)
    finish(rubric_score, config_cls=RubricScoreConfig, handler=rubric.compute_scores)
    rubric_eval = rubric_subparsers.add_parser(
        "eval", help="End-to-end: rubric (authored or fixed) -> grade -> score in one run."
    )
    add_runner_args(rubric_eval, include_input_file=False)
    rubric_eval.add_argument("--create-input", type=Path, default=SUPPRESS, help="Candidates to CREATE nuggets from, then author (E2E).")
    rubric_eval.add_argument("--nuggets-file", type=Path, default=SUPPRESS, help="Fixed nuggets to author a rubric from.")
    rubric_eval.add_argument("--rubric-file", type=Path, default=SUPPRESS, help="Pre-authored rubric to grade against.")
    rubric_eval.add_argument("--answers-file", type=Path, default=SUPPRESS, help="Answers (topic x run) to grade.")
    rubric_eval.add_argument("--gold", type=Path, default=SUPPRESS, help="Reference graded file for correlation.")
    rubric_eval.add_argument("--output-dir", type=Path, default=SUPPRESS)
    rubric_eval.add_argument("--max-nuggets", type=int, default=SUPPRESS)
    add_nugget_window_args(rubric_eval)
    rubric_eval.add_argument("--include-trace", action="store_true", default=SUPPRESS)
    finish(rubric_eval, config_cls=RubricEvalConfig, handler=rubric.rubric_eval_pipeline)

    doctor_parser = subparsers.add_parser("doctor", help="Check the Pi binary, agent auth state, and optionally a model.")
    doctor_parser.add_argument("--agent-binary", default=SUPPRESS)
    doctor_parser.add_argument("--model", default=SUPPRESS)
    doctor_parser.add_argument("--thinking", default=SUPPRESS)
    doctor_parser.add_argument("--agent-state-dir", type=Path, default=SUPPRESS)
    doctor_parser.add_argument("--timeout-seconds", type=float, default=SUPPRESS)
    doctor_parser.add_argument("--probe", action="store_true", default=SUPPRESS, help="Also send a trivial prompt to the model.")
    add_logging_args(doctor_parser)
    finish(doctor_parser, config_cls=DoctorConfig, handler=doctor_mod.doctor)

    validate_parser = subparsers.add_parser("validate", help="Schema-check an input JSONL before a long run.")
    validate_parser.add_argument("--input-file", type=Path, default=SUPPRESS)
    validate_parser.add_argument("--kind", choices=["auto", *validate_mod.KINDS], default=SUPPRESS)
    add_logging_args(validate_parser)
    finish(validate_parser, config_cls=ValidateConfig, handler=validate_mod.validate)

    cost_parser = subparsers.add_parser("cost", help="Total token usage from a raw-events dir and price it.")
    cost_parser.add_argument("--raw-events-dir", type=Path, default=SUPPRESS)
    cost_parser.add_argument("--output-file", type=Path, default=SUPPRESS, help="Optional per-task usage CSV.")
    cost_parser.add_argument("--model", default=SUPPRESS)
    cost_parser.add_argument("--input-price", type=float, default=SUPPRESS, help="USD per 1M input tokens.")
    cost_parser.add_argument("--output-price", type=float, default=SUPPRESS, help="USD per 1M output tokens.")
    add_logging_args(cost_parser)
    finish(cost_parser, config_cls=CostConfig, handler=cost_mod.cost)

    support_parser = subparsers.add_parser("support", help="Run support evaluation through Pi.")
    support_subparsers = support_parser.add_subparsers(dest="support_command", required=True)
    support_judge = support_subparsers.add_parser("judge", help="Judge statement-citation support through Pi.")
    add_runner_args(support_judge)
    support_judge.add_argument("--include-prompt", action="store_true", default=SUPPRESS)
    finish(support_judge, config_cls=SupportJudgeConfig, handler=support.judge)
    support_resolve = support_subparsers.add_parser(
        "resolve-references", help="Resolve RAG answer references to passage text for support judging."
    )
    support_resolve.add_argument("--input-file", type=Path, default=SUPPRESS)
    support_resolve.add_argument("--output-file", type=Path, default=SUPPRESS)
    support_resolve.add_argument(
        "--api-base-url",
        default=SUPPRESS,
        help="Search API base URL, e.g. http://api.castorini.uwaterloo.ca.",
    )
    support_resolve.add_argument("--index", default=SUPPRESS)
    support_resolve.add_argument("--read-limit", type=int, default=SUPPRESS)
    support_resolve.add_argument("--read-word-limit", type=int, default=SUPPRESS)
    support_resolve.add_argument("--token-env", default=SUPPRESS)
    finish(support_resolve, config_cls=SupportResolveReferencesConfig, handler=support.resolve_references)
    support_metrics = support_subparsers.add_parser(
        "metrics", help="Compute weighted/hard support precision and recall by topic/run."
    )
    support_metrics.add_argument("--input-file", type=Path, default=SUPPRESS)
    support_metrics.add_argument("--output-file", type=Path, default=SUPPRESS)
    finish(support_metrics, config_cls=SupportMetricsConfig, handler=support.compute_metrics)
    support_metric_rows = support_subparsers.add_parser(
        "metric-rows", help="Convert support metrics JSONL to run topic metric score rows."
    )
    support_metric_rows.add_argument("--input-file", type=Path, default=SUPPRESS)
    support_metric_rows.add_argument("--output-file", type=Path, default=SUPPRESS)
    support_metric_rows.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        default=SUPPRESS,
        choices=[
            "weighted_precision_first_citation",
            "weighted_recall_first_citation",
            "weighted_precision_all_judged_citations",
            "weighted_recall_all_judged_citations",
            "hard_precision",
            "hard_recall",
        ],
        help="Metric column to emit; repeat to control order. Defaults to first/all weighted precision and recall.",
    )
    finish(support_metric_rows, config_cls=SupportMetricRowsConfig, handler=support.write_metric_rows)
    support_assemble = support_subparsers.add_parser(
        "assemble", help="Build human-style support assignment JSONL from answers and parsed judgments."
    )
    support_assemble.add_argument("--answers-file", type=Path, default=SUPPRESS)
    support_assemble.add_argument("--judgments", type=Path, action="append", default=SUPPRESS)
    support_assemble.add_argument("--output-file", type=Path, default=SUPPRESS)
    support_assemble.add_argument("--run-id", default=SUPPRESS, help="Fallback run ID when absent from answers.")
    finish(support_assemble, config_cls=SupportAssembleConfig, handler=support.assemble)
    support_summarize = support_subparsers.add_parser(
        "summarize", help="Create support assignments, metrics JSONL, and metric rows for runfiles."
    )
    summarize_inputs = support_summarize.add_mutually_exclusive_group()
    summarize_inputs.add_argument("--answers", type=Path, action="append", default=SUPPRESS)
    summarize_inputs.add_argument("--answers-dir", type=Path, default=SUPPRESS)
    support_summarize.add_argument("--judgments-root", type=Path, default=SUPPRESS)
    support_summarize.add_argument("--output-dir", type=Path, default=SUPPRESS)
    support_summarize.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        default=SUPPRESS,
        choices=[
            "weighted_precision_first_citation",
            "weighted_recall_first_citation",
            "weighted_precision_all_judged_citations",
            "weighted_recall_all_judged_citations",
            "hard_precision",
            "hard_recall",
        ],
        help="Metric column to emit in support_metric_rows.txt; repeat to control order.",
    )
    finish(support_summarize, config_cls=SupportSummarizeConfig, handler=support.summarize)

    arena_parser = subparsers.add_parser("arena", help="Run Search Arena-style system comparisons.")
    arena_subparsers = arena_parser.add_subparsers(dest="arena_command", required=True)
    arena_compare_all = arena_subparsers.add_parser("compare-all", help="Rank systems from all pairwise arena battles.")
    add_arena_compare_args(arena_compare_all)
    finish(arena_compare_all, config_cls=ArenaCompareAllConfig, handler=arena.compare_all)

    visualize = subparsers.add_parser("visualize", help="Render evaluation figures.")
    visualize_subparsers = visualize.add_subparsers(dest="visualize_command", required=True)
    alignment = visualize_subparsers.add_parser(
        "alignment", help="Scatter a system vs reference nugget assignment with Kendall tau."
    )
    alignment.add_argument("--system", type=Path, default=SUPPRESS, help="System assignment JSONL (x axis).")
    alignment.add_argument("--reference", type=Path, default=SUPPRESS, help="Reference assignment JSONL (y axis).")
    alignment.add_argument("--output-file", type=Path, default=SUPPRESS)
    alignment.add_argument("--metric", choices=["V_strict", "A_strict", "V", "A"], default=SUPPRESS)
    alignment.add_argument("--x-label", default=SUPPRESS)
    alignment.add_argument("--y-label", default=SUPPRESS)
    alignment.add_argument("--compact", action="store_true", default=SUPPRESS)
    add_logging_args(alignment)
    finish(alignment, config_cls=VisualizeAlignmentConfig, handler=visual_alignment.render_alignment)
    return parser


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML config file of field-name/value pairs. Explicit CLI flags override its values.",
    )


def finish(parser: argparse.ArgumentParser, *, config_cls: type[BaseConfig], handler) -> None:
    """Add ``--config`` and bind the config class and handler to the subparser."""
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=SUPPRESS,
        help="Sampling temperature. Unset by default (required for gpt-5* models).",
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--cache-dir",
        type=Path,
        default=SUPPRESS,
        help="Reuse identical prompt results from this dir. Defaults to .cache/ragdoll.",
    )
    cache_group.add_argument(
        "--no-cache",
        action="store_true",
        default=SUPPRESS,
        help="Disable caching for this run so every task hits inference.",
    )
    parser.add_argument("--resume", action="store_true", default=SUPPRESS)
    parser.add_argument("--overwrite", action="store_true", default=SUPPRESS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=SUPPRESS,
        help="Select and count tasks without calling Pi.",
    )
    parser.add_argument("--limit", type=int, default=SUPPRESS)
    parser.add_argument("--shuffle", action="store_true", default=SUPPRESS)
    parser.add_argument("--seed", type=int, default=SUPPRESS)
    add_logging_args(parser)


def add_arena_compare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--answers", type=Path, action="append", default=SUPPRESS)
    parser.add_argument("--answers-dir", type=Path, default=SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=SUPPRESS)
    parser.add_argument("--rubric-file", "--rubrics-file", dest="rubrics_file", type=Path, default=SUPPRESS)
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
    parser.add_argument("--temperature", type=float, default=SUPPRESS)
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--cache-dir", type=Path, default=SUPPRESS)
    cache_group.add_argument("--no-cache", action="store_true", default=SUPPRESS)
    parser.add_argument("--resume", action="store_true", default=SUPPRESS)
    parser.add_argument("--overwrite", action="store_true", default=SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", default=SUPPRESS)
    parser.add_argument("--limit", type=int, default=SUPPRESS)
    parser.add_argument("--shuffle", action="store_true", default=SUPPRESS)
    parser.add_argument("--seed", type=int, default=SUPPRESS)
    parser.add_argument("--sample-topics-per-pair", type=int, default=SUPPRESS)
    parser.add_argument("--sample-battles-per-topic", type=int, default=SUPPRESS)
    parser.add_argument("--sample-battles-per-system-per-topic", type=float, default=SUPPRESS)
    parser.add_argument("--sampling-seed", type=int, default=SUPPRESS)
    add_logging_args(parser)


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true", default=SUPPRESS, help="Verbose (DEBUG) logging.")
    group.add_argument("-q", "--quiet", action="store_true", default=SUPPRESS, help="Quiet (WARNING) logging.")


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
    if getattr(args, "no_cache", False):
        cli_overrides["cache_dir"] = None
    return config_cls.from_sources(file_data=file_data, cli_overrides=cli_overrides)


def setup_logging(args: argparse.Namespace) -> None:
    """Configure root logging from ``--verbose``/``--quiet`` (default INFO)."""
    if getattr(args, "verbose", False):
        level = logging.DEBUG
    elif getattr(args, "quiet", False):
        level = logging.WARNING
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    load_env_file()
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args)
    config = build_config(args)
    config.validate()
    handler = args.handler
    if inspect.iscoroutinefunction(handler):
        asyncio.run(handler(config))
    else:
        handler(config)


if __name__ == "__main__":
    main()
