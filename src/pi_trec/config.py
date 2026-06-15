"""Typed configuration objects for every pi-trec subcommand.

This module is the single root of the package import graph: it depends only on
the standard library (plus PyYAML, imported lazily) so that every other module
can import configuration from here without risking an import cycle.

Each subcommand has a config dataclass whose field defaults match the values the
CLI used before this module existed. Configs can be built from a YAML file, from
explicit CLI flags, or both (CLI overrides YAML overrides dataclass defaults) via
:meth:`BaseConfig.from_sources`.
"""

from __future__ import annotations

import dataclasses
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping, Union, get_args, get_origin, get_type_hints

# --- Shared runner/agent defaults (previously in runner.py) -----------------
DEFAULT_MODEL = "openai-codex/gpt-5.5"
DEFAULT_THINKING = "medium"
DEFAULT_PROVIDER = "pi"
DEFAULT_TIMEOUT_SECONDS = 900.0
DEFAULT_MAX_CONCURRENCY = 8
DEFAULT_SYSTEM_PROMPT = ""
DEFAULT_SEED = 13

# --- Nuggetizer windowing defaults (previously in nuggetizer.py) ------------
# castorini AutoNuggetizer windowing defaults: each stage processes at most
# `WINDOW_SIZE` items per LLM call (creator slides over documents; scorer and
# assigner slide over nuggets), and a parse miss is retried up to `MAX_TRIALS`.
DEFAULT_WINDOW_SIZE = 10
DEFAULT_MAX_TRIALS = 4

# --- Topic-generation defaults (previously in topics.py) --------------------
DEFAULT_EPISODES = 200
DEFAULT_CANDIDATES_PER_EPISODE = 1
DEFAULT_MAX_SEARCH_CALLS = 50
DEFAULT_SEARCH_TOPK = 20
DEFAULT_MIN_UNIQUE_CITED_DOCIDS = 8
DEFAULT_MIN_SEARCH_CALLS_PER_CANDIDATE = 8
DEFAULT_ICL_EXAMPLES = 4
DEFAULT_ICL_SEED = 13
DEFAULT_TOPIC_CATEGORY_SEED = 13
DEFAULT_INFORMAL_STYLE_PROBABILITY = 0.25
DEFAULT_CATEGORY_COUNT = 200


@dataclass(frozen=True)
class LocalAgentConfig:
    """Execution settings for a single Pi local-agent invocation."""

    agent_binary: str = "pi"
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    thinking: str = DEFAULT_THINKING
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    agent_state_dir: Path | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    extension_path: Path | None = None
    extension_cwd: Path | None = None
    extension_env: dict[str, str] | None = None


@dataclass(frozen=True)
class PyseriniWrapperConfig:
    """Settings for the Pyserini HTTP wrapper request handlers."""

    pyserini_base_url: str
    pyserini_index: str
    backend_id: str = "pyserini-http"
    default_limit: int = 10
    max_page_size: int = 100
    read_limit: int = 200
    search_word_limit: int = 512
    read_word_limit: int = 4096
    token_env: str = "PYSERINI_API_TOKEN"


# --- Type coercion ----------------------------------------------------------
def _unwrap_optional(hint: Any) -> Any:
    """Return ``T`` for ``Optional[T]`` / ``T | None``; otherwise ``hint``."""
    if get_origin(hint) in (Union, types.UnionType):
        non_none = [arg for arg in get_args(hint) if arg is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return hint


def _coerce(hint: Any, value: Any, *, field_name: str) -> Any:
    """Coerce a YAML/CLI scalar into the field's declared type.

    Handles the cases the CLI and YAML disagree on: paths arrive as strings from
    YAML, ``extension_env`` arrives as a list of ``(key, value)`` pairs from the
    repeated CLI flag, and ``list[Path]`` fields hold strings from YAML.
    """
    if value is None:
        return None
    if field_name == "extension_env":
        return dict(value) if isinstance(value, list) else value
    base = _unwrap_optional(hint)
    if base is Path:
        return value if isinstance(value, Path) else Path(value)
    if get_origin(base) is list:
        args = get_args(base)
        if args and args[0] is Path:
            return [item if isinstance(item, Path) else Path(item) for item in value]
        return list(value)
    return value


class BaseConfig:
    """Shared construction/validation behavior for every config dataclass."""

    #: Field names that must be set (non-``None``) for the command to run.
    _required: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BaseConfig":
        """Build a config from a mapping, coercing types and ignoring unknown keys.

        Unknown keys are ignored so a single shared YAML file can carry settings
        for several commands.
        """
        hints = get_type_hints(cls)
        field_names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        kwargs = {
            key: _coerce(hints[key], value, field_name=key)
            for key, value in data.items()
            if key in field_names
        }
        return cls(**kwargs)  # type: ignore[call-arg]

    @classmethod
    def from_sources(
        cls,
        *,
        file_data: Mapping[str, Any] | None = None,
        cli_overrides: Mapping[str, Any],
    ) -> "BaseConfig":
        """Merge defaults < YAML file < explicit CLI flags into a config."""
        merged: dict[str, Any] = {}
        if file_data:
            merged.update(file_data)
        merged.update(cli_overrides)
        return cls.from_mapping(merged)

    def validate(self) -> None:
        """Raise ``SystemExit`` if any required field is missing."""
        missing = [name for name in self._required if getattr(self, name) is None]
        if missing:
            flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            raise SystemExit(f"missing required configuration: {flags} (set via flag or --config)")


# --- File-transform commands (no Pi execution) ------------------------------
@dataclass
class FileIOConfig(BaseConfig):
    input_file: Path | None = None
    output_file: Path | None = None

    _required: ClassVar[tuple[str, ...]] = ("input_file", "output_file")


@dataclass
class MaterializeUmbrelaConfig(FileIOConfig):
    prompt_type: str = "bing"


@dataclass
class MaterializeNuggetCreateConfig(FileIOConfig):
    max_nuggets: int = 30


@dataclass
class MaterializeNuggetAgenticCreateConfig(FileIOConfig):
    max_nuggets: int = 30


@dataclass
class MaterializeNuggetScoreConfig(FileIOConfig):
    pass


@dataclass
class MaterializeNuggetAssignConfig(FileIOConfig):
    input_json: str | None = None
    assign_mode: str = "support-grade-3"

    _required: ClassVar[tuple[str, ...]] = ("output_file",)

    def validate(self) -> None:
        super().validate()
        if bool(self.input_file) == bool(self.input_json):
            raise SystemExit("nugget assign requires exactly one of --input-file or --input-json")


@dataclass
class MaterializeSupportConfig(FileIOConfig):
    pass


# --- Pi-executing commands --------------------------------------------------
@dataclass
class RunConfig(BaseConfig):
    """Base config for commands that run prompts through the Pi local agent."""

    input_file: Path | None = None
    output_file: Path | None = None
    failed_output: Path | None = None
    raw_events_dir: Path | None = None
    agent_binary: str = "pi"
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    thinking: str = DEFAULT_THINKING
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    agent_state_dir: Path | None = None
    extension_path: Path | None = None
    extension_cwd: Path | None = None
    extension_env: dict[str, str] | None = None
    resume: bool = False
    overwrite: bool = False
    limit: int | None = None
    shuffle: bool = False
    seed: int = DEFAULT_SEED

    _required: ClassVar[tuple[str, ...]] = ("input_file", "output_file")

    def local_agent_config(self) -> LocalAgentConfig:
        return LocalAgentConfig(
            agent_binary=self.agent_binary,
            provider=self.provider,
            model=self.model,
            thinking=self.thinking,
            timeout_seconds=self.timeout_seconds,
            agent_state_dir=self.agent_state_dir,
            system_prompt=self.system_prompt,
            extension_path=self.extension_path,
            extension_cwd=self.extension_cwd,
            extension_env=dict(self.extension_env) if self.extension_env else None,
        )


@dataclass
class LocalAgentRunConfig(RunConfig):
    """`run local-agent`: run an already-materialized task JSONL through Pi."""


@dataclass
class UmbrelaJudgeConfig(RunConfig):
    prompt_type: str = "bing"
    include_trace: bool = False
    redact_prompts: bool = False


@dataclass
class SupportJudgeConfig(RunConfig):
    include_prompt: bool = False


@dataclass
class _NuggetRunConfig(RunConfig):
    window_size: int = DEFAULT_WINDOW_SIZE
    max_trials: int = DEFAULT_MAX_TRIALS
    include_trace: bool = False


@dataclass
class NuggetCreateConfig(_NuggetRunConfig):
    max_nuggets: int = 30


@dataclass
class NuggetAgenticCreateConfig(_NuggetRunConfig):
    max_nuggets: int = 30


@dataclass
class NuggetAssignConfig(_NuggetRunConfig):
    input_json: str | None = None
    assign_mode: str = "support-grade-3"

    _required: ClassVar[tuple[str, ...]] = ("output_file",)

    def validate(self) -> None:
        super().validate()
        if bool(self.input_file) == bool(self.input_json):
            raise SystemExit("nugget assign requires exactly one of --input-file or --input-json")


@dataclass
class TopicsGenerateConfig(RunConfig):
    """`topics generate`: run topic-generation tasks through Pi."""


# --- Topic-generation helper commands ---------------------------------------
@dataclass
class TopicsMaterializeConfig(BaseConfig):
    episodes: int = DEFAULT_EPISODES
    candidates_per_episode: int = DEFAULT_CANDIDATES_PER_EPISODE
    max_search_calls: int = DEFAULT_MAX_SEARCH_CALLS
    search_topk: int = DEFAULT_SEARCH_TOPK
    min_unique_cited_docids: int = DEFAULT_MIN_UNIQUE_CITED_DOCIDS
    min_search_calls_per_candidate: int = DEFAULT_MIN_SEARCH_CALLS_PER_CANDIDATE
    icl_source: str = "researchrubrics"
    icl_examples: int = DEFAULT_ICL_EXAMPLES
    icl_seed: int = DEFAULT_ICL_SEED
    informal_style_probability: float = DEFAULT_INFORMAL_STYLE_PROBABILITY
    researchrubrics_path: Path | None = None
    topic_categories: Path | None = None
    topic_category_seed: int = DEFAULT_TOPIC_CATEGORY_SEED
    output_file: Path | None = None

    _required: ClassVar[tuple[str, ...]] = ("output_file",)


@dataclass
class TopicsParseConfig(FileIOConfig):
    rejected_output: Path | None = None
    summary_output: Path | None = None
    candidates_per_episode: int = DEFAULT_CANDIDATES_PER_EPISODE
    existing_prompt_file: list[Path] = field(default_factory=list)
    skip_existing_dedup: bool = False

    _required: ClassVar[tuple[str, ...]] = (
        "input_file",
        "output_file",
        "rejected_output",
        "summary_output",
    )


@dataclass
class TopicsReportConfig(FileIOConfig):
    summary_input: Path | None = None

    _required: ClassVar[tuple[str, ...]] = ("input_file", "output_file", "summary_input")


@dataclass
class TopicsCategoryTaskConfig(BaseConfig):
    researchrubrics_path: Path | None = None
    output_file: Path | None = None
    category_count: int = DEFAULT_CATEGORY_COUNT

    _required: ClassVar[tuple[str, ...]] = ("researchrubrics_path", "output_file")


@dataclass
class TopicsParseCategoriesConfig(FileIOConfig):
    summary_output: Path | None = None
    category_count: int = DEFAULT_CATEGORY_COUNT

    _required: ClassVar[tuple[str, ...]] = ("input_file", "output_file", "summary_output")


# --- Serve command ----------------------------------------------------------
@dataclass
class PyseriniServeConfig(BaseConfig):
    pyserini_base_url: str | None = None
    pyserini_index: str | None = None
    host: str = "127.0.0.1"
    port: int = 8091
    backend_id: str = "pyserini-http"
    default_limit: int = 10
    max_page_size: int = 100
    read_limit: int = 200
    search_word_limit: int = 512
    read_word_limit: int = 4096
    token_env: str = "PYSERINI_API_TOKEN"
    print_config: bool = False

    _required: ClassVar[tuple[str, ...]] = ("pyserini_base_url", "pyserini_index")

    def wrapper_config(self) -> PyseriniWrapperConfig:
        return PyseriniWrapperConfig(
            pyserini_base_url=self.pyserini_base_url,
            pyserini_index=self.pyserini_index,
            backend_id=self.backend_id,
            default_limit=self.default_limit,
            max_page_size=self.max_page_size,
            read_limit=self.read_limit,
            search_word_limit=self.search_word_limit,
            read_word_limit=self.read_word_limit,
            token_env=self.token_env,
        )


def load_config_file(path: Path) -> dict[str, Any]:
    """Load a YAML config file into a flat mapping of field names to values."""
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: config file must contain a top-level mapping")
    return data
