import json
import os
import sys
from pathlib import Path

import pytest

from pi_trec.cli import main
from pi_trec.config import (
    DEFAULT_MAX_TRIALS,
    DEFAULT_WINDOW_SIZE,
    LocalAgentConfig,
    NuggetAssignConfig,
    NuggetCreateConfig,
    PyseriniServeConfig,
    PyseriniWrapperConfig,
    RunConfig,
    UmbrelaJudgeConfig,
    load_config_file,
    load_env_file,
)


def test_defaults_match_legacy_values() -> None:
    run = RunConfig()
    assert run.agent_binary == "pi"
    assert run.provider == "pi"
    assert run.model == "openai-codex/gpt-5.5"
    assert run.thinking == "medium"
    assert run.system_prompt == ""
    assert run.timeout_seconds == 900.0
    assert run.max_concurrency == 8
    assert run.seed == 13
    assert run.resume is False and run.overwrite is False and run.shuffle is False

    nugget = NuggetCreateConfig()
    assert nugget.max_nuggets == 30
    assert nugget.window_size == DEFAULT_WINDOW_SIZE == 10
    assert nugget.max_trials == DEFAULT_MAX_TRIALS == 4
    assert nugget.include_trace is False

    assert UmbrelaJudgeConfig().prompt_type == "bing"


def test_from_mapping_coerces_paths_and_extension_env() -> None:
    config = RunConfig.from_mapping(
        {
            "input_file": "in.jsonl",
            "output_file": "out.jsonl",
            "agent_state_dir": "/tmp/state",
            "extension_env": [("A", "1"), ("B", "2")],
            "unknown_key": "ignored",
        }
    )
    assert config.input_file == Path("in.jsonl")
    assert config.output_file == Path("out.jsonl")
    assert config.agent_state_dir == Path("/tmp/state")
    assert config.extension_env == {"A": "1", "B": "2"}
    # Unknown keys are ignored so one shared YAML can serve several commands.
    assert not hasattr(config, "unknown_key")


def test_from_mapping_accepts_yaml_dict_extension_env() -> None:
    config = RunConfig.from_mapping({"extension_env": {"A": "1"}})
    assert config.extension_env == {"A": "1"}


def test_from_sources_cli_overrides_yaml_overrides_defaults() -> None:
    config = UmbrelaJudgeConfig.from_sources(
        file_data={"model": "yaml-model", "prompt_type": "basic", "thinking": "low"},
        cli_overrides={"prompt_type": "bing"},
    )
    assert config.prompt_type == "bing"  # CLI wins
    assert config.model == "yaml-model"  # YAML wins over default
    assert config.thinking == "low"  # YAML wins over default
    assert config.provider == "pi"  # untouched default


def test_local_agent_config_round_trip() -> None:
    config = RunConfig(model="m", thinking="low", extension_env={"K": "V"})
    agent = config.local_agent_config()
    assert isinstance(agent, LocalAgentConfig)
    assert agent.model == "m"
    assert agent.thinking == "low"
    assert agent.extension_env == {"K": "V"}


def test_validate_reports_missing_required() -> None:
    with pytest.raises(SystemExit, match="--input-file"):
        RunConfig(output_file=Path("out.jsonl")).validate()


def test_assign_validate_requires_exactly_one_input() -> None:
    with pytest.raises(SystemExit, match="exactly one"):
        NuggetAssignConfig(output_file=Path("out.jsonl")).validate()
    with pytest.raises(SystemExit, match="exactly one"):
        NuggetAssignConfig(
            output_file=Path("out.jsonl"), input_file=Path("in.jsonl"), input_json="{}"
        ).validate()
    # Exactly one is fine.
    NuggetAssignConfig(output_file=Path("out.jsonl"), input_json="{}").validate()


def test_pyserini_serve_config_builds_wrapper_config() -> None:
    serve = PyseriniServeConfig(pyserini_base_url="http://up", pyserini_index="idx", max_page_size=50)
    wrapper = serve.wrapper_config()
    assert isinstance(wrapper, PyseriniWrapperConfig)
    assert wrapper.pyserini_base_url == "http://up"
    assert wrapper.pyserini_index == "idx"
    assert wrapper.max_page_size == 50


def test_load_config_file_reads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "run.yaml"
    config_path.write_text("model: custom\nmax_concurrency: 3\n", encoding="utf-8")
    assert load_config_file(config_path) == {"model": "custom", "max_concurrency": 3}


def test_load_config_file_rejects_non_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="top-level mapping"):
        load_config_file(config_path)


def test_cli_runs_from_yaml_config_alone(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "requests.jsonl"
    output_path = tmp_path / "tasks.jsonl"
    config_path = tmp_path / "materialize.yaml"
    input_path.write_text('{"query":"q","candidates":["p"]}\n', encoding="utf-8")
    config_path.write_text(
        f"input_file: {input_path}\noutput_file: {output_path}\nprompt_type: basic\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["pi-trec", "materialize", "umbrela", "--config", str(config_path)])
    main()
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["evaluator"] == "umbrela"
    assert "Query: q" in row["instruction"]
    assert row["metadata"]["prompt_type"] == "basic"


def test_cli_flag_overrides_yaml_config(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "requests.jsonl"
    output_path = tmp_path / "tasks.jsonl"
    config_path = tmp_path / "materialize.yaml"
    input_path.write_text('{"query":"q","candidates":["p"]}\n', encoding="utf-8")
    config_path.write_text(
        f"input_file: {input_path}\noutput_file: {output_path}\nprompt_type: basic\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["pi-trec", "materialize", "umbrela", "--config", str(config_path), "--prompt-type", "bing"],
    )
    main()
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["metadata"]["prompt_type"] == "bing"


def test_cli_missing_required_config_exits(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["pi-trec", "materialize", "umbrela", "--output-file", str(tmp_path / "o.jsonl")])
    with pytest.raises(SystemExit, match="--input-file"):
        main()


def test_load_env_file_sets_missing_var(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PYSERINI_API_TOKEN", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text('# token\nPYSERINI_API_TOKEN="secret"\n', encoding="utf-8")
    load_env_file(env_path)
    assert os.environ["PYSERINI_API_TOKEN"] == "secret"


def test_load_env_file_does_not_override_shell_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PYSERINI_API_TOKEN", "from-shell")
    env_path = tmp_path / ".env"
    env_path.write_text("PYSERINI_API_TOKEN=from-file\n", encoding="utf-8")
    load_env_file(env_path)
    assert os.environ["PYSERINI_API_TOKEN"] == "from-shell"


def test_load_env_file_missing_is_noop(tmp_path: Path) -> None:
    load_env_file(tmp_path / "absent.env")
