# Pi-TREC

Private Pi/Codex-based runner for RAG evaluation prompts.

This repository provides a barebones local-agent execution layer for UMBRELA relevance assessment, Nuggetizer nugget evaluation, and support evaluation. Public inputs and outputs follow the existing evaluator request formats where possible; the internal prompt-task format is only for materialization and debugging.

## Install

```bash
uv sync
```

## Test

```bash
uv run pytest
```

## Runner Defaults

All evaluator commands run prompts through:

```bash
pi --no-tools --no-session --no-skills --no-context-files \
  --no-extensions --no-prompt-templates --no-themes \
  --system-prompt "" --mode json \
  --model openai-codex/gpt-5.5 --thinking minimal @/tmp/.../prompt.txt
```

`--thinking` defaults to `medium`. Pass `--thinking auto` to resolve per model (`minimal` for `gpt-5*`, `medium` otherwise), or pass an explicit level such as `minimal`/`high` to override. `--temperature` is left unset by default (required for `gpt-5*`); set it only for models that accept it.

The rendered prompt is written to a temporary UTF-8 text file and passed to Pi with its `@file` initial-message syntax. This avoids OS command-line argument length limits for long RAG prompts.
The default system prompt is explicitly set to the empty string so the model-facing instruction is the evaluator prompt rather than Pi's coding-assistant system prompt.
Nuggetizer is the exception because the source Nuggetizer templates define real `system_message` values. For Nuggetizer create, score, and assign commands, Pi receives the copied Nuggetizer system message through `--system-prompt`, and `prompt.txt` contains only the copied `prefix_user` prompt. UMBRELA and support evaluation use an empty system prompt because their source prompt templates/reference script do not define a non-empty system role.

Useful shared flags include `--agent-binary`, `--model`, `--thinking`, `--temperature`, `--max-concurrency`, `--timeout-seconds`, `--raw-events-dir`, `--limit`, `--resume`, `--overwrite`, `--dry-run` (select and count tasks without calling Pi), `--cache-dir` (reuse identical prompt results, with a 3-hour TTL; defaults to the gitignored `.cache/pi-trec`), `--no-cache` (disable caching for a run so every task hits inference), and `-v`/`-q` for logging. `create`, `assign`, and `agentic-create` run cells/topics concurrently (bounded by `--max-concurrency`).

## Configuration (CLI or YAML)

Every subcommand is driven by a typed config object (see `src/pi_trec/config.py`). Values come from three layers, later layers winning:

1. dataclass defaults (the values documented above),
2. an optional `--config <file>.yaml`,
3. explicit CLI flags.

So you can keep shared settings in a YAML file and still override individual values on the command line. YAML keys are the snake_case field names (the CLI flag without the leading `--`, dashes as underscores), e.g. `--max-nuggets` is `max_nuggets`:

```yaml
# umbrela-judge.yaml
input_file: examples/umbrela.requests.jsonl
output_file: results/umbrela.judgments.jsonl
model: openai-codex/gpt-5.5
thinking: medium
max_concurrency: 8
prompt_type: bing
```

```bash
# Run entirely from the YAML file:
uv run pi-trec umbrela judge --config examples/configs/umbrela-judge.yaml

# Same file, but override one value for this run:
uv run pi-trec umbrela judge --config examples/configs/umbrela-judge.yaml --thinking high
```

Required fields (such as `input_file`/`output_file`) may be supplied through either the YAML file or CLI flags; a missing required value fails fast with a clear message. Unknown YAML keys are ignored, so one shared file can hold settings for several commands.

## Pyserini Wrapper for Pi Search

Pi-TREC can expose a Pyserini HTTP endpoint as the Pine-compatible `pi-search` `http-json` backend contract:

```bash
uv run pi-trec serve pyserini-wrapper \
  --pyserini-base-url http://127.0.0.1:8081 \
  --pyserini-index msmarco-v2.1-doc-segmented \
  --port 8092 \
  --search-word-limit 512 \
  --read-word-limit 4096 \
  --print-config
```

The wrapper serves:

- `POST /search`: search requests mapped to Pyserini `/v1/<index>/search`.
- `POST /read_document`: document reads mapped to Pyserini `/v1/<index>/doc/<docid>`.
- `GET /pi_search_config`: the `PI_SEARCH_EXTENSION_CONFIG` JSON for the Pi search extension.

Protected Pyserini services read bearer tokens from `PYSERINI_API_TOKEN` by default; override that with `--token-env`. The token is never passed as a flag: set it in your shell or in a local `.env` file (copy `.env.example`, which is the only `.env*` file committed to git). Values already present in the shell environment take precedence over `.env`.

## UMBRELA

Materialize exact UMBRELA prompts without running Pi:

```bash
uv run pi-trec materialize umbrela \
  --input-file examples/umbrela.requests.jsonl \
  --output-file results/umbrela.tasks.jsonl \
  --prompt-type bing
```

Run query-candidate relevance judging:

```bash
uv run pi-trec umbrela judge \
  --input-file examples/umbrela.requests.jsonl \
  --output-file results/umbrela.judgments.jsonl \
  --raw-events-dir results/umbrela.raw-events \
  --include-trace \
  --overwrite
```

The input follows UMBRELA's shared query-candidate shape: `query` plus `candidates`, where candidates may be strings or records with `doc.segment`.

## Nuggetizer

Create and score nuggets from Nuggetizer-style `query` plus `candidates` input:

```bash
uv run pi-trec nuggetizer create \
  --input-file examples/nuggetizer.create.jsonl \
  --output-file results/nuggets.jsonl \
  --raw-events-dir results/nugget-create.raw-events \
  --overwrite
```

Assign nuggets to a direct context:

```bash
uv run pi-trec nuggetizer assign \
  --input-json '{"query":"What is Python used for?","context":"Python is used for web development.","nuggets":[{"text":"Python is used for web development.","importance":"vital"}]}' \
  --output-file results/assignments.jsonl \
  --raw-events-dir results/nugget-assign.raw-events \
  --overwrite
```

Prompt materialization commands are available as `materialize nugget-create`, `materialize nugget-score`, and `materialize nugget-assign`. These rows include both `system_prompt` and `instruction` so the original Nuggetizer system/user role split is visible before execution.

Agentically create nuggets by giving Pi the same search/read-document tool style used by Pine. The input contains `query` plus optional starting `nuggets`; the agent searches the wrapped Pyserini corpus, reads documents, returns an updated nugget list, and Pi-TREC scores that final list with the existing Nuggetizer scorer prompt:

```bash
uv run pi-trec nuggetizer agentic-create \
  --input-file examples/nuggetizer.agentic-create.jsonl \
  --output-file results/agentic-nuggets.jsonl \
  --failed-output results/agentic-nuggets.failed.jsonl \
  --raw-events-dir results/agentic-nuggets.raw-events \
  --extension-path ../research/external/pi-serini/src/extensions/pi_search.ts \
  --extension-cwd ../research/external/pi-serini \
  --extension-env PI_SEARCH_EXTENSION_CONFIG='{"backend":{"kind":"http-json",...}}' \
  --max-nuggets 30 \
  --overwrite
```

Materialize the agentic creator prompt without running Pi:

```bash
uv run pi-trec materialize nugget-agentic-create \
  --input-file examples/nuggetizer.agentic-create.jsonl \
  --output-file results/nugget-agentic-create.tasks.jsonl \
  --max-nuggets 30
```

### Score an assignment

`nuggetizer metrics` scores an assignment file (rows with `qid`, `run_id`, and `nuggets` carrying `importance` + `assignment`) into per-cell, per-run, and label-distribution CSVs. Pass `--reference` (a gold/reference assignment) to also get run-level and topic-run Kendall `tau`. Coverage follows the reference semantics: `support`=1.0, `partial_support`=0.5 (non-strict only), vital metrics restricted to vital nuggets.

```bash
uv run pi-trec nuggetizer metrics \
  --input-file results/assignments.jsonl \
  --output-dir results/metrics \
  --reference data/TREC2024/assignment-nist5/human_gold.jsonl
```

### End-to-end eval

`nuggetizer eval` chains create → assign → metrics. Give it either `--create-input` (generate nuggets from candidates, E2E) or `--nuggets-file` (assignment-only, fixed nuggets), plus `--answers-file` and an output dir; add `--gold` to score against a reference.

```bash
# Assignment-only (fixed nuggets):
uv run pi-trec nuggetizer eval \
  --nuggets-file data/TREC2024/assignment-nist5/nuggets.jsonl \
  --answers-file data/TREC2024/assignment-nist5/answers.jsonl \
  --gold data/TREC2024/assignment-nist5/human_gold.jsonl \
  --output-dir results/TREC2024/assignment-nist5/gpt-5.5 \
  --model openai-codex/gpt-5.5 --thinking minimal

# End-to-end (UMBRELA -> create -> assign):
uv run pi-trec nuggetizer eval \
  --create-input data/TREC2024/e2e-umbrela/create_input.jsonl \
  --answers-file data/TREC2024/e2e-umbrela/answers.jsonl \
  --gold data/TREC2024/e2e-umbrela/human_gold.jsonl \
  --output-dir results/TREC2024/e2e-umbrela/gpt-5.5
```

### TREC adapters

Convert a TREC RAG answer run into the `answers` schema, and join `answers` + `nuggets` (by `qid`) into assign payloads for batch assignment:

```bash
uv run pi-trec materialize trec-answers --input-file run.jsonl --output-file answers.jsonl
uv run pi-trec materialize assign-inputs \
  --answers-file answers.jsonl --nuggets-file nuggets.jsonl --output-file assign_inputs.jsonl
```

### Operate: doctor, validate, cost

```bash
uv run pi-trec doctor --probe              # check pi binary, agent auth, and a model
uv run pi-trec validate --input-file data/TREC2024/assignment-nist5/nuggets.jsonl
uv run pi-trec cost --raw-events-dir results/metrics/../raw-events \
  --input-price 0.25 --output-price 2.0    # USD per 1M tokens (optional)
```

## Rubric (Nuggets → Rubrics)

A ResearchRubrics-style evaluation path that reuses nugget **extraction** but replaces nugget **assignment**: turn a topic's scored nuggets into a weighted, reference-independent rubric, grade each criterion with a ternary verdict against the *full* answer, and score it with the normalized weighted formula

```
S_k = Σ_i (w_i · m_i) / Σ_{i: w_i > 0} w_i      m ∈ {satisfied=1, partially_satisfied=0.5, not_satisfied=0}   (binary collapses partial → 0)
```

Four stages — `author`, `grade`, `score`, and an `eval` orchestrator that chains them. Authoring runs once per topic, independent of any answer; weights are assigned deterministically from each nugget's importance (vital → mandatory `+5`, okay → optional `+2`).

Author a weighted rubric from a nuggets file (`query`, `qid`, `nuggets` with `importance`):

```bash
uv run pi-trec rubric author \
  --input-file path/to/nuggets.jsonl \
  --output-file results/rubric.jsonl \
  --model openai-codex/gpt-5.4-mini --thinking minimal
```

Grade answers against the rubric. `grade` consumes the answers×rubric join `eval` writes as `grade_inputs.jsonl` (criteria are windowed; the whole answer is always shown), or a single inline payload via `--input-json`:

```bash
uv run pi-trec rubric grade \
  --input-file results/grade_inputs.jsonl \
  --output-file results/graded.jsonl \
  --window-size 10 --model openai-codex/gpt-5.4-mini --thinking minimal
```

Score a graded file into ternary + binary `S_k` with per-cell and per-run CSVs; `--reference` (a gold/reference *graded* file) adds the Kendall `tau` between the two graded runs:

```bash
uv run pi-trec rubric score \
  --input-file results/graded.jsonl \
  --output-dir results/scores
```

### End-to-end eval

`rubric eval` chains author → grade → score in one run. Give it either `--nuggets-file` (assignment-only, fixed nuggets), `--create-input` (generate nuggets first, E2E), or a pre-authored `--rubric-file`, plus `--answers-file` and `--output-dir`. It writes `rubric.jsonl`, `grade_inputs.jsonl`, `graded.jsonl`, and a `scores/` dir (`cell_scores.csv`, `run_scores.csv`, `category_failures.csv`, `correlations.csv`).

```bash
# Assignment (fixed nugget pool):
uv run pi-trec rubric eval \
  --nuggets-file path/to/nuggets.jsonl \
  --answers-file path/to/answers.jsonl \
  --output-dir results/rubric/assignment \
  --model openai-codex/gpt-5.4-mini --thinking minimal \
  --window-size 10 --max-concurrency 16

# End-to-end (create → author → grade → score):
uv run pi-trec rubric eval \
  --create-input path/to/create_input.jsonl \
  --answers-file path/to/answers.jsonl \
  --output-dir results/rubric/e2e \
  --model openai-codex/gpt-5.4-mini --thinking minimal --max-nuggets 30
```

To align the rubric run-ranking with the human nugget-coverage ranking, build human coverage with `pi-trec nuggetizer metrics` (Kendall `tau` vs the human assignment) and correlate it against `scores/run_scores.csv` / `scores/cell_scores.csv`.

## Support Evaluation

Run support judgment on pre-resolved statement/citation pairs:

```bash
uv run pi-trec support judge \
  --input-file examples/support.requests.jsonl \
  --output-file results/support.jsonl \
  --raw-events-dir results/support.raw-events \
  --overwrite
```

The support prompt is copied exactly from `trec2024-rag/support_eval/code/support_evaluation_individual_gpt4o.py`, associated with the SIGIR 2025 support evaluation paper: <https://doi.org/10.1145/3726302.3730165>.

## Arena System Comparison

Rank a set of RAG systems with Search Arena-style pairwise LLM judgments. Each
answer file is one system/run, using either the normalized `answers` schema
from `materialize trec-answers` or official TREC RAG rows. Sentence-level
`answer[].text` fields are concatenated into plain answer text for the judge;
citations and references are not rendered in the prompt.

```bash
uv run pi-trec arena compare-all \
  --answers-dir answers/ \
  --output-dir results/arena \
  --model openai-codex/gpt-5.5 \
  --thinking medium \
  --overwrite
```

`--answers-dir` loads all `*.jsonl` files in sorted filename order. For a small
ad hoc run, pass repeated `--answers <file>` flags instead.

For every pair of systems, Pi-TREC compares only shared `qid`s and never mixes
answers from different questions. Assistant A/B display order is randomized per
task from `--seed` and recorded in `judgments.jsonl`, so `[[A]]`/`[[B]]` verdicts
can be mapped back to the original `run_id`. The output directory contains
`tasks.jsonl`, `judgments.jsonl`, `pairwise.csv`, `coverage.csv`,
`leaderboard.csv`, and `raw-events/`. The headline ranking is an Arena-style
rating fit from the pairwise judgments; pairwise preference rates are
diagnostics.

To reduce judge cost, sample a fixed number of shared topics per system pair
before materialization or judging:

```bash
uv run pi-trec arena compare-all \
  --answers-dir answers/ \
  --output-dir results/arena-k5 \
  --sample-topics-per-pair 5 \
  --sampling-seed 13 \
  --model openai-codex/gpt-5.5
```

The same flags work with `materialize arena`; sampling is deterministic per
system pair and never compares answers from different `qid`s.

For a more Chatbot Arena-like sparse comparison graph, sample battles within
each topic instead. This targets a fixed number of battle appearances per
available system per topic:

```bash
uv run pi-trec arena compare-all \
  --answers-dir answers/ \
  --output-dir results/arena-d2 \
  --sample-battles-per-system-per-topic 2 \
  --sampling-seed 13 \
  --model openai-codex/gpt-5.5
```

With 76 systems, `--sample-battles-per-system-per-topic 2` materializes about
76 battles per topic; with 102 systems, it materializes about 102 battles per
topic. Pi-TREC also supports the absolute `--sample-battles-per-topic <n>` flag
for fixed-size experiments. The topic and battle sampling flags are mutually
exclusive.

The judging workflow and stored verdicts are independent of the ranking backend:
`judgments.jsonl` records the same pairwise `[[A]]`, `[[B]]`, and `[[Tie]]`
outcomes, while `leaderboard.csv` converts those outcomes into Arena ratings.
Leaderboard rows contain `rank`, `run_id`, `arena_score`, `n_judgments`, `wins`,
`losses`, and `ties`.

Arena scores are fit with the `arena-rank` package from LMArena
(<https://github.com/lmarena/arena-rank>). Pi-TREC converts each valid
judgment into the package's standard `model_a`, `model_b`, `winner` schema, with
wins as `model_a`/`model_b` and ties as `tie`. `arena-rank` then fits the
Bradley-Terry/logistic paired-comparison model and reports ratings on an
Elo-like Arena scale centered at `1000`, with `400` points corresponding to one
base-10 log-odds unit.

Materialize exact judge prompts without running Pi:

```bash
uv run pi-trec materialize arena \
  --answers-dir answers/ \
  --output-file results/arena.tasks.jsonl
```

## Data

Evaluation inputs and gold for the Assignment and E2E tasks live under `data/`, organized by year (`data/TREC2024/`, `data/TREC2025/`). See `data/README.md` for per-dataset schemas, cell counts, gold type (human vs reference), and provenance.

## Development

```bash
uv sync --group dev      # install dev tools (pytest, ruff, mypy)
uv run pytest            # tests
uv run ruff check src tests
uv run mypy src          # advisory (gradual typing)
pre-commit install       # optional: lint on commit (.pre-commit-config.yaml)
```

CI runs ruff + pytest as required checks and mypy as an advisory job (see `.github/workflows/ci.yml`).
