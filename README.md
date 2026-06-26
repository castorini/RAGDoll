# Pi-TREC

Private Pi/Codex-based runner for TREC RAG evaluation prompts.

Pi-TREC provides local-agent execution for the main TREC RAG evaluation
workflow: generating gold-standard artifacts, scoring submitted RAG answers, and
running optional system-vs-system comparisons. Public inputs and outputs follow
the corresponding UMBRELA, Nuggetizer, and support-evaluation workflows where
possible; materialized prompt-task JSONL files are mainly for debugging and
reproducible runs.

# Generating Gold Standard

## Qrels Generation

Qrels generation uses UMBRELA-style query-passage relevance judging. The input
follows UMBRELA's shared query-candidate shape: `query` plus `candidates`, where
candidates may be strings or records with `doc.segment`.

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

## Rubric Generation

Rubric generation starts from Nuggetizer-style nugget creation and scoring. In
this repo, nuggets are the rubric-like gold-standard units used for isolated
answer scoring. The basic Nuggetizer path creates and scores nuggets from
`query` plus `candidates` input:

```bash
uv run pi-trec nuggetizer create \
  --input-file examples/nuggetizer.create.jsonl \
  --output-file results/nuggets.jsonl \
  --raw-events-dir results/nugget-create.raw-events \
  --overwrite
```

Prompt materialization commands are available as `materialize nugget-create`,
`materialize nugget-score`, and `materialize nugget-assign`. These rows include
both `system_prompt` and `instruction` so the original Nuggetizer system/user
role split is visible before execution.

Agentically create nuggets by giving Pi the same search/read-document tool style
used by Pine. The input contains `query` plus optional starting `nuggets`; the
agent searches the wrapped Pyserini corpus, reads documents, returns an updated
nugget list, and Pi-TREC scores that final list with the existing Nuggetizer
scorer prompt:

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

# System Scoring

## Support Assessment

Support assessment starts from official TREC RAG answer rows. Each JSONL row is
one `(narrative_id, run_id)` answer. The `answer` field contains sentence text
and zero-indexed citation references; `references` maps those citation indices
to document IDs.

```json
{
  "metadata": {
    "team_id": "example-team",
    "run_id": "example-run",
    "type": "automatic",
    "narrative_id": "14",
    "title": "Example topic",
    "narrative": "Example topic narrative."
  },
  "references": ["doc-a"],
  "answer": [
    {
      "text": "Answer sentence.",
      "citations": [0]
    }
  ]
}
```

First resolve the cited document IDs to passage text. With a Pyserini HTTP API:

```bash
uv run pi-trec support resolve-references \
  --input-file answers.jsonl \
  --output-file answers.resolved.jsonl \
  --pyserini-api http://127.0.0.1:8081 \
  --pyserini-index climbmix-400b
```

Or directly with a Pyserini index. The index can be a local Lucene index path
or a Pyserini prebuilt index name:

```bash
uv run pi-trec support resolve-references \
  --input-file answers.jsonl \
  --output-file answers.resolved.jsonl \
  --pyserini-index msmarco-v2.1-doc-segmented
```

The resolved file preserves the original row and adds `topic_id`, `run_id`, and
`segments`, where `segments` maps each reference document ID to the text that
the support judge will read.

```json
{
  "topic_id": "14",
  "run_id": "example-run",
  "references": ["doc-a"],
  "segments": {
    "doc-a": "Full cited passage text."
  },
  "answer": [
    {
      "text": "Answer sentence.",
      "citations": [0]
    }
  ]
}
```

Run support judgment on the resolved answer file:

```bash
uv run pi-trec support judge \
  --input-file answers.resolved.jsonl \
  --output-file judgments.parsed.jsonl \
  --raw-events-dir results/support.raw-events \
  --overwrite
```

This writes one row per judged sentence/citation pair:

```json
{
  "task_id": "example-run:14:s0:c0",
  "statement": "Answer sentence.",
  "citation": "Full cited passage text.",
  "support_label": "FS",
  "raw_output": "Full Support",
  "status": "completed",
  "metadata": {
    "topic_id": "14",
    "run_id": "example-run",
    "sentence_index": 0,
    "citation_index": 0,
    "docid": "doc-a"
  }
}
```

Assemble the parsed judgments back onto the original answer structure:

```bash
uv run pi-trec support assemble \
  --answers-file answers.resolved.jsonl \
  --judgments judgments.parsed.jsonl \
  --output-file support_assignments.jsonl
```

The assignment file is one row per `(topic_id, run_id)`:

```json
{
  "topic_id": "14",
  "narrative_id": "14",
  "run_id": "example-run",
  "sentences": [
    {
      "sentenceID": 0,
      "text": "Answer sentence.",
      "citations": [
        {
          "citationID": 0,
          "reference": "doc-a",
          "support": "2"
        }
      ]
    }
  ]
}
```

Support values are `2` = full support, `1` = partial support, `0` = no support,
and `-1` = missing, failed, or unparseable.

Compute run/topic support metrics:

```bash
uv run pi-trec support metrics \
  --input-file support_assignments.jsonl \
  --output-file support_metrics.jsonl
```

This writes one score row per `(topic_id, run_id)`:

```json
{
  "topic_id": "14",
  "run_id": "example-run",
  "weighted_precision_first_citation": 1.0,
  "weighted_recall_first_citation": 1.0,
  "weighted_precision_all_judged_citations": 1.0,
  "weighted_recall_all_judged_citations": 1.0,
  "hard_precision": 1.0,
  "hard_recall": 1.0,
  "sentences": 1
}
```

Optionally export metric rows:

```bash
uv run pi-trec support metric-rows \
  --input-file support_metrics.jsonl \
  --output-file support_metric_rows.txt
```

```text
example-run 14 weighted_precision_first_citation 1
example-run 14 weighted_recall_first_citation 1
example-run 14 weighted_precision_all_judged_citations 1
example-run 14 weighted_recall_all_judged_citations 1
```

For many resolved answer files, `support summarize` combines assignment
assembly, metric computation, and metric-row export after judgments already
exist:

```bash
uv run pi-trec support summarize \
  --answers-dir answers/ \
  --judgments-root judgments/ \
  --output-dir support_scores/
```

It writes `support_assignments.jsonl`, `support_metrics.jsonl`, and
`support_metric_rows.txt` under the output directory.

The support prompt follows the TREC RAG Task assessor-facing support
instructions, using `Cited Passage`, `Sentence`, and `Sentence Context` fields.

## Isolated Rubric Scoring

Isolated rubric scoring assigns the generated gold nuggets to submitted answers.
For Nuggetizer-style assignment runs, `nuggetizer assign` labels nuggets against
a direct context:

```bash
uv run pi-trec nuggetizer assign \
  --input-json '{"query":"What is Python used for?","context":"Python is used for web development.","nuggets":[{"text":"Python is used for web development.","importance":"vital"}]}' \
  --output-file results/assignments.jsonl \
  --raw-events-dir results/nugget-assign.raw-events \
  --overwrite
```

`nuggetizer metrics` scores an assignment file with `qid`, `run_id`, and
`nuggets` carrying `importance` plus `assignment`. Pass `--reference` to also
get run-level and topic-run Kendall `tau`. Coverage follows the reference
semantics: `support`=1.0, `partial_support`=0.5 (non-strict only), vital metrics
restricted to vital nuggets.

```bash
uv run pi-trec nuggetizer metrics \
  --input-file results/assignments.jsonl \
  --output-dir results/metrics \
  --reference data/TREC2024/assignment-nist5/human_gold.jsonl
```

`nuggetizer eval` chains create -> assign -> metrics. Give it either
`--create-input` (generate nuggets from candidates, E2E) or `--nuggets-file`
(assignment-only, fixed nuggets), plus `--answers-file` and an output directory;
add `--gold` to score against a reference.

```bash
uv run pi-trec nuggetizer eval \
  --nuggets-file data/TREC2024/assignment-nist5/nuggets.jsonl \
  --answers-file data/TREC2024/assignment-nist5/answers.jsonl \
  --gold data/TREC2024/assignment-nist5/human_gold.jsonl \
  --output-dir results/TREC2024/assignment-nist5/gpt-5.5 \
  --model openai-codex/gpt-5.5 --thinking minimal
```

## Pairwise System Comparison

Rank a set of RAG systems with Search Arena-style pairwise LLM judgments. Each
answer file is one system/run, using either the normalized `answers` schema from
`materialize trec-answers` or official TREC RAG rows. Sentence-level
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
task from `--seed` and recorded in `judgments.jsonl`, so `[[A]]`/`[[B]]`
verdicts can be mapped back to the original `run_id`. The output directory
contains `tasks.jsonl`, `judgments.jsonl`, `pairwise.csv`, `coverage.csv`,
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
(<https://github.com/lmarena/arena-rank>). Pi-TREC converts each valid judgment
into the package's standard `model_a`, `model_b`, `winner` schema, with wins as
`model_a`/`model_b` and ties as `tie`. `arena-rank` then fits the
Bradley-Terry/logistic paired-comparison model and reports ratings on an
Elo-like Arena scale centered at `1000`, with `400` points corresponding to one
base-10 log-odds unit.

Materialize exact judge prompts without running Pi:

```bash
uv run pi-trec materialize arena \
  --answers-dir answers/ \
  --output-file results/arena.tasks.jsonl
```

# Shared Setup And Operations

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

`--thinking` defaults to `medium`. Pass `--thinking auto` to resolve per model
(`minimal` for `gpt-5*`, `medium` otherwise), or pass an explicit level such as
`minimal` or `high` to override. `--temperature` is left unset by default
(required for `gpt-5*`); set it only for models that accept it.

The rendered prompt is written to a temporary UTF-8 text file and passed to Pi
with its `@file` initial-message syntax. This avoids OS command-line argument
length limits for long RAG prompts. The default system prompt is explicitly set
to the empty string so the model-facing instruction is the evaluator prompt
rather than Pi's coding-assistant system prompt.

Nuggetizer is the exception because the source Nuggetizer templates define real
`system_message` values. For Nuggetizer create, score, and assign commands, Pi
receives the copied Nuggetizer system message through `--system-prompt`, and
`prompt.txt` contains only the copied `prefix_user` prompt. UMBRELA and support
evaluation use an empty system prompt because their source prompt templates do
not define a non-empty system role.

Useful shared flags include `--agent-binary`, `--model`, `--thinking`,
`--temperature`, `--max-concurrency`, `--timeout-seconds`, `--raw-events-dir`,
`--limit`, `--resume`, `--overwrite`, `--dry-run`, `--cache-dir`, `--no-cache`,
and `-v`/`-q` for logging. `--cache-dir` defaults to the gitignored
`.cache/pi-trec`; `--no-cache` disables caching for a run so every task hits
inference.

## Configuration

Every subcommand is driven by a typed config object in `src/pi_trec/config.py`.
Values come from three layers, later layers winning:

1. dataclass defaults,
2. an optional `--config <file>.yaml`,
3. explicit CLI flags.

YAML keys are the snake_case field names, e.g. `--max-nuggets` is `max_nuggets`:

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
uv run pi-trec umbrela judge --config examples/configs/umbrela-judge.yaml
uv run pi-trec umbrela judge --config examples/configs/umbrela-judge.yaml --thinking high
```

Required fields such as `input_file` and `output_file` may be supplied through
either the YAML file or CLI flags. Unknown YAML keys are ignored, so one shared
file can hold settings for several commands.

## Pyserini Wrapper

Pi-TREC can expose a Pyserini HTTP endpoint as the Pine-compatible `pi-search`
`http-json` backend contract:

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

Protected Pyserini services read bearer tokens from `PYSERINI_API_TOKEN` by
default; override that with `--token-env`. The token is never passed as a flag:
set it in your shell or in a local `.env` file. Values already present in the
shell environment take precedence over `.env`.

## TREC Adapters

Convert a TREC RAG answer run into the `answers` schema, and join `answers` plus
`nuggets` by `qid` into assign payloads for batch assignment:

```bash
uv run pi-trec materialize trec-answers --input-file run.jsonl --output-file answers.jsonl
uv run pi-trec materialize assign-inputs \
  --answers-file answers.jsonl --nuggets-file nuggets.jsonl --output-file assign_inputs.jsonl
```

## Operate

```bash
uv run pi-trec doctor --probe
uv run pi-trec validate --input-file data/TREC2024/assignment-nist5/nuggets.jsonl
uv run pi-trec cost --raw-events-dir results/metrics/../raw-events \
  --input-price 0.25 --output-price 2.0
```

## Data

Evaluation inputs and gold for the Assignment and E2E tasks live under `data/`,
organized by year (`data/TREC2024/`, `data/TREC2025/`). See `data/README.md`
for per-dataset schemas, cell counts, gold type, and provenance.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check src tests
uv run mypy src
pre-commit install
```

CI runs ruff plus pytest as required checks and mypy as an advisory job.
