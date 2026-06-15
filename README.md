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
  --model openai-codex/gpt-5.5 --thinking medium @/tmp/.../prompt.txt
```

The rendered prompt is written to a temporary UTF-8 text file and passed to Pi with its `@file` initial-message syntax. This avoids OS command-line argument length limits for long RAG prompts.
The default system prompt is explicitly set to the empty string so the model-facing instruction is the evaluator prompt rather than Pi's coding-assistant system prompt.
Nuggetizer is the exception because the source Nuggetizer templates define real `system_message` values. For Nuggetizer create, score, and assign commands, Pi receives the copied Nuggetizer system message through `--system-prompt`, and `prompt.txt` contains only the copied `prefix_user` prompt. UMBRELA and support evaluation use an empty system prompt because their source prompt templates/reference script do not define a non-empty system role.

Useful shared flags include `--agent-binary`, `--model`, `--thinking`, `--max-concurrency`, `--timeout-seconds`, `--raw-events-dir`, `--limit`, and `--overwrite`.

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

Protected Pyserini services read bearer tokens from `PYSERINI_API_TOKEN` by default; override that with `--token-env`.

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

## Topic Generation / KARL-Style Synthesis

`pi-trec topics` mimics the previous Pine/KARL workflow from `agentic-search-datasets/dev/KARL_SYNTHESIS_RUNBOOK.md`: build Pine-shaped task rows, run them through Pi with the same search extension style, parse accepted/rejected grounded prompt/reference-answer candidates, and report summary statistics.

The defaults match the Pine workflow: 200 episodes, 1 candidate per episode, 50 search calls per episode, at least 8 distinct search calls per generated candidate, top 20 search results, at least 8 unique cited source documents, 4 in-context examples, in-context seed 13, topic-category seed 13, and informal-style probability 0.25.

Do not start with a full real run. Use offline tests and fake fixtures first, then run exactly one real smoke task before any larger generation job.

Start the Pi-TREC Pyserini wrapper in one terminal:

```bash
cd /store/scratch/rpradeep/castorini-monorepo/research/pi-trec
uv run pi-trec serve pyserini-wrapper \
  --pyserini-base-url http://99.251.12.72:8081 \
  --pyserini-index climbmix-400b \
  --port 8092 \
  --backend-id pyserini-http \
  --default-limit 20 \
  --search-word-limit 512 \
  --read-word-limit 4096 \
  --print-config
```

Copy the printed `PI_SEARCH_EXTENSION_CONFIG=...` value.

Optionally build ResearchRubrics topic categories, matching the old Step 0:

```bash
cd /store/scratch/rpradeep/castorini-monorepo/research/pi-trec
uv run pi-trec topics category-task \
  --researchrubrics-path /store/scratch/rpradeep/castorini-monorepo/agentic-search-datasets/data/raw/researchrubrics/processed_data.jsonl \
  --output-file results/topics/step0-rr-categories.tasks.jsonl \
  --category-count 200

uv run pi-trec topics generate \
  --input-file results/topics/step0-rr-categories.tasks.jsonl \
  --output-file results/topics/step0-rr-categories.results.jsonl \
  --failed-output results/topics/step0-rr-categories.failed.jsonl \
  --raw-events-dir results/topics/step0-rr-categories.raw-events \
  --model openai-codex/gpt-5.5 \
  --thinking medium \
  --max-concurrency 1 \
  --timeout-seconds 1800 \
  --overwrite

uv run pi-trec topics parse-categories \
  --input-file results/topics/step0-rr-categories.results.jsonl \
  --output-file results/topics/rr_top_level_categories.txt \
  --summary-output results/topics/rr_top_level_categories.summary.json \
  --category-count 200
```

Materialize a one-episode smoke task:

```bash
cd /store/scratch/rpradeep/castorini-monorepo/research/pi-trec
uv run pi-trec topics materialize \
  --episodes 1 \
  --candidates-per-episode 1 \
  --max-search-calls 50 \
  --search-topk 20 \
  --min-unique-cited-docids 8 \
  --min-search-calls-per-candidate 8 \
  --icl-source researchrubrics \
  --icl-examples 4 \
  --icl-seed 13 \
  --researchrubrics-path /store/scratch/rpradeep/castorini-monorepo/agentic-search-datasets/data/raw/researchrubrics/processed_data.jsonl \
  --topic-categories results/topics/rr_top_level_categories.txt \
  --topic-category-seed 13 \
  --output-file results/topics/smoke.tasks.jsonl
```

Run the smoke through Pi with the same search extension style previously used through Pine:

```bash
uv run pi-trec topics generate \
  --input-file results/topics/smoke.tasks.jsonl \
  --output-file results/topics/smoke.results.jsonl \
  --failed-output results/topics/smoke.failed.jsonl \
  --raw-events-dir results/topics/smoke.raw-events \
  --extension-path ../external/pi-serini/src/extensions/pi_search.ts \
  --extension-cwd ../external/pi-serini \
  --extension-env PI_SEARCH_EXTENSION_CONFIG='<printed wrapper config>' \
  --model openai-codex/gpt-5.5 \
  --thinking medium \
  --max-concurrency 1 \
  --timeout-seconds 3600 \
  --overwrite
```

Parse and report the smoke output:

```bash
uv run pi-trec topics parse \
  --input-file results/topics/smoke.results.jsonl \
  --output-file results/topics/smoke.candidates.jsonl \
  --rejected-output results/topics/smoke.rejected.jsonl \
  --summary-output results/topics/smoke.summary.json \
  --candidates-per-episode 1

uv run pi-trec topics report \
  --input-file results/topics/smoke.candidates.jsonl \
  --summary-input results/topics/smoke.summary.json \
  --output-file results/topics/smoke.report.md
```

For the full run, use the same workflow with 200 episodes and `--resume`:

```bash
uv run pi-trec topics materialize \
  --episodes 200 \
  --candidates-per-episode 1 \
  --max-search-calls 50 \
  --search-topk 20 \
  --min-unique-cited-docids 8 \
  --min-search-calls-per-candidate 8 \
  --icl-source researchrubrics \
  --icl-examples 4 \
  --icl-seed 13 \
  --researchrubrics-path /store/scratch/rpradeep/castorini-monorepo/agentic-search-datasets/data/raw/researchrubrics/processed_data.jsonl \
  --topic-categories results/topics/rr_top_level_categories.txt \
  --topic-category-seed 13 \
  --output-file results/topics/full.tasks.jsonl

uv run pi-trec topics generate \
  --input-file results/topics/full.tasks.jsonl \
  --output-file results/topics/full.results.jsonl \
  --failed-output results/topics/full.failed.jsonl \
  --raw-events-dir results/topics/full.raw-events \
  --extension-path ../external/pi-serini/src/extensions/pi_search.ts \
  --extension-cwd ../external/pi-serini \
  --extension-env PI_SEARCH_EXTENSION_CONFIG='<printed wrapper config>' \
  --model openai-codex/gpt-5.5 \
  --thinking medium \
  --max-concurrency 4 \
  --timeout-seconds 3600 \
  --resume

uv run pi-trec topics parse \
  --input-file results/topics/full.results.jsonl \
  --output-file results/topics/full.candidates.jsonl \
  --rejected-output results/topics/full.rejected.jsonl \
  --summary-output results/topics/full.summary.json \
  --candidates-per-episode 1

uv run pi-trec topics report \
  --input-file results/topics/full.candidates.jsonl \
  --summary-input results/topics/full.summary.json \
  --output-file results/topics/full.report.md
```
