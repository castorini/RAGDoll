# Reproducing Side-by-Side Arena Runs with RAGDoll

This document reproduces the side-by-side system evaluation pipeline used for
TREC RAG 2024, TREC RAG 2025, and Search Arena 7K. It covers the commands and
expected file layouts only; large answer files, rubric files, raw events, and
result artifacts are intentionally not stored in the repository.

| Benchmark | Native judge | Rubric-guided judge | Produces |
|---|---|---|---|
| TREC RAG 2024/2025 | `ragdoll arena compare-all --prompt-variant rich-human-voter` | `ragdoll arena compare-all --rubric-file ... --prompt-variant coverage-count` | `tasks.jsonl`, `judgments.jsonl`, `pairwise.csv`, `coverage.csv`, `leaderboard.csv` |
| Search Arena 7K | `experiments/search_arena_prompt_search.py --prompt rich-human-voter` | `experiments/search_arena_prompt_search.py --prompt rich-human-voter-rubric --rubric-file ...` | `all-rows.jsonl`, `judgments.jsonl`, `leaderboard.csv`, `summary.csv` |

---

## Step 0: Set Up

```bash
uv sync
uv run ragdoll doctor --probe --model openai-codex/gpt-5.5 --thinking medium
```

The commands below use the Codex/OpenAI OAuth-backed local agent provider via
Pi. If a long run stops because of a transient auth, timeout, or rate-limit
failure, rerun the same command. TREC arena runs use `--resume`, which skips
completed task IDs and retries unfinished/failed tasks.

---

## Step 1: Prepare TREC RAG Answer Inputs

`ragdoll arena compare-all` expects either repeated `--answers <file>` flags or
an `--answers-dir` containing one JSONL file per run/system. Each row can use
the normalized answer schema:

```json
{"run_id": "example-run", "qid": "14", "query": "topic narrative", "answer_text": "system answer"}
```

Official TREC RAG rows are also accepted by the TREC adapter. If you have one
combined answer file, split it into one file per run:

```bash
mkdir -p data/side-by-side/rag25/answers-qrels22
uv run python - <<'PY'
import json
import pathlib
from collections import defaultdict

input_file = pathlib.Path("data/side-by-side/rag25/answers-qrels22.jsonl")
output_dir = pathlib.Path("data/side-by-side/rag25/answers-qrels22")
output_dir.mkdir(parents=True, exist_ok=True)

rows = defaultdict(list)
for line in input_file.open():
    row = json.loads(line)
    rows[str(row["run_id"])].append(row)

for run_id, records in rows.items():
    safe = run_id.replace("/", "_")
    with (output_dir / f"{safe}.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
PY
```

For the reported TREC RAG runs, prepare these answer directories:

| Run | Answer subset |
|---|---|
| RAG24 native | 20 rubric/NIST nugget topics, 45 systems |
| RAG24 rubric-guided | same 20-topic directory as RAG24 native |
| RAG25 native | 22 qrel topics, 76 systems |
| RAG25 rubric-guided | 20 rubric-covered topics, 76 systems |

The RAG25 rubric-guided subset has 20 topics because the GPT-5.5 nugget-rubric
file used for that run did not cover qids `72` and `225`.

---

## Step 2: Prepare Topic Rubrics

Rubric-guided TREC runs use one JSONL row per topic:

```json
{
  "qid": "14",
  "status": "completed",
  "criteria": [
    {
      "text": "A strong answer should cover this important point.",
      "tier": "mandatory",
      "weight": 5,
      "type": "explicit",
      "source_nugget": "n0"
    }
  ]
}
```

The side-by-side judge uses `criteria[].text`, while `tier`, `weight`, `type`,
and `source_nugget` are preserved in the rendered prompt for priority and
provenance.

---

## Step 3: Run TREC RAG Native Side-by-Side

Native TREC runs used GPT-5.5 with medium reasoning and the
`rich-human-voter` prompt variant. No rubric or nugget file is passed.

### RAG25 Native, 22 Qrel Topics

```bash
uv run ragdoll arena compare-all \
  --answers-dir data/side-by-side/rag25/answers-qrels22 \
  --output-dir results/side-by-side/rag25/native-gpt55-medium-rich-human-voter-qrels22 \
  --model openai-codex/gpt-5.5 \
  --thinking medium \
  --prompt-variant rich-human-voter \
  --max-concurrency 24 \
  --timeout-seconds 420 \
  --resume \
  --shuffle \
  --seed 20260702
```

### RAG24 Native, 20 Rubric Topics

```bash
uv run ragdoll arena compare-all \
  --answers-dir data/side-by-side/rag24/answers-rubric20 \
  --output-dir results/side-by-side/rag24/native-gpt55-medium-rich-human-voter-rubric20 \
  --model openai-codex/gpt-5.5 \
  --thinking medium \
  --prompt-variant rich-human-voter \
  --max-concurrency 12 \
  --timeout-seconds 300 \
  --resume \
  --shuffle \
  --seed 20260703
```

---

## Step 4: Run TREC RAG Rubric-Guided Side-by-Side

Rubric-guided TREC runs use the same arena pipeline, plus `--rubric-file` and
the `coverage-count` prompt variant.

### RAG25 Rubric-Guided, 20 Rubric Topics

```bash
uv run ragdoll arena compare-all \
  --answers-dir data/side-by-side/rag25/answers-rubric20 \
  --rubric-file data/side-by-side/rag25/rubric-trec-rag-2025-gpt55-nuggets.jsonl \
  --prompt-variant coverage-count \
  --output-dir results/side-by-side/rag25/rubric-gpt55-medium-coverage-count-rubric20 \
  --model openai-codex/gpt-5.5 \
  --thinking medium \
  --max-concurrency 24 \
  --timeout-seconds 420 \
  --resume \
  --shuffle \
  --seed 20260702
```

### RAG24 Rubric-Guided, 20 Rubric Topics

```bash
uv run ragdoll arena compare-all \
  --answers-dir data/side-by-side/rag24/answers-rubric20 \
  --rubric-file data/side-by-side/rag24/rubric-trec-rag-2024-gpt55-nuggets.jsonl \
  --prompt-variant coverage-count \
  --output-dir results/side-by-side/rag24/rubric-gpt55-medium-coverage-count-rubric20 \
  --model openai-codex/gpt-5.5 \
  --thinking medium \
  --max-concurrency 12 \
  --timeout-seconds 300 \
  --resume \
  --shuffle \
  --seed 20260703
```

If the provider starts returning empty local-agent outputs or rate-limit errors,
lower `--max-concurrency` and rerun the same command with `--resume`.

---

## Step 5: Run Search Arena 7K

Search Arena uses the Hugging Face dataset
`lmarena-ai/search-arena-v1-7k`. The experiment driver filters self-battles by
default, materializes the selected rows, runs the requested prompt(s), fits
arena rankings, and writes correlations against the human Bradley--Terry
leaderboard.

### Native Search Arena

```bash
uv run python experiments/search_arena_prompt_search.py \
  --split all \
  --prompt rich-human-voter \
  --output-dir results/search-arena-prompt-search/gpt55-high-full-rich-human-voter \
  --model openai-codex/gpt-5.5 \
  --thinking high \
  --concurrency 24 \
  --timeout-seconds 900
```

### Rubric-Guided Search Arena

```bash
uv run python experiments/search_arena_prompt_search.py \
  --split all \
  --prompt rich-human-voter-rubric \
  --rubric-file data/search-arena/rubric-arena-7k.jsonl \
  --output-dir results/search-arena-prompt-search/gpt55-high-full-rich-human-voter-rubric \
  --model openai-codex/gpt-5.5 \
  --thinking high \
  --concurrency 24 \
  --timeout-seconds 900
```

The Search Arena driver resumes by reading existing
`<output-dir>/<prompt>/judgments.jsonl`; omit `--overwrite` unless you want to
discard completed judgments and start over.

---

## Step 6: Inspect Outputs

TREC arena output directories contain:

| File | Meaning |
|---|---|
| `tasks.jsonl` | materialized pairwise prompts and metadata |
| `judgments.jsonl` | parsed judge verdicts and preferred run IDs |
| `pairwise.csv` | pair-level win/tie counts |
| `coverage.csv` | shared-topic coverage for each system pair |
| `leaderboard.csv` | Arena ratings from the pairwise judgments |
| `raw-events/` | raw local-agent event logs |

Search Arena output directories contain:

| File | Meaning |
|---|---|
| `all-rows.jsonl` | selected dataset rows for `--split all` |
| `<prompt>/judgments.jsonl` | parsed judge verdicts |
| `<prompt>/leaderboard.csv` | Arena ratings from GPT judgments |
| `summary.csv` | exact agreement and Kendall/Spearman vs human BT rankings |

Quick count checks:

```bash
wc -l results/side-by-side/rag25/native-gpt55-medium-rich-human-voter-qrels22/tasks.jsonl
wc -l results/side-by-side/rag25/native-gpt55-medium-rich-human-voter-qrels22/judgments.jsonl
sed -n '1,20p' results/search-arena-prompt-search/gpt55-high-full-rich-human-voter/summary.csv
```

---

## Step 7: Compute TREC Kendall Tau Against Human Scores

For TREC RAG, compute correlation by joining `leaderboard.csv` to a human
run-score table. The paper uses NIST `strict_vital_score` as the primary human
reference.

The human score CSV should have at least:

```text
run_id,strict_vital_score
example-run,0.42
```

Then run:

```bash
uv run python - \
  results/side-by-side/rag25/native-gpt55-medium-rich-human-voter-qrels22/leaderboard.csv \
  data/side-by-side/rag25/human-strict-vital-run-scores.csv <<'PY'
import csv
import sys
from scipy.stats import kendalltau, spearmanr

leaderboard_path = sys.argv[1]
human_path = sys.argv[2]

with open(leaderboard_path, newline="") as handle:
    arena = {row["run_id"]: float(row["arena_score"]) for row in csv.DictReader(handle)}
with open(human_path, newline="") as handle:
    human = {row["run_id"]: float(row["strict_vital_score"]) for row in csv.DictReader(handle)}

common = sorted(set(arena) & set(human))
xs = [arena[run_id] for run_id in common]
ys = [human[run_id] for run_id in common]
print(f"n={len(common)}")
print(f"kendall_tau_b={kendalltau(xs, ys).statistic:.6f}")
print(f"spearman={spearmanr(xs, ys).statistic:.6f}")
PY
```

Search Arena `summary.csv` already includes Kendall tau and Spearman against the
human Bradley--Terry leaderboard.
