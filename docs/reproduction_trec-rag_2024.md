# Reproducing TREC RAG 2024 with RAGDoll

Build the gold standard (qrels generation $\to$ rubric generation), then score
answers three ways: support assessment, nugget rubric scoring, and pairwise
comparison. The paper reports the 884-cell subset (20 `rag24.test` topics that
also have a UMBRELA E2E pool).

| Assessment method | Command | Produces |
|---|---|---|
| Qrels generation | `ragdoll umbrela judge` | relevance-graded pools (qrels) |
| Rubric generation | `ragdoll nuggetizer create` $\to$ `ragdoll rubric author` | nugget rubrics |
| Support assessment | `ragdoll support judge` | support scores |
| Nugget rubric scoring | `ragdoll rubric eval` | rubric scores |
| Pairwise comparison | `ragdoll arena compare-all` | Arena leaderboard |

---

## Step 0: Download the raw NIST data

Source: <https://trec.nist.gov/data/rag2024.html> (files under
`https://trec.nist.gov/data/rag/`).

| Download | Used for |
|---|---|
| `nugget_assignment.20241218.jsonl` | gold-standard source (pool + answers + human assignments) |
| `topics.rag24.test.txt` | topic text |
| `2024-retrieval-qrels.txt` | relevant passages |
| `final.citation_judgments_with_prediction.20241025.jsonl` | human support reference |

Not on NIST, only for the optional methods: MS MARCO v2.1 corpus + UMBRELA
passage qrels (<https://trec-rag.github.io/annoucements/umbrela-qrels/>) for the
E2E create-input; participant run submissions (carrying `references`) for support
assessment and pairwise comparison.

### Step 0.1: Download

```bash
mkdir -p downloads/2024 && cd downloads/2024
BASE=https://trec.nist.gov/data/rag
curl -sO $BASE/topics.rag24.test.txt
curl -sO $BASE/nugget_assignment.20241218.jsonl
curl -sO $BASE/2024-retrieval-qrels.txt
curl -sO $BASE/final.citation_judgments_with_prediction.20241025.jsonl
cd -
```

### Step 0.2: Process into the RAGDoll layout

`nugget_assignment.20241218.jsonl` holds every assessed `(qid, run_id)` cell; the
three Assignment-task files derive from it. This subsets to the 884-cell set:

```bash
python3 - <<'PY'
import json, pathlib
SRC = "downloads/2024/nugget_assignment.20241218.jsonl"
OUT = pathlib.Path("data/TREC2024/assignment"); OUT.mkdir(parents=True, exist_ok=True)
# 20 topics that also have a UMBRELA E2E pool (drop this filter to keep all 56).
QIDS = {"2024-105741","2024-121840","2024-127266","2024-141577","2024-145979",
        "2024-152259","2024-152817","2024-158743","2024-18963","2024-213978",
        "2024-214467","2024-215952","2024-216592","2024-224926","2024-22886",
        "2024-35269","2024-42497","2024-43983","2024-44060","2024-88894"}
rows = [json.loads(l) for l in open(SRC)]
rows = [r for r in rows if r["qid"] in QIDS]

def dump(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

dump(OUT / "human_gold.jsonl", rows)  # raw rows: nuggets keep `assignment`, row keeps `run_id`
dump(OUT / "answers.jsonl",
     ({"run_id": r["run_id"], "qid": r["qid"], "topic_id": r["qid"], "query": r["query"],
       "topic": r["query"], "response_length": r.get("response_length"),
       "answer_text": r["answer_text"]} for r in rows))
pool = {}
for r in rows:
    pool.setdefault(r["qid"], {"query": r["query"], "qid": r["qid"],
        "nuggets": [{"text": n["text"], "importance": n["importance"]} for n in r["nuggets"]]})
dump(OUT / "nuggets.jsonl", pool.values())
print(f"wrote {len(rows)} cells, {len(pool)} topics -> {OUT}")
PY
```

Output: `wrote 884 cells, 20 topics` $\to$
`data/TREC2024/assignment/{nuggets,answers,human_gold}.jsonl`.

<details>
<summary><strong>Optional: E2E create-input</strong></summary>

One row per topic of UMBRELA-passed passages with text:
`{query, candidates:[{docid, doc:{segment, ...}}]}`. Take the UMBRELA passage
qrels at grade $\ge 2$ and join each `docid` to its MS MARCO v2.1 passage text
$\to$ `data/TREC2024/e2e-umbrela/create_input.jsonl`. Copy `answers.jsonl` and
`human_gold.jsonl` alongside it.

</details>

---

## Step 1: Set up

```bash
uv sync
cp .env.example .env          # CASTORINI_API_TOKEN: only for support assessment + E2E create-input
uv run ragdoll doctor --probe
```

Defaults: model `openai-codex/gpt-5.5`, `--thinking minimal`, cache in
`.cache/ragdoll` (`--no-cache` to disable). Add `--dry-run` to count tasks
without calling Pi.

---

# Gold standard construction

## Qrels generation (UMBRELA)

```bash
uv run ragdoll umbrela judge \
  --input-file examples/umbrela.requests.jsonl \
  --output-file results/TREC2024/umbrela.judgments.jsonl \
  --raw-events-dir results/TREC2024/umbrela.raw-events \
  --prompt-type bing --include-trace --overwrite
```

Output: one graded row per candidate; keep grade $\ge 2$ for the relevant pool.
`data/TREC2024/e2e-umbrela/create_input.jsonl` is this pool already filtered.

## Rubric generation

Nugget creation (skip for the fixed pool and use
`data/TREC2024/assignment/nuggets.jsonl`):

```bash
uv run ragdoll nuggetizer create \
  --input-file data/TREC2024/e2e-umbrela/create_input.jsonl \
  --output-file results/TREC2024/nuggets.jsonl \
  --overwrite
```

Author the rubric:

```bash
uv run ragdoll rubric author \
  --input-file data/TREC2024/assignment/nuggets.jsonl \
  --output-file results/TREC2024/rubric.jsonl \
  --overwrite
```

Output: one row per topic with weighted `criteria` (`type`, `tier`, `weight`,
`source_nugget`).

---

# System scoring

## Support assessment

Needs the raw run submissions (with `references`); the flattened `answers.jsonl`
drops `references`.

```bash
uv run ragdoll support resolve-references \
  --input-file runs/2024/example-run.jsonl \
  --output-file results/TREC2024/support/answers.resolved.jsonl \
  --api-base-url http://api.castorini.uwaterloo.ca \
  --index msmarco-v2.1-doc-segmented

uv run ragdoll support judge \
  --input-file results/TREC2024/support/answers.resolved.jsonl \
  --output-file results/TREC2024/support/judgments.parsed.jsonl \
  --raw-events-dir results/TREC2024/support/raw-events --overwrite

uv run ragdoll support assemble \
  --answers-file results/TREC2024/support/answers.resolved.jsonl \
  --judgments results/TREC2024/support/judgments.parsed.jsonl \
  --output-file results/TREC2024/support/support_assignments.jsonl

uv run ragdoll support metrics \
  --input-file results/TREC2024/support/support_assignments.jsonl \
  --output-file results/TREC2024/support/support_metrics.jsonl
```

Output: one row per `(topic_id, run_id)` with weighted/hard precision and recall.
Human reference `final.citation_judgments_with_prediction.20241025.jsonl` has the
same schema as `support assemble`, so score it through `support metrics` too.

## Nugget rubric scoring

Fixed pool (author $\to$ grade $\to$ score in one command):

```bash
uv run ragdoll rubric eval \
  --nuggets-file data/TREC2024/assignment/nuggets.jsonl \
  --answers-file data/TREC2024/assignment/answers.jsonl \
  --output-dir results/TREC2024/rubric-55 \
  --model openai-codex/gpt-5.5 --thinking minimal
```

E2E (each model creates its own nuggets):

```bash
uv run ragdoll rubric eval \
  --create-input data/TREC2024/e2e-umbrela/create_input.jsonl \
  --answers-file data/TREC2024/e2e-umbrela/answers.jsonl \
  --output-dir results/TREC2024/rubric-55-e2e \
  --model openai-codex/gpt-5.5 --thinking minimal
```

Output: `rubric.jsonl`, `graded.jsonl`,
`scores/{run_scores,cell_scores,correlations,category_failures}.csv`.
`run_scores.csv` is the headline (per-run `ternary_score`, `binary_score`);
`ternary_vs_binary` run $\tau$ = 0.917 (gpt-5.5), 0.903 (gpt-5.4-mini).

<details>
<summary><strong>Companion method: AutoNuggetizer 3-grade assignment</strong></summary>

Assign each gold nugget `support | partial_support | not_support` vs the NIST
human assignment (detailed write-up in
[`TREC2024_repro.md`](TREC2024_repro.md)):

```bash
uv run ragdoll nuggetizer eval \
  --nuggets-file data/TREC2024/assignment/nuggets.jsonl \
  --answers-file data/TREC2024/assignment/answers.jsonl \
  --gold data/TREC2024/assignment/human_gold.jsonl \
  --output-dir results/TREC2024/assignment/gpt-5.5 \
  --model openai-codex/gpt-5.5 --thinking minimal
```

Agreement (gpt-5.5, minimal): V_strict run $\tau$ 0.770, A_strict run $\tau$ 0.788.

</details>

## Pairwise answer comparisons

Arena needs one file per run; split the combined answers:

```bash
mkdir -p runs/2024/answers-by-run
uv run python -c "import json,collections,pathlib; \
rows=collections.defaultdict(list); \
[rows[json.loads(l)['run_id']].append(l) for l in open('data/TREC2024/assignment/answers.jsonl')]; \
[pathlib.Path(f'runs/2024/answers-by-run/{r}.jsonl').write_text(''.join(v)) for r,v in rows.items()]"
```

Compare all pairs:

```bash
uv run ragdoll arena compare-all \
  --answers-dir runs/2024/answers-by-run/ \
  --output-dir results/TREC2024/arena \
  --model openai-codex/gpt-5.5 --thinking medium --overwrite
```

Cap cost by sampling (optional): add `--sample-topics-per-pair 5 --sampling-seed 13`.

Output: `judgments.jsonl`, `pairwise.csv`, `coverage.csv`, `leaderboard.csv`
(Arena ratings: `rank`, `arena_score`, `wins`, `losses`, `ties`).

---

## Cost

```bash
uv run ragdoll cost --raw-events-dir results/TREC2024/rubric-55/raw-events \
  --input-price 0.25 --output-price 2.0
```

## Artifacts

`data/` and `results/` are gitignored (local only):

- Inputs: `data/TREC2024/assignment/{nuggets,answers,human_gold}.jsonl`,
  `data/TREC2024/e2e-umbrela/{create_input,answers,human_gold}.jsonl`.
- Rubric scoring: `results/TREC2024/rubric-{54mini,55}` and `...-e2e`.
- Support / pairwise: `results/TREC2024/{support,arena}`.
