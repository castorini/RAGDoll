# Reproducing RAG24 Support Metrics

This note records how we reproduced the TREC RAG 2024 support-assessment
metrics used in the RAGDoll-vs-human support plots.

The key conventions are:

- Use the NIST support labels where the assessor was **not** shown the
  prediction.
- Keep only the first citation per answer sentence.
- Exclude `-1` labels from the support metric denominators.
- Compare RAGDoll and human labels on the same first-citation sentence/citation
  scaffold.

## Inputs

NIST source page: <https://trec.nist.gov/data/rag2024.html>

The page lists the RAG24 citation/support assessment files under
`https://trec.nist.gov/data/rag/`. We used:

| File | Use |
|---|---|
| `final.citation_judgments_without_prediction.20241025.jsonl` | Human support labels; assessor not shown prediction |
| `final.citation_judgments_Webassess.20241031.jsonl` | Human support labels; assessor not shown prediction; may contain more than one assessed citation per sentence |

We did **not** use `final.citation_judgments_with_prediction.20241025.jsonl`
for this comparison.

## 1. Download the human support labels

```bash
mkdir -p data/trec-rag-2024/human-support-labels
cd data/trec-rag-2024/human-support-labels

BASE=https://trec.nist.gov/data/rag
curl -L -O $BASE/final.citation_judgments_without_prediction.20241025.jsonl
curl -L -O $BASE/final.citation_judgments_Webassess.20241031.jsonl

cd -
```

Expected raw counts:

```text
final.citation_judgments_without_prediction.20241025.jsonl
  rows: 381
  sentences: 3,406
  citations: 2,840
  support counts: NS=872, PS=731, FS=1,237

final.citation_judgments_Webassess.20241031.jsonl
  rows: 549
  sentences: 4,757
  citations: 4,855
  support counts: -1=110, NS=1,857, PS=1,151, FS=1,737
```

## 2. Aggregate the human files

Concatenate the two not-shown-prediction files into one canonical human label
file:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path

root = Path("data/trec-rag-2024/human-support-labels")
inputs = [
    root / "final.citation_judgments_without_prediction.20241025.jsonl",
    root / "final.citation_judgments_Webassess.20241031.jsonl",
]
out = root / "rag24_human_support_labels.jsonl"

with out.open("w", encoding="utf-8") as writer:
    for path in inputs:
        for line in path.open(encoding="utf-8"):
            if line.strip():
                writer.write(line)

print(out)
PY
```

Expected output:

```text
data/trec-rag-2024/human-support-labels/rag24_human_support_labels.jsonl
  rows: 930
  sentences: 8,163
  citations: 7,695
  support counts: -1=110, NS=2,729, PS=1,882, FS=2,974
```

## 3. Keep only the first citation

The RAG24 official support setup is one citation per sentence. The Webassess file
may include extra assessments, so we keep only the first citation per sentence.

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("data/trec-rag-2024/human-support-labels")
src = root / "rag24_human_support_labels.jsonl"
out = root / "rag24_human_support_labels_first_citation.jsonl"

with src.open(encoding="utf-8") as reader, out.open("w", encoding="utf-8") as writer:
    for line in reader:
        if not line.strip():
            continue
        row = json.loads(line)
        for sentence in row.get("sentences", []):
            citations = sentence.get("citations") or []
            sentence["citations"] = citations[:1]
        writer.write(json.dumps(row, ensure_ascii=False) + "\n")

print(out)
PY
```

Expected output:

```text
data/trec-rag-2024/human-support-labels/rag24_human_support_labels_first_citation.jsonl
  rows: 930
  sentences: 8,163
  citations: 6,792
  support counts: -1=50, NS=2,338, PS=1,652, FS=2,752
```

## 4. Build the empty RAGDoll support input

RAGDoll should judge the same first-citation cases for which the human labels
contain a valid support label. Rows where every first-citation label is `-1` are
omitted from the judge input. The input has no support labels; it keeps the
answer sentence, citation reference, and resolved segment text.

We saved this as:

```text
data/trec-rag-2024/ragdoll-support-input/rag24_empty_ragdoll_support_input_from_human_labels.jsonl
```

Expected shape:

```text
rows: 927
answer sentences: 8,147
first-citation tasks: 6,742
```

The file is directly judge-ready because it already contains `segments`.

## 5. Run RAGDoll support judging

This is the expensive/networked step. We used `openai-codex/gpt-5.5`, the
current support prompt, and one task per valid first citation.

```bash
.venv/bin/python -m pi_trec.cli support judge \
  --input-file data/trec-rag-2024/ragdoll-support-input/rag24_empty_ragdoll_support_input_from_human_labels.jsonl \
  --output-file results/support-ragdoll-rag24/rag24_human-label-derived_gpt5.5-original-support-prompt/judgments.parsed.jsonl \
  --raw-events-dir results/support-ragdoll-rag24/rag24_human-label-derived_gpt5.5-original-support-prompt/raw-events \
  --provider pi \
  --model openai-codex/gpt-5.5 \
  --thinking medium \
  --max-concurrency 8 \
  --timeout-seconds 900 \
  --resume
```

Expected judge output:

```text
judgments.parsed.jsonl: 6,742 completed rows
labels: FS, PS, NS only
```

Then assemble labels onto the judged input:

```bash
.venv/bin/python -m pi_trec.cli support assemble \
  --answers-file data/trec-rag-2024/ragdoll-support-input/rag24_empty_ragdoll_support_input_from_human_labels.jsonl \
  --judgments results/support-ragdoll-rag24/rag24_human-label-derived_gpt5.5-original-support-prompt/judgments.parsed.jsonl \
  --output-file results/support-ragdoll-rag24/rag24_human-label-derived_gpt5.5-original-support-prompt/support_assignments.jsonl
```

The run artifacts were later trimmed to keep only the assignment files and final
metrics. The retained raw RAGDoll assignment file is:

```text
results/support-ragdoll-rag24/rag24_human-label-derived_gpt5.5-original-support-prompt/support_assignments.jsonl
```

## 6. Align RAGDoll output back to the human first-citation scaffold

For metric comparison, we need RAGDoll predictions on the same 930-row
human-label scaffold, including the `-1` holes. This produces:

```text
results/support-ragdoll-rag24/rag24_human-label-derived_gpt5.5-original-support-prompt/support_assignments_with_incomplete_sentences_first_citation.jsonl
```

Expected shape:

```text
rows: 930
sentences: 8,163
citations: 6,792
support counts: -1=50, NS=660, PS=3,469, FS=2,613
```

We verified the aligned RAGDoll assignment file and
`rag24_human_support_labels_first_citation.jsonl` have the same rows, sentence
counts, and citation scaffold; only the support labels differ.

## 7. Compute support metrics

The metric code uses:

```text
weighted: NS=0, PS=0.5, FS=1
hard:     NS=0, PS=0,   FS=1
-1: excluded from denominators
```

Compute human and RAGDoll metrics:

```bash
OUT=results/support-ragdoll-rag24/rag24_human-label-derived_gpt5.5-original-support-prompt/kendall_first_citation_unjudged_excluded
mkdir -p $OUT

.venv/bin/python -m pi_trec.cli support metrics \
  --input-file data/trec-rag-2024/human-support-labels/rag24_human_support_labels_first_citation.jsonl \
  --output-file $OUT/human_support_metrics.jsonl

.venv/bin/python -m pi_trec.cli support metrics \
  --input-file results/support-ragdoll-rag24/rag24_human-label-derived_gpt5.5-original-support-prompt/support_assignments_with_incomplete_sentences_first_citation.jsonl \
  --output-file $OUT/ragdoll_support_metrics.jsonl
```

Expected output:

```text
human_support_metrics.jsonl: 930 rows
ragdoll_support_metrics.jsonl: 930 rows
```

## 8. Generate weighted Kendall plots and CSVs

Run-level aggregation divides by all 22 RAG24 topics. Missing topic rows
contribute 0 only for run-level aggregation. Topic-average Kendall tau ignores
missing run-topic rows within each topic.

```bash
.venv/bin/python tools/plot_support_kendall.py \
  --human-metrics $OUT/human_support_metrics.jsonl \
  --ragdoll-metrics $OUT/ragdoll_support_metrics.jsonl \
  --output-dir $OUT \
  --run-level-denominator 22
```

Expected `kendall_stats.csv`:

```text
weighted_precision:
  runs: 45
  topics: 22
  paired_topic_rows: 930
  missing cells ignored for topic tau: 60
  run-level tau: 0.854545
  topic-avg tau: 0.530218
  all paired rows tau: 0.425476

weighted_recall:
  runs: 45
  topics: 22
  paired_topic_rows: 930
  missing cells ignored for topic tau: 60
  run-level tau: 0.858586
  topic-avg tau: 0.589567
  all paired rows tau: 0.496991
```

Main outputs:

```text
$OUT/human_support_metrics.jsonl
$OUT/ragdoll_support_metrics.jsonl
$OUT/kendall_stats.csv
$OUT/topic_level_taus.csv
$OUT/weighted_precision_paired_topic_rows.csv
$OUT/weighted_precision_run_level_divide_by_22.csv
$OUT/weighted_recall_paired_topic_rows.csv
$OUT/weighted_recall_run_level_divide_by_22.csv
$OUT/ragdoll_vs_nist_weighted_precision.{png,pdf}
$OUT/ragdoll_vs_nist_weighted_recall.{png,pdf}
```

The human/manual RAG24 top weighted-precision run should be:

```text
IITD-IRL.ag_rag_gpt35_expansion_rrf_20  0.793074
```

## 9. Optional: compare support labels directly

For the confusion matrix, count only valid paired first-citation labels where
both human and RAGDoll labels are in `{NS, PS, FS}`. Exclude `-1`.

Expected RAG24 count matrix, with rows as Manual (Human Judge) and columns as
RAGDoll:

```text
Manual \ RAGDoll     NS     PS     FS
NS                  559   1460    319
PS                   56   1162    434
FS                   45    847   1860
```

Expected totals:

```text
valid paired labels: 6,742
exact agreement: 3,581 / 6,742 = 53.115%
excluded -1 labels: 50
missing first citation/prediction: 1,371
```

