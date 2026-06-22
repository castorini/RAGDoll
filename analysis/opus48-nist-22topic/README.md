# Opus-4.8 × PiTREC NIST support prompt — 22-topic human-set validation

Validates PiTREC's support-assessment prompt (`src/pi_trec/support/prompts.py::SUPPORT_EVAL_PROMPT`,
the official TREC-RAG NIST `Cited Passage` / `Sentence` / `Sentence Context` form) by judging the
TREC-RAG 2024 **22 "from-scratch" human-reference topics** with **Claude Opus 4.8 (medium effort)**
and comparing against the human labels and the other judges.

## Result (N = 6,693 pairs)

| Judge (identical 22-topic human reference) | Agreement | Cohen's κ |
| --- | ---: | ---: |
| **Opus-4.8 — PiTREC NIST prompt (this run)** | **59.9%** | **0.394** |
| Opus-4.8 — simple classifier prompt | 58.9% | 0.377 |
| GPT-4o | 56.1% | 0.339 |
| GPT-5.5-V2 | 50.8% | 0.273 |

PiTREC's NIST prompt is the **highest-agreement** judge of the four — consistent with it giving the
model the same sentence-context the human assessors saw. Full confusion matrix in `summary.md`.

## How it was judged (important)

The prompt is PiTREC's, but the judging was **NOT** run through `pi-trec support judge` / the `pi`
runner. At the time, pi's Claude Pro/Max OAuth hit Anthropic's "extra usage" billing gate and a
nested `claude -p` was unauthenticated in the Desktop session, so the pairs were judged by
**in-session Claude Opus 4.8 (medium) subagents** that received PiTREC's **verbatim** NIST prompt
per pair (one independent judgment each, batched for throughput).

A **PiTREC-native input** for a genuine `pi-trec support judge` run is prepared at
`data/nour-repro/nour-22topic-cap3.{gen,auggen}.jsonl` (22 topics, ≤3 citations/sentence = Nour's
exact pairs). Once pi has an Anthropic credential (enable extra-usage, or an `ANTHROPIC_API_KEY`):

```bash
uv run pi-trec support judge \
  --input-file data/nour-repro/nour-22topic-cap3.gen.jsonl \
  --output-file results/support-nist-opus/gen.jsonl \
  --model anthropic/claude-opus-4-8 --thinking medium --overwrite
# + the auggen file
```

## Files

- `pairs.jsonl` — the 6,693 judged pairs (statement, cited passage, sentence context, human_label).
- `opus48_nist_verdicts.jsonl` — per-pair `human_label` + `opus_nist_label`.
- `summary.md` — aggregate metrics + confusion matrix + judge comparison.
- `assemble_nist.py` / `build_retry.py` — provenance: collect batch outputs + score; build the throttled retry (paths are absolute, from the run machine).
- `topics_22_fromscratch.txt` — the 22 topic IDs.

## Provenance

Pair set built by aligning the participant-submission runfiles (≤3 citations/sentence) to the human
citation judgments by `topic_id` + normalized answer text + citation slot. Data (runfiles + human
reference `final.citation_judgments_*`) lives in `castorini/citation-support-agents`
(`support-results/36-topics/participant-submissions-45-runfiles`, `data/trec-rag2024/`). The same
result also lives there under `analysis-lingwei/opus48-nist-rerun/`.
