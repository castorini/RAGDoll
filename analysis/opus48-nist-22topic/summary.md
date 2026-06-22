# Opus-4.8 (NIST prompt) vs Human — 22-topic from-scratch rerun

Reproduction of the support evaluation on the **22 from-scratch human-reference topics**, using Claude **Opus 4.8 (medium effort)** with the **official TREC RAG NIST support assessor prompt** (== PiTREC `SUPPORT_EVAL_PROMPT`, with Sentence Context). Judged in-session via batched Opus subagents (Nour's pipeline logic; one verbatim NIST prompt per pair, judged independently). Pairs aligned to human labels by topic + answer-text + citation slot (same as the existing 22-topic analysis).

## Aggregate

| Metric | Value |
| --- | ---: |
| Compared citation judgments (N) | 6693 |
| Agreement | 4012/6693 (59.9%) |
| Cohen's kappa | 0.394 |
| Opus more generous than human | 1809 |
| Opus stricter than human | 872 |

## Confusion matrix (rows = Human, cols = Opus-4.8 NIST)

|  | **Opus NS** | **Opus PS** | **Opus FS** | Human total |
|--|--:|--:|--:|--:|
| **Human NS** | 1188 | 798 | 416 | 2402 |
| **Human PS** | 240 | 822 | 595 | 1657 |
| **Human FS** | 146 | 486 | 2002 | 2634 |

Opus-4.8 NIST verdict distribution: NS 1574, PS 2106, FS 3013.

## Judge comparison on the identical 22-topic human reference

| Judge | Agreement | Cohen's kappa |
| --- | ---: | ---: |
| **Opus-4.8 — NIST prompt (this run)** | **59.9%** | **0.394** |
| Opus-4.8 — simple classifier prompt | 58.9% | 0.377 |
| GPT-4o | 56.1% | 0.339 |
| GPT-5.5-V2 | 50.8% | 0.273 |

Per-pair verdicts: `opus48_nist_verdicts.jsonl`. Pair set: `pairs.jsonl` (6693 unique pairs).
