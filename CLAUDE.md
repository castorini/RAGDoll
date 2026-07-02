# RAGDoll Repository Instructions

## Scope

This repository is used for running TREC-RAG's 2026 evaluation through Pi.

## Development Rules

- Direct pushes are not allowed. For each development phase, if you are on the `main` branch, create a new branch to work on.
- Keep public input/output shapes compatible with the corresponding UMBRELA, Nuggetizer, and support-evaluation workflows.
- Keep raw event logs, temporary run artifacts, local environments, and credentials ignored by git.
- Preserve exact prompt surfaces with provenance tests whenever copying prompts from sibling repositories or papers.
- Reuse existing functions whenever possible. Comments should be rare and short!

## Validation

- Run `uv run pytest` before committing behavior changes.

