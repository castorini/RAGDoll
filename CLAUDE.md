# Pi-TREC Repository Instructions

## Scope

This repository is used for running TREC-RAG's 2026 evaluation through Pi.

## Development Rules

- Direct pushes are not allowed. For each development phase, create a new branch and then a pull request.
- Keep public input/output shapes compatible with the corresponding UMBRELA, Nuggetizer, and support-evaluation workflows.
- Keep raw event logs, temporary run artifacts, local environments, and credentials ignored by git.
- Preserve exact prompt surfaces with provenance tests whenever copying prompts from sibling repositories or papers.

## Validation

- Run `uv run pytest` before committing behavior changes.

