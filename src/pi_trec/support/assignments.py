from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pi_trec.config import SupportAssembleConfig, SupportSummarizeConfig
from pi_trec.jsonl import read_jsonl, write_jsonl
from pi_trec.support.metrics import metric_lines, support_metric_rows
from pi_trec.support.prompts import parse_support_label

SUPPORT_LABEL_SCORES = {"FS": 2, "PS": 1, "NS": 0}


def _topic_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(
        row.get("topic_id")
        or row.get("narrative_id")
        or row.get("qid")
        or row.get("query_id")
        or metadata.get("topic_id")
        or metadata.get("narrative_id")
        or metadata.get("request_id")
        or ""
    )


def _run_id(row: dict[str, Any], *, default: str = "") -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("run_id") or metadata.get("run_id") or metadata.get("team_id") or default)


def _answer_sentences(row: dict[str, Any]) -> list[dict[str, Any]]:
    if "answer" in row:
        answer = row.get("answer")
    elif "response" in row:
        answer = row.get("response")
    else:
        answer = row.get("responses", [])
    if isinstance(answer, str):
        return [{"text": answer, "citations": []}]
    if not isinstance(answer, list):
        return []
    sentences: list[dict[str, Any]] = []
    for item in answer:
        if isinstance(item, dict):
            citations = item.get("citations")
            if isinstance(citations, dict):
                citations = list(citations)
            sentences.append(
                {
                    "text": str(item.get("text", "")),
                    "citations": citations if isinstance(citations, list) else [],
                }
            )
        else:
            sentences.append({"text": str(item), "citations": []})
    return sentences


def _citation_reference(citation: Any, references: list[Any]) -> str:
    if isinstance(citation, dict):
        return str(citation.get("reference") or citation.get("docid") or citation.get("document_id") or citation)
    if isinstance(citation, int) and 0 <= citation < len(references):
        return str(references[citation])
    return str(citation)


def _judgment_support(row: dict[str, Any]) -> int:
    if row.get("status") != "completed":
        return -1
    label = row.get("support_label")
    if not isinstance(label, str) or label not in SUPPORT_LABEL_SCORES:
        label = parse_support_label(str(row.get("raw_output") or ""))
    return SUPPORT_LABEL_SCORES.get(label, -1)


def _judgment_key(row: dict[str, Any]) -> tuple[str, str, int, int] | None:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return None
    source = metadata.get("source") if isinstance(metadata.get("source"), dict) else {}
    try:
        return (
            str(metadata.get("run_id") or source.get("run_id") or ""),
            str(
                metadata.get("topic_id")
                or metadata.get("narrative_id")
                or source.get("topic_id")
                or source.get("narrative_id")
                or ""
            ),
            int(
                metadata.get("sentence_index")
                if metadata.get("sentence_index") is not None
                else source.get("sentence_index")
            ),
            int(
                metadata.get("citation_index")
                if metadata.get("citation_index") is not None
                else source.get("citation_index")
            ),
        )
    except (TypeError, ValueError):
        return None


def _judgment_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_file():
            out.append(path)
        elif path.is_dir():
            matched = sorted(path.rglob("judgments.parsed.jsonl"))
            out.extend(matched if matched else sorted(path.rglob("*.jsonl")))
    return out


def load_support_judgments(paths: Iterable[Path]) -> dict[tuple[str, str, int, int], int]:
    judgments: dict[tuple[str, str, int, int], int] = {}
    for path in _judgment_paths(paths):
        for row in read_jsonl(path):
            key = _judgment_key(row)
            if key is not None:
                judgments[key] = _judgment_support(row)
    return judgments


def assemble_support_assignments(
    answers_file: Path,
    judgment_paths: Iterable[Path],
    *,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    judgments = load_support_judgments(judgment_paths)
    rows: list[dict[str, Any]] = []
    for answer_row in read_jsonl(answers_file):
        topic_id = _topic_id(answer_row)
        answer_run_id = _run_id(answer_row, default=run_id or answers_file.name)
        references = answer_row.get("references")
        references = references if isinstance(references, list) else []
        sentences: list[dict[str, Any]] = []
        for sentence_index, sentence in enumerate(_answer_sentences(answer_row)):
            citations: list[dict[str, Any]] = []
            for citation_index, citation in enumerate(sentence["citations"]):
                support = judgments.get((answer_run_id, topic_id, sentence_index, citation_index), -1)
                citations.append(
                    {
                        "citationID": citation_index,
                        "reference": _citation_reference(citation, references),
                        "support": str(support),
                    }
                )
            sentences.append({"sentenceID": sentence_index, "text": sentence["text"], "citations": citations})
        rows.append({"topic_id": topic_id, "narrative_id": topic_id, "run_id": answer_run_id, "sentences": sentences})
    return sorted(rows, key=lambda row: (_run_sort_key(row["run_id"]), _topic_sort_key(row["topic_id"])))


def _run_sort_key(run_id: str) -> str:
    return str(run_id)


def _topic_sort_key(topic_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(topic_id))
    except ValueError:
        return (1, topic_id)


def assemble(config: SupportAssembleConfig) -> None:
    rows = assemble_support_assignments(config.answers_file, config.judgments, run_id=config.run_id)
    count = write_jsonl(config.output_file, rows)
    print(f"wrote assignments={count} output={config.output_file}")


def _answer_files(config: SupportSummarizeConfig) -> list[Path]:
    if config.answers:
        return sorted(config.answers)
    return sorted(path for path in config.answers_dir.iterdir() if path.is_file() and not path.name.startswith("."))


def _judgments_for_run(root: Path, run_id: str) -> list[Path]:
    candidates = [
        root / run_id / "judgments.parsed.jsonl",
        root / "gen" / run_id / "judgments.parsed.jsonl",
        root / "auggen" / run_id / "judgments.parsed.jsonl",
    ]
    if root.name == run_id:
        candidates.append(root / "judgments.parsed.jsonl")
    found = [path for path in candidates if path.exists()]
    if found:
        return found
    return sorted(path for path in root.rglob("judgments.parsed.jsonl") if path.parent.name == run_id)


def summarize(config: SupportSummarizeConfig) -> None:
    assignments: list[dict[str, Any]] = []
    missing: list[str] = []
    for answers_file in _answer_files(config):
        run_id = answers_file.name
        judgment_paths = _judgments_for_run(config.judgments_root, run_id)
        if not judgment_paths:
            missing.append(run_id)
        assignments.extend(assemble_support_assignments(answers_file, judgment_paths, run_id=run_id))

    config.output_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = config.output_dir / "support_assignments.jsonl"
    metrics_path = config.output_dir / "support_metrics.jsonl"
    metric_rows_path = config.output_dir / "support_metric_rows.txt"

    assignment_count = write_jsonl(assignment_path, assignments)
    metrics = support_metric_rows(assignments)
    metric_count = write_jsonl(metrics_path, metrics)
    lines = metric_lines(metrics, config.metrics)
    metric_rows_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    print(f"wrote assignments={assignment_count} output={assignment_path}")
    print(f"wrote metrics={metric_count} output={metrics_path}")
    print(f"wrote metric_rows={len(lines)} output={metric_rows_path}")
    if missing:
        preview = ", ".join(missing[:10]) + ("..." if len(missing) > 10 else "")
        print(f"warning: missing judgments for {len(missing)} run(s): {preview}")
