from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from pi_trec.config import PyseriniWrapperConfig, SupportResolveReferencesConfig
from pi_trec.jsonl import read_jsonl, write_jsonl
from pi_trec.pyserini_wrapper import read_pyserini_document


class ReferenceResolver(Protocol):
    def read(self, docid: str) -> str: ...


class PyseriniApiResolver:
    def __init__(self, config: SupportResolveReferencesConfig) -> None:
        self.config = PyseriniWrapperConfig(
            pyserini_base_url=str(config.pyserini_api),
            pyserini_index=str(config.pyserini_index),
            read_limit=config.read_limit,
            read_word_limit=config.read_word_limit,
            token_env=config.token_env,
        )

    def read(self, docid: str) -> str:
        payload = read_pyserini_document(self.config, {"docid": docid, "limit": self.config.read_limit})
        if not payload.get("found"):
            raise ValueError(f"reference docid not found in Pyserini HTTP index: {docid}")
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"reference docid resolved to empty text: {docid}")
        return text


class PyseriniIndexResolver:
    def __init__(self, index: str, *, read_limit: int, read_word_limit: int) -> None:
        self.searcher = _lucene_searcher(index)
        self.read_limit = read_limit
        self.read_word_limit = read_word_limit

    def read(self, docid: str) -> str:
        doc = self.searcher.doc(docid)
        if doc is None:
            raise ValueError(f"reference docid not found in local Pyserini index: {docid}")
        raw = _call_doc_method(doc, "raw")
        if raw is None:
            raw = _call_doc_method(doc, "contents")
        text = _doc_text(raw)
        selected = "\n".join((text.splitlines() or [text])[: self.read_limit])
        selected = _truncate_words(selected, self.read_word_limit)
        if not selected.strip():
            raise ValueError(f"reference docid resolved to empty text: {docid}")
        return selected


def resolve_references(config: SupportResolveReferencesConfig) -> None:
    resolver: ReferenceResolver
    if config.pyserini_api is None:
        resolver = PyseriniIndexResolver(
            str(config.pyserini_index),
            read_limit=config.read_limit,
            read_word_limit=config.read_word_limit,
        )
    else:
        resolver = PyseriniApiResolver(config)

    cache: dict[str, str] = {}
    rows = [
        _resolve_row(row, resolver=resolver, cache=cache, row_number=row_number)
        for row_number, row in enumerate(read_jsonl(config.input_file), start=1)
    ]
    count = write_jsonl(config.output_file, rows)
    print(f"wrote={count} output={config.output_file}")


def _resolve_row(
    row: dict[str, Any],
    *,
    resolver: ReferenceResolver,
    cache: dict[str, str],
    row_number: int,
) -> dict[str, Any]:
    references = row.get("references")
    answer = row.get("answer")
    if not isinstance(references, list) or not all(isinstance(item, str) for item in references):
        raise ValueError(f"row {row_number}: support reference resolution requires a string `references` list")
    if not isinstance(answer, list):
        raise ValueError(f"row {row_number}: support reference resolution requires an `answer` list")
    _validate_answer_citations(answer, references, row_number=row_number)

    segments = dict(row.get("segments")) if isinstance(row.get("segments"), dict) else {}
    for docid in references:
        if isinstance(segments.get(docid), str) and segments[docid].strip():
            cache.setdefault(docid, segments[docid])
            continue
        if docid not in cache:
            cache[docid] = resolver.read(docid)
        segments[docid] = cache[docid]

    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    resolved = dict(row)
    resolved["topic_id"] = str(row.get("topic_id") or metadata.get("narrative_id") or metadata.get("topic_id") or "")
    resolved["run_id"] = str(row.get("run_id") or metadata.get("run_id") or metadata.get("team_id") or "")
    resolved["segments"] = segments
    if not resolved["topic_id"]:
        raise ValueError(f"row {row_number}: missing topic_id or metadata.narrative_id")
    if not resolved["run_id"]:
        raise ValueError(f"row {row_number}: missing run_id or metadata.run_id")
    return resolved


def _validate_answer_citations(answer: list[Any], references: list[str], *, row_number: int) -> None:
    for sentence_index, sentence in enumerate(answer):
        if not isinstance(sentence, dict):
            raise ValueError(f"row {row_number}: answer[{sentence_index}] must be an object")
        if not isinstance(sentence.get("text"), str):
            raise ValueError(f"row {row_number}: answer[{sentence_index}].text must be a string")
        citations = sentence.get("citations")
        if not isinstance(citations, list):
            raise ValueError(f"row {row_number}: answer[{sentence_index}].citations must be a list")
        for citation in citations:
            if not isinstance(citation, int):
                raise ValueError(f"row {row_number}: citations must be zero-indexed integer reference positions")
            if citation < 0 or citation >= len(references):
                raise ValueError(f"row {row_number}: citation index {citation} is outside references")


def _lucene_searcher(index: str) -> Any:
    searcher_cls = _lucene_searcher_cls()
    path = Path(index).expanduser()
    if path.exists():
        return searcher_cls(str(path))
    if hasattr(searcher_cls, "from_prebuilt_index"):
        return searcher_cls.from_prebuilt_index(index)
    return searcher_cls(index)


def _lucene_searcher_cls() -> Any:
    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError:
        try:
            from pyserini.search import LuceneSearcher
        except ImportError as exc:
            raise RuntimeError("local reference resolution requires Pyserini: install `pyserini` first") from exc
    return LuceneSearcher


def _call_doc_method(doc: Any, name: str) -> Any:
    value = getattr(doc, name, None)
    return value() if callable(value) else value


def _doc_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return _doc_text(parsed)
    if isinstance(value, dict):
        title = value.get("title")
        segment = value.get("segment")
        if isinstance(title, str) and isinstance(segment, str):
            return f"{title}: {segment}"
        for key in ("contents", "content", "text", "body"):
            text = value.get(key)
            if isinstance(text, str):
                return text
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return json.dumps(value, ensure_ascii=False)


def _truncate_words(text: str, limit: int) -> str:
    if limit <= 0:
        return text
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit])
