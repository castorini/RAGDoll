"""Pyserini HTTP wrapper compatible with Pine's pi-search backend contract."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class PyseriniWrapperConfig:
    pyserini_base_url: str
    pyserini_index: str
    backend_id: str = "pyserini-http"
    default_limit: int = 10
    max_page_size: int = 100
    read_limit: int = 200
    search_word_limit: int = 512
    read_word_limit: int = 4096
    token_env: str = "PYSERINI_API_TOKEN"


def build_pi_search_http_json_config(
    *,
    wrapper_base_url: str,
    backend_id: str = "pyserini-http",
    max_page_size: int = 100,
    max_read_limit: int = 200,
) -> dict[str, Any]:
    base_url = wrapper_base_url.rstrip("/")
    return {
        "backend": {
            "kind": "http-json",
            "capabilities": {
                "backendId": backend_id,
                "supportsScore": True,
                "supportsSnippets": True,
                "supportsExactTotalHits": False,
                "maxPageSize": max_page_size,
                "maxReadLimit": max_read_limit,
            },
            "endpoints": {
                "searchUrl": f"{base_url}/search",
                "readDocumentUrl": f"{base_url}/read_document",
            },
        }
    }


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("content-length", "0"))
    if length <= 0:
        return {}
    body = handler.rfile.read(length).decode("utf-8")
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError("request body must be a JSON object")
    return value


def _clip_positive_int(value: Any, *, default: int, minimum: int = 1, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _pyserini_headers(config: PyseriniWrapperConfig) -> dict[str, str]:
    token = os.environ.get(config.token_env, "").strip()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _fetch_json(url: str, *, headers: dict[str, str]) -> dict[str, Any]:
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=30.0) as response:
            value = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"upstream HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"upstream unavailable: {exc.reason}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("upstream returned non-object JSON")
    return value


def _pyserini_search_url(config: PyseriniWrapperConfig, *, query: str, limit: int) -> str:
    params = urlencode({"query": query, "hits": str(limit)})
    return f"{config.pyserini_base_url.rstrip('/')}/v1/{config.pyserini_index}/search?{params}"


def _pyserini_doc_url(config: PyseriniWrapperConfig, *, docid: str) -> str:
    return f"{config.pyserini_base_url.rstrip('/')}/v1/{config.pyserini_index}/doc/{docid}"


def _truncate_words(text: str, limit: int) -> tuple[str, bool]:
    if limit <= 0:
        return text, False
    words = text.split()
    if len(words) <= limit:
        return text, False
    return " ".join(words[:limit]), True


def _doc_payload_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("contents", "content", "text", "body"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return json.dumps(payload, ensure_ascii=False)


def _hit_from_candidate(candidate: dict[str, Any], *, word_limit: int) -> dict[str, Any] | None:
    docid = candidate.get("docid")
    if docid is None:
        return None
    text, truncated = _truncate_words(_doc_payload_text(candidate.get("doc")), word_limit)
    hit: dict[str, Any] = {
        "docid": str(docid),
        "snippet": text,
        "snippetTruncated": truncated,
    }
    if candidate.get("score") is not None:
        hit["score"] = float(candidate["score"])
    if candidate.get("title") is not None:
        hit["title"] = str(candidate["title"])
    return hit


def search_pyserini(config: PyseriniWrapperConfig, request_body: dict[str, Any]) -> dict[str, Any]:
    query = str(request_body.get("query", "")).strip()
    if not query:
        raise ValueError("query must be a non-empty string")
    offset = _clip_positive_int(request_body.get("offset"), default=1, maximum=10_000_000)
    limit = _clip_positive_int(
        request_body.get("limit"),
        default=config.default_limit,
        maximum=config.max_page_size,
    )
    upstream_limit = offset + limit - 1
    started = time.perf_counter()
    payload = _fetch_json(
        _pyserini_search_url(config, query=query, limit=upstream_limit),
        headers=_pyserini_headers(config),
    )
    candidates = [item for item in list(payload.get("candidates", [])) if isinstance(item, dict)]
    page = candidates[offset - 1 : offset - 1 + limit]
    hits = [
        hit
        for hit in (_hit_from_candidate(candidate, word_limit=config.search_word_limit) for candidate in page)
        if hit is not None
    ]
    has_more = len(candidates) >= upstream_limit and len(hits) == limit
    response: dict[str, Any] = {
        "hits": hits,
        "hasMore": has_more,
        "timingMs": {"request": round((time.perf_counter() - started) * 1000, 3)},
    }
    if has_more:
        response["nextOffset"] = offset + len(hits)
    return response


def read_pyserini_document(config: PyseriniWrapperConfig, request_body: dict[str, Any]) -> dict[str, Any]:
    docid = str(request_body.get("docid", "")).strip()
    if not docid:
        raise ValueError("docid must be a non-empty string")
    offset = _clip_positive_int(request_body.get("offset"), default=1, maximum=10_000_000)
    limit = _clip_positive_int(
        request_body.get("limit"),
        default=config.read_limit,
        maximum=config.read_limit,
    )
    started = time.perf_counter()
    payload = _fetch_json(_pyserini_doc_url(config, docid=docid), headers=_pyserini_headers(config))
    text = payload.get("doc")
    if text is None:
        return {
            "found": False,
            "docid": docid,
            "timingMs": {"request": round((time.perf_counter() - started) * 1000, 3)},
        }
    lines = _doc_payload_text(text).splitlines() or [_doc_payload_text(text)]
    start_index = offset - 1
    selected = lines[start_index : start_index + limit]
    selected_text, word_truncated = _truncate_words("\n".join(selected), config.read_word_limit)
    returned_end = offset + len(selected) - 1
    truncated = returned_end < len(lines) or word_truncated
    response: dict[str, Any] = {
        "found": True,
        "docid": str(payload.get("docid") or docid),
        "text": selected_text,
        "offset": offset,
        "limit": limit,
        "totalUnits": len(lines),
        "returnedOffsetStart": offset if selected else offset,
        "returnedOffsetEnd": returned_end if selected else offset,
        "truncated": truncated,
        "timingMs": {"request": round((time.perf_counter() - started) * 1000, 3)},
    }
    if truncated:
        response["nextOffset"] = returned_end + 1
    if payload.get("score") is not None:
        response["metadata"] = {"score": payload["score"]}
    return response


def make_pyserini_wrapper_handler(config: PyseriniWrapperConfig) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                _json_response(self, HTTPStatus.OK, {"ok": True, "backendId": config.backend_id})
                return
            if self.path == "/pi_search_config":
                host = self.headers.get("host", f"127.0.0.1:{self.server.server_port}")
                _json_response(
                    self,
                    HTTPStatus.OK,
                    build_pi_search_http_json_config(
                        wrapper_base_url=f"http://{host}",
                        backend_id=config.backend_id,
                        max_page_size=config.max_page_size,
                        max_read_limit=config.read_limit,
                    ),
                )
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            try:
                body = _read_json_body(self)
                if self.path == "/search":
                    _json_response(self, HTTPStatus.OK, search_pyserini(config, body))
                    return
                if self.path == "/read_document":
                    _json_response(self, HTTPStatus.OK, read_pyserini_document(config, body))
                    return
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
            except ValueError as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive server boundary
                _json_response(self, HTTPStatus.BAD_GATEWAY, {"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

    return Handler


def serve_pyserini_wrapper(args: Any) -> None:
    config = PyseriniWrapperConfig(
        pyserini_base_url=args.pyserini_base_url,
        pyserini_index=args.pyserini_index,
        backend_id=args.backend_id,
        default_limit=args.default_limit,
        max_page_size=args.max_page_size,
        read_limit=args.read_limit,
        search_word_limit=args.search_word_limit,
        read_word_limit=args.read_word_limit,
        token_env=args.token_env,
    )
    wrapper_base_url = f"http://{args.host}:{args.port}"
    pi_search_config = build_pi_search_http_json_config(
        wrapper_base_url=wrapper_base_url,
        backend_id=config.backend_id,
        max_page_size=config.max_page_size,
        max_read_limit=config.read_limit,
    )
    if args.print_config:
        print("PI_SEARCH_EXTENSION_CONFIG=" + json.dumps(pi_search_config, separators=(",", ":")))
    server = ThreadingHTTPServer((args.host, args.port), make_pyserini_wrapper_handler(config))
    print(f"serving pyserini wrapper at {wrapper_base_url}")
    print(f"upstream={config.pyserini_base_url.rstrip('/')}/v1/{config.pyserini_index}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping pyserini wrapper")
    finally:
        server.server_close()
