from pi_trec.pyserini_wrapper import (
    PyseriniWrapperConfig,
    build_pi_search_http_json_config,
    read_pyserini_document,
    search_pyserini,
)


def test_build_pi_search_http_json_config() -> None:
    config = build_pi_search_http_json_config(
        wrapper_base_url="http://127.0.0.1:8091/",
        backend_id="climbmix",
        max_page_size=50,
        max_read_limit=200,
    )

    assert config["backend"]["kind"] == "http-json"
    assert config["backend"]["capabilities"]["backendId"] == "climbmix"
    assert config["backend"]["endpoints"] == {
        "searchUrl": "http://127.0.0.1:8091/search",
        "readDocumentUrl": "http://127.0.0.1:8091/read_document",
    }


def test_search_pyserini_maps_candidates(monkeypatch) -> None:
    calls = []

    def fake_fetch_json(url: str, *, headers: dict[str, str]):
        calls.append((url, headers))
        return {
            "candidates": [
                {"docid": "d1", "doc": {"contents": "alpha"}, "score": 2.5},
                {"docid": "d2", "doc": {"text": "beta"}, "score": 1.5},
                {"docid": "d3", "doc": "gamma"},
            ]
        }

    monkeypatch.setattr("pi_trec.pyserini_wrapper._fetch_json", fake_fetch_json)
    monkeypatch.setenv("PYSERINI_API_TOKEN", "secret")

    response = search_pyserini(
        PyseriniWrapperConfig("http://upstream", "idx", max_page_size=2),
        {"query": "alpha", "offset": 2, "limit": 2},
    )

    assert calls[0] == (
        "http://upstream/v1/idx/search?query=alpha&hits=3",
        {"Authorization": "Bearer secret"},
    )
    assert response["hits"] == [
        {"docid": "d2", "snippet": "beta", "snippetTruncated": False, "score": 1.5},
        {"docid": "d3", "snippet": "gamma", "snippetTruncated": False},
    ]
    assert response["hasMore"] is True
    assert response["nextOffset"] == 4


def test_search_pyserini_truncates_snippets_by_word_limit(monkeypatch) -> None:
    def fake_fetch_json(url: str, *, headers: dict[str, str]):
        return {"candidates": [{"docid": "d1", "doc": {"contents": "one two three four"}, "score": 2.5}]}

    monkeypatch.setattr("pi_trec.pyserini_wrapper._fetch_json", fake_fetch_json)

    response = search_pyserini(
        PyseriniWrapperConfig("http://upstream", "idx", search_word_limit=3),
        {"query": "alpha", "limit": 1},
    )

    assert response["hits"] == [
        {"docid": "d1", "snippet": "one two three", "snippetTruncated": True, "score": 2.5}
    ]


def test_read_pyserini_document_paginates_lines(monkeypatch) -> None:
    def fake_fetch_json(url: str, *, headers: dict[str, str]):
        assert url == "http://upstream/v1/idx/doc/d1"
        return {"docid": "d1", "doc": {"contents": "one\ntwo\nthree"}, "score": 4.0}

    monkeypatch.setattr("pi_trec.pyserini_wrapper._fetch_json", fake_fetch_json)

    response = read_pyserini_document(
        PyseriniWrapperConfig("http://upstream", "idx", read_limit=2),
        {"docid": "d1", "offset": 2, "limit": 2},
    )

    assert response["found"] is True
    assert response["text"] == "two\nthree"
    assert response["offset"] == 2
    assert response["returnedOffsetEnd"] == 3
    assert response["truncated"] is False
    assert response["metadata"] == {"score": 4.0}


def test_read_pyserini_document_truncates_text_by_word_limit(monkeypatch) -> None:
    def fake_fetch_json(url: str, *, headers: dict[str, str]):
        return {"docid": "d1", "doc": {"body": "one two three\nfour five six"}}

    monkeypatch.setattr("pi_trec.pyserini_wrapper._fetch_json", fake_fetch_json)

    response = read_pyserini_document(
        PyseriniWrapperConfig("http://upstream", "idx", read_limit=10, read_word_limit=4),
        {"docid": "d1"},
    )

    assert response["found"] is True
    assert response["text"] == "one two three four"
    assert response["truncated"] is True
