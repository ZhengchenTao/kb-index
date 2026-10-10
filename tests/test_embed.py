from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from kbindex.embed import GeminiEmbedder, HashEmbedder, OpenAIEmbedder, from_meta, make_embedder


def test_hash_embedder_is_deterministic_and_normalised():
    e = HashEmbedder(dim=64)
    a, b = e.embed_documents(["rolling back a release", "rolling back a release"])
    assert a.shape == (64,) and a.dtype == np.float32
    assert np.allclose(a, b) and abs(float(np.linalg.norm(a)) - 1.0) < 1e-5
    q = e.embed_query("roll back release")
    assert float(a @ q) > float(e.embed_query("sourdough starter") @ a)


def test_factory():
    assert isinstance(make_embedder("hash", "hash", 32), HashEmbedder)
    with pytest.raises(ValueError):
        make_embedder("nope")


def test_gemini_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("MY_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        make_embedder("gemini")
    with pytest.raises(RuntimeError, match="MY_KEY"):
        make_embedder("gemini", api_key_env="MY_KEY")
    monkeypatch.setenv("MY_KEY", "k")
    assert make_embedder("gemini", api_key_env="MY_KEY").key == "k"


def test_from_meta_defaults_to_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    e = from_meta({"model": "gemini-embedding-001", "dim": "1536"})       # a database from before providers
    assert isinstance(e, GeminiEmbedder) and e.dim == 1536
    assert isinstance(from_meta({"provider": "hash", "model": "hash", "dim": "16"}), HashEmbedder)


class _Fake(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Fake.seen.append((self.path, dict(self.headers), body))
        data = [{"index": i, "embedding": [float(len(t)), 1.0, 0.0]} for i, t in enumerate(body["input"])]
        raw = json.dumps({"data": list(reversed(data))}).encode()      # out of order on purpose
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_server():
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    _Fake.seen = []
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_openai_compatible(fake_server, monkeypatch):
    monkeypatch.delenv("NO_SUCH_KEY", raising=False)
    e = OpenAIEmbedder("m", dim=3, base_url=fake_server, api_key_env="NO_SUCH_KEY",
                       query_prefix="query: ", document_prefix="passage: ")
    docs = e.embed_documents(["a", "bbb"])
    path, headers, body = _Fake.seen[-1]
    assert path == "/v1/embeddings" and body["model"] == "m"
    assert body["input"] == ["passage: a", "passage: bbb"] and "dimensions" not in body
    assert "Authorization" not in headers
    assert docs[0][0] < docs[1][0]                                     # reordered by index
    e.embed_query("q")
    assert _Fake.seen[-1][2]["input"] == ["query: q"]

    monkeypatch.setenv("SOME_KEY", "secret-value")
    OpenAIEmbedder("m", dim=3, base_url=fake_server, api_key_env="SOME_KEY").embed_query("q")
    assert _Fake.seen[-1][1]["Authorization"] == "Bearer secret-value"
    with pytest.raises(RuntimeError, match="3-d"):
        OpenAIEmbedder("m", dim=8, base_url=fake_server, api_key_env="").embed_query("q")


def test_usage_log(tmp_path, monkeypatch):
    from kbindex import usage
    log = tmp_path / "u.jsonl"
    monkeypatch.setenv("KB_USAGE_LOG", str(log))
    usage.record("query", "m", 12, 1, 40, False)
    rec = json.loads(log.read_text(encoding="utf-8"))
    assert list(rec) == ["ts", "date", "kind", "model", "tokens", "items", "chars", "exact"]
    assert usage.estimate_tokens(["\u4f60\u597d abcdefgh"]) == 2 + 9 // 4
