"""Embedding providers.

  gemini   Google Generative Language API. Documents go through batchEmbedContents with taskType
           RETRIEVAL_DOCUMENT, queries through embedContent with RETRIEVAL_QUERY. This is an asymmetric
           model: doc-to-doc and query-to-doc cosines are on different scales, never mix the task types.
  openai   Any OpenAI-compatible /embeddings endpoint (OpenAI, Ollama, LM Studio, vLLM, TEI, most cloud
           vendors' compatible modes). Asymmetric models are emulated with query_prefix / document_prefix.
  hash     Deterministic pseudo-vectors, no network. For tests and for trying the pipeline without a key:
           the vector leg is not semantic, the BM25 leg works as usual.

All providers return L2-normalised float32 vectors. API keys are only ever read from the environment;
the name of the variable (never the value) is stored in index.db so the query side can find it.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

from . import usage

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
BASE = GEMINI_BASE   # backwards-compatible alias

DEFAULT_KEY_ENV = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "hash": ""}


def _norm(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return a / n if n else a


def _post(url: str, body: dict, headers: dict, timeout: int = 120) -> dict:
    """POST JSON with retries on 429 / 5xx and network errors (5 attempts, linear backoff)."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={**headers, "Content-Type": "application/json"})
    last: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]
            last = RuntimeError(f"HTTP {e.code}: {msg}")
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(5 * (attempt + 1))
                continue
            raise last from None
        except Exception as e:  # network hiccup
            last = e
            if attempt < 4:
                time.sleep(5 * (attempt + 1))
                continue
            raise
    raise last or RuntimeError("embed failed")


class GeminiEmbedder:
    provider = "gemini"

    def __init__(self, model: str = "gemini-embedding-001", dim: int = 1536, api_key: str | None = None,
                 api_key_env: str = "GEMINI_API_KEY", **_):
        self.model = model
        self.dim = dim
        env = api_key_env or "GEMINI_API_KEY"
        self.key = api_key or os.environ.get(env)
        if not self.key:
            raise RuntimeError(f"{env} is not set")

    def _post(self, url: str, body: dict, timeout: int = 120) -> dict:
        return _post(url, body, {"x-goog-api-key": self.key}, timeout)

    _norm = staticmethod(_norm)

    def count_tokens(self, texts: list[str]) -> int:
        """One call for the whole batch: countTokens accepts an array of contents and returns the total
        (two identical contents count twice). countTokens is free, so the index side uses it for exact
        accounting at the cost of one extra request per batch."""
        body = {"contents": [{"parts": [{"text": t}]} for t in texts]}
        res = self._post(f"{GEMINI_BASE}/{self.model}:countTokens", body, timeout=60)
        return int(res.get("totalTokens", 0))

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        # Record usage before the embedding request: a failed countTokens must not waste the batch.
        try:
            tok = self.count_tokens(texts)
            exact = True
        except Exception:
            tok, exact = usage.estimate_tokens(texts), False
        usage.record("index", self.model, tok, len(texts), sum(len(t) for t in texts), exact)

        body = {"requests": [{
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": t}]},
            "outputDimensionality": self.dim,
            "taskType": "RETRIEVAL_DOCUMENT",
        } for t in texts]}
        res = self._post(f"{GEMINI_BASE}/{self.model}:batchEmbedContents", body)
        return [_norm(e["values"]) for e in res["embeddings"]]

    def embed_query(self, text: str) -> np.ndarray:
        # Deliberately no countTokens here: every search goes through this path and an extra HTTP round trip
        # would add straight to search latency. A query is tens of tokens; a local estimate is enough.
        usage.record("query", self.model, usage.estimate_tokens([text]), 1, len(text), False)

        body = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": self.dim,
            "taskType": "RETRIEVAL_QUERY",
        }
        res = self._post(f"{GEMINI_BASE}/{self.model}:embedContent", body, timeout=60)
        return _norm(res["embedding"]["values"])


# Backwards-compatible name: before providers existed, Embedder was the Gemini client.
Embedder = GeminiEmbedder


class OpenAIEmbedder:
    """POST {base_url}/embeddings with `input` as a list.

    `dimensions` is sent only to the official API by default (text-embedding-3-* accept it); many local
    servers reject the field. The returned width is checked against `dim` when `dim` is set."""
    provider = "openai"

    def __init__(self, model: str, dim: int = 0, base_url: str = "", api_key_env: str = "OPENAI_API_KEY",
                 query_prefix: str = "", document_prefix: str = "", send_dimensions: bool | None = None, **_):
        self.model = model
        self.dim = dim
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.key = os.environ.get(api_key_env) if api_key_env else None
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.send_dimensions = (send_dimensions if send_dimensions is not None
                                else self.base_url.startswith("https://api.openai.com"))

    def _embed(self, texts: list[str]) -> list[np.ndarray]:
        body: dict = {"model": self.model, "input": texts}
        if self.send_dimensions and self.dim:
            body["dimensions"] = self.dim
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        res = _post(f"{self.base_url}/embeddings", body, headers)
        data = sorted(res["data"], key=lambda d: d.get("index", 0))
        out = [_norm(d["embedding"]) for d in data]
        if self.dim and out and out[0].shape[0] != self.dim:
            raise RuntimeError(f"{self.model} returned {out[0].shape[0]}-d vectors but embedding.dim is {self.dim}")
        return out

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        texts = [self.document_prefix + t for t in texts]
        usage.record("index", self.model, usage.estimate_tokens(texts), len(texts), sum(len(t) for t in texts), False)
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        text = self.query_prefix + text
        usage.record("query", self.model, usage.estimate_tokens([text]), 1, len(text), False)
        return self._embed([text])[0]


class HashEmbedder:
    """Deterministic pseudo-embeddings: character trigrams hashed into `dim` signed buckets.

    Not semantic, but texts that share many trigrams get a higher cosine, so demos and tests behave
    sensibly. Never touches the network and records no usage."""
    provider = "hash"

    def __init__(self, model: str = "hash", dim: int = 256, **_):
        self.model = model
        self.dim = dim or 256

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = " ".join(text.lower().split())
        for i in range(max(1, len(t) - 2)):
            h = int.from_bytes(hashlib.blake2b(t[i:i + 3].encode("utf-8"), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _norm(v)

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


PROVIDERS = {"gemini": GeminiEmbedder, "openai": OpenAIEmbedder, "hash": HashEmbedder}


def make_embedder(provider: str = "gemini", model: str = "gemini-embedding-001", dim: int = 1536,
                  base_url: str = "", api_key_env: str = "", query_prefix: str = "", document_prefix: str = ""):
    provider = (provider or "gemini").lower()
    cls = PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(f"unknown embedding provider {provider!r}; known: {sorted(PROVIDERS)}")
    return cls(model=model, dim=dim, base_url=base_url, api_key_env=api_key_env or DEFAULT_KEY_ENV[provider],
               query_prefix=query_prefix, document_prefix=document_prefix)


def from_config(cfg):
    return make_embedder(cfg.provider, cfg.model, cfg.dim, cfg.base_url, cfg.api_key_env,
                         cfg.query_prefix, cfg.document_prefix)


def from_meta(meta: dict):
    """Query-side embedder built from index.db meta. Databases written before providers existed carry only
    model / dim, which means Gemini with GEMINI_API_KEY."""
    return make_embedder(meta.get("provider") or "gemini", meta.get("model") or "gemini-embedding-001",
                         int(meta.get("dim") or 1536), meta.get("base_url", ""), meta.get("api_key_env", ""),
                         meta.get("query_prefix", ""), "")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
