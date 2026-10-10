"""Configuration: sources.yaml + .indexignore.

Only the indexer reads this file. The MCP server and the query commands read everything they need
(repo rules, URLs, thresholds, exclusion patterns, client settings) from the `meta` table of index.db,
so a client machine needs nothing but the database file.

Lookup order:
  config file     --config  > $KB_CONFIG  > ./sources.yaml
  ignore file     --indexignore > $KB_INDEXIGNORE > `ignore_file` in the config (relative to the config file)
                  > .indexignore next to the config file
  source paths    absolute paths are used as-is; relative paths are resolved against the config file's directory

Environment:
  KB_CONFIG       path to sources.yaml
  KB_INDEXIGNORE  path to .indexignore
  KB_DB           indexer work database (default ./work.db)
  KB_OUT          publish directory for index.db / index.db.sha256 / deltas (default: KB_DB's directory)
  KB_USAGE_LOG    embedding usage log (default: <KB_DB dir>/embed-usage.jsonl; nothing is written if neither is set)
  <api_key_env>   embedding API key; the variable name comes from `embedding.api_key_env`
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_PREAMBLE = "(preamble)"


@dataclass
class Source:
    repo: str
    type: str                      # git | dir
    path: str
    domain: str = "misc"           # default domain
    domains: dict[str, str] = field(default_factory=dict)   # path prefix -> domain
    include: list[str] = field(default_factory=lambda: ["*.md"])
    only: list[str] = field(default_factory=list)          # index only these prefixes (useful while tuning chunking)
    rule: str = ""                 # one-line rule for this repo, returned alongside its content by every tool
    url: str = ""                  # canonical clone URL; returned to clients that do not have the repo locally
    clone_hint: str = ""           # free-text hint for clients that do not have the repo locally (e.g. not a git repo)
    ref: str = "HEAD"              # git sources only

    def domain_of(self, rel: str) -> str:
        best = ""
        for prefix, dom in self.domains.items():
            if rel.startswith(prefix) and len(prefix) > len(best):
                best, chosen = prefix, dom
        return chosen if best else self.domain  # type: ignore[possibly-undefined]


@dataclass
class Config:
    sources: list[Source]
    provider: str = "gemini"       # gemini | openai | hash
    model: str = "gemini-embedding-001"
    dim: int = 1536
    batch: int = 50
    base_url: str = ""             # openai provider: e.g. https://api.openai.com/v1 or http://localhost:11434/v1
    api_key_env: str = ""          # name of the environment variable holding the key ("" = provider default)
    query_prefix: str = ""         # prepended to queries (instruction-tuned / asymmetric models)
    document_prefix: str = ""      # prepended to documents at embedding time
    threshold: float = 0.70
    rrf_k: int = 60
    min_chunk: int = 30
    max_chunk: int = 1200
    preamble_label: str = DEFAULT_PREAMBLE   # heading given to text before the first heading; changing it re-embeds
    ignore_file: str = ""
    extended_meta: bool = True     # write provider / client:* / hint:* keys into meta (see indexer.run)
    client: dict = field(default_factory=dict)
    path: Path = field(default_factory=lambda: Path("sources.yaml"))

    def source(self, repo: str) -> Source:
        for s in self.sources:
            if s.repo == repo:
                return s
        raise KeyError(f"unknown repo {repo!r}; known: {[s.repo for s in self.sources]}")

    @property
    def cache_model(self) -> str:
        """Key of vec_cache rows. The bare model name, so existing caches stay valid; a document prefix
        changes the vectors without changing the text hash, so it is folded into the key."""
        if not self.document_prefix:
            return self.model
        import hashlib
        return f"{self.model}#doc:{hashlib.sha1(self.document_prefix.encode('utf-8')).hexdigest()[:8]}"


def _is_abs(p: str) -> bool:
    # Treat POSIX-rooted paths as absolute on Windows too: configs written for a Linux container
    # (/sources/...) must not be re-rooted under the config directory.
    return p.startswith(("/", "\\", "~")) or os.path.isabs(p)


def _resolve(base: Path, p: str) -> str:
    if p.startswith("~"):
        return os.path.expanduser(p)
    if not p or _is_abs(p):
        return p
    return str((base / p).resolve())


def load_config(path: str | os.PathLike | None = None) -> Config:
    p = Path(path or os.environ.get("KB_CONFIG", "sources.yaml"))
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    emb = raw.get("embedding", {}) or {}
    srch = raw.get("search", {}) or {}
    chk = raw.get("chunk", {}) or {}
    base = p.parent
    sources = []
    for s in raw.get("sources", []) or []:
        src = Source(**s)
        src.path = _resolve(base, src.path)
        sources.append(src)
    return Config(
        sources=sources,
        provider=str(emb.get("provider", "gemini")),
        model=emb.get("model", "gemini-embedding-001"),
        dim=int(emb.get("dim", 1536)),
        batch=int(emb.get("batch", 50)),
        base_url=str(emb.get("base_url", "") or ""),
        api_key_env=str(emb.get("api_key_env", "") or ""),
        query_prefix=str(emb.get("query_prefix", "") or ""),
        document_prefix=str(emb.get("document_prefix", "") or ""),
        threshold=float(srch.get("threshold", 0.70)),
        rrf_k=int(srch.get("rrf_k", 60)),
        min_chunk=int(chk.get("min", 30)),
        max_chunk=int(chk.get("max", 1200)),
        preamble_label=str(chk.get("preamble_label", DEFAULT_PREAMBLE)),
        ignore_file=str(raw.get("ignore_file", "") or ""),
        extended_meta=bool(raw.get("extended_meta", True)),
        client=dict(raw.get("client", {}) or {}),
        path=p,
    )


# ---------------------------------------------------------------- .indexignore

def _glob_to_re(pat: str) -> re.Pattern:
    """gitignore-style pattern -> regex. Matched against `repo/relative/path`.

    `**` crosses directories; `*` does not cross `/`; `?` is one character; a pattern without `/` matches
    a file name at any depth; a trailing `/` means a directory (same as `dir/**`).
    """
    pat = pat.strip()
    if pat.endswith("/"):
        pat = pat + "**"
    anchored = "/" in pat.rstrip("/")
    out = []
    i = 0
    while i < len(pat):
        c = pat[i]
        if c == "*":
            if pat[i:i + 2] == "**":
                i += 2
                if i < len(pat) and pat[i] == "/":
                    i += 1
                    out.append("(?:.*/)?")
                else:
                    out.append(".*")
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(c))
        i += 1
    body = "".join(out)
    if anchored:
        return re.compile(rf"^{body}$")
    return re.compile(rf"(?:^|/){body}$")


class IgnoreRules:
    def __init__(self, patterns: list[str]):
        self.patterns = [p for p in patterns if p and not p.startswith("#")]
        self._res = [(_glob_to_re(p), p) for p in self.patterns]

    @classmethod
    def load(cls, path: str | os.PathLike | None, config_path: Path, ignore_file: str = "") -> "IgnoreRules":
        if path or os.environ.get("KB_INDEXIGNORE"):
            p = Path(path or os.environ["KB_INDEXIGNORE"])
        elif ignore_file:
            p = Path(_resolve(config_path.parent, ignore_file))
        else:
            p = config_path.with_name(".indexignore")
        if not p.exists():
            return cls([])
        return cls(p.read_text(encoding="utf-8").splitlines())

    @classmethod
    def for_config(cls, cfg: Config, path: str | os.PathLike | None = None) -> "IgnoreRules":
        return cls.load(path, cfg.path, cfg.ignore_file)

    def match(self, repo: str, rel: str) -> str | None:
        """Return the first matching pattern, or None."""
        key = f"{repo}/{rel}"
        for rx, pat in self._res:
            if rx.search(key):
                return pat
        return None


def matches_any(rel: str, globs: list[str]) -> bool:
    if not globs:
        return True
    return any(_glob_to_re(g).search(rel) for g in globs)
