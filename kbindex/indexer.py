"""Indexer: write-only. For every source it chunks the changed files, embeds what the cache does not have,
and extracts links.

The run is deliberately split in two phases: first every missing vector is embedded and written to vec_cache
batch by batch (each batch committed), then the chunks are written. A crash halfway costs nothing on re-run.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .chunker import Chunk, split_file
from .config import Config, IgnoreRules, Source
from .embed import from_config, log
from .links import Resolver, extract_raw_links
from .sources import DirSource, FileEntry, GitSource, open_source
from .store import Store, vec_to_blob


@dataclass
class Pending:
    src: Source
    entry: FileEntry
    chunks: list[Chunk]
    links: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class RunReport:
    started: str = ""
    per_repo: dict = field(default_factory=dict)
    embedded: int = 0
    cached: int = 0
    failures: list[str] = field(default_factory=list)
    elapsed: float = 0.0


def plan_source(cfg: Config, ignore: IgnoreRules, store: Store, src: Source, force: bool = False):
    """Return (pending, deletes, skipped count, ignored count, head)."""
    reader = open_source(src)
    known = store.known_files(src.repo)
    if isinstance(reader, DirSource):
        entries = reader.list_files(known)
    else:
        entries = reader.list_files()
    head = reader.head()

    kept: list[FileEntry] = []
    ignored = 0
    for e in entries:
        if ignore.match(src.repo, e.rel):
            ignored += 1
            continue
        kept.append(e)
    resolver = Resolver([e.rel for e in kept])
    current = {e.rel for e in kept}
    deletes = [p for p in known if p not in current]

    pending: list[Pending] = []
    skipped = 0
    for e in kept:
        k = known.get(e.rel)
        if k and k[2] == e.hash and not force:
            skipped += 1
            continue
        try:
            content = reader.read(e.rel)
        except Exception as ex:
            log(f"  ! read failed {src.repo}/{e.rel}: {ex}")
            continue
        chunks = split_file(content, e.rel, cfg.min_chunk, cfg.max_chunk, cfg.preamble_label)
        links: list[tuple[int, str]] = []
        seen: set[tuple[int, str]] = set()
        for c in chunks:
            for raw in extract_raw_links(c.text):
                to = resolver.resolve(raw, e.rel)
                if to and to != e.rel and (c.ord, to) not in seen:
                    seen.add((c.ord, to))
                    links.append((c.ord, to))
        pending.append(Pending(src, e, chunks, links))
    return pending, deletes, skipped, ignored, head


def extended_meta(cfg: Config) -> dict[str, str | list | dict]:
    """Meta keys beyond the original set (model / dim / threshold / rrf_k / ignore / rule:* / url:*).

    Written only when `extended_meta` is on, so a deployment can switch to this code first and prove that the
    published database is unchanged before it starts shipping new keys. Values that equal the implicit
    defaults of older databases are still written, which makes the database self-describing."""
    out: dict[str, str | list | dict] = {
        "provider": cfg.provider,
        "api_key_env": cfg.api_key_env,
        "preamble_label": cfg.preamble_label,
    }
    if cfg.base_url:
        out["base_url"] = cfg.base_url
    if cfg.query_prefix:
        out["query_prefix"] = cfg.query_prefix
    for k, v in (cfg.client or {}).items():
        out[f"client:{k}"] = v
    for s in cfg.sources:
        if s.clone_hint:
            out[f"hint:{s.repo}"] = s.clone_hint
    return out


EXTENDED_KEYS = ("provider", "api_key_env", "preamble_label", "base_url", "query_prefix")
EXTENDED_PREFIXES = ("client:", "hint:")


def sync_extended_meta(store: Store, cfg: Config) -> None:
    """Make the extended keys in meta match the config exactly: stale ones (a removed client setting, or
    everything after switching extended_meta off) are deleted, so switching off is a real rollback."""
    want = extended_meta(cfg) if cfg.extended_meta else {}
    for (k,) in store.db.execute("select key from meta").fetchall():
        if (k in EXTENDED_KEYS or k.startswith(EXTENDED_PREFIXES)) and k not in want:
            store.db.execute("delete from meta where key=?", (k,))
    for k, v in want.items():
        store.set_meta(k, v)


def run(cfg: Config, ignore: IgnoreRules, store: Store, repos: list[str] | None = None,
        force: bool = False, dry_run: bool = False, embedder=None) -> RunReport:
    rep = RunReport(started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    t0 = time.time()
    all_pending: list[Pending] = []
    for src in cfg.sources:
        if repos and src.repo not in repos:
            continue
        try:
            pending, deletes, skipped, ignored, head = plan_source(cfg, ignore, store, src, force)
        except Exception as ex:
            rep.failures.append(f"{src.repo}: {ex}")
            log(f"! {src.repo}: {ex}")
            continue
        nchunks = sum(len(p.chunks) for p in pending)
        rep.per_repo[src.repo] = {"changed": len(pending), "deleted": len(deletes), "unchanged": skipped,
                                  "ignored": ignored, "new_chunks": nchunks, "head": head[:12]}
        log(f"- {src.repo}: changed={len(pending)} deleted={len(deletes)} unchanged={skipped} ignored={ignored} chunks={nchunks}")
        if dry_run:
            continue
        for p in deletes:
            store.delete_file(src.repo, p)
        store.set_meta(f"head:{src.repo}", head)
        store.set_meta(f"rule:{src.repo}", src.rule or "")
        store.set_meta(f"url:{src.repo}", src.url or "")
        all_pending.extend(pending)
    if dry_run:
        rep.elapsed = time.time() - t0
        return rep

    # ---- phase 1: fill the vector cache
    cache_key = cfg.cache_model
    hashes: dict[str, str] = {}
    for p in all_pending:
        for c in p.chunks:
            hashes.setdefault(c.hash, c.embed_text)
    have = store.cached_vectors(list(hashes), cache_key, cfg.dim)
    missing = [h for h in hashes if h not in have]
    rep.cached = len(hashes) - len(missing)
    if missing:
        emb = embedder or from_config(cfg)
        log(f"  embedding {len(missing)} chunks ({rep.cached} from cache) ...")
        for i in range(0, len(missing), cfg.batch):
            part = missing[i:i + cfg.batch]
            vecs = emb.embed_documents([hashes[h] for h in part])
            store.put_cache([(h, vec_to_blob(v)) for h, v in zip(part, vecs)], cache_key)
            rep.embedded += len(part)
            if (i // cfg.batch) % 10 == 0 or i + cfg.batch >= len(missing):
                log(f"    {min(i + cfg.batch, len(missing))}/{len(missing)}  {time.time() - t0:.0f}s")
        have = store.cached_vectors(list(hashes), cache_key, cfg.dim)

    # ---- phase 2: write chunks / fts / edges / files
    for p in all_pending:
        store.insert_file_chunks(p.src.repo, p.src.domain_of, p.entry.rel, p.entry.hash, p.entry.mtime,
                                 p.entry.size, p.chunks, have, p.links)
    store.set_meta("model", cfg.model)
    store.set_meta("dim", str(cfg.dim))
    store.set_meta("threshold", str(cfg.threshold))
    store.set_meta("rrf_k", str(cfg.rrf_k))
    # The exclusion patterns ship with the database: when a search finds nothing, clients can tell
    # "not in the corpus" apart from "excluded by .indexignore, look in the local checkout instead".
    store.set_meta("ignore", ignore.patterns)
    sync_extended_meta(store, cfg)
    store.set_meta("last_run", rep.started)
    store.set_meta("last_report", {"per_repo": rep.per_repo, "embedded": rep.embedded, "cached": rep.cached,
                                   "failures": rep.failures})
    store.commit()
    rep.elapsed = time.time() - t0
    log(f"done: embedded={rep.embedded} cached={rep.cached} elapsed={rep.elapsed:.0f}s")
    return rep
