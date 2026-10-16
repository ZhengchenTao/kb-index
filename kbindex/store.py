"""SQLite storage. Three main tables plus a cache and metadata:

  files      content hash per file, the basis of incremental indexing
  chunks     text + vector + structural edges (parent / prev / next)
  edges      link edges (chunk -> repo-relative path)
  chunks_fts FTS5 trigram full-text index (rowid = chunks.id)
  vec_cache  embed-text hash -> vector, so re-chunking or rebuilding never pays twice
  meta       watermarks, per-repo rules, thresholds, client settings; the MCP server needs only this file
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path

import numpy as np

SCHEMA = """
create table if not exists meta(key text primary key, value text);
create table if not exists files(
  repo text not null, path text not null, hash text not null,
  mtime real default 0, size integer default 0, nchunks integer default 0, indexed_at text,
  primary key(repo, path));
create table if not exists chunks(
  id integer primary key,
  repo text not null, domain text not null, path text not null, ord integer not null,
  heading text not null, text text not null, hash text not null,
  vector blob, depth integer default 0,
  parent_id integer, prev_id integer, next_id integer);
create index if not exists chunks_repo_path on chunks(repo, path, ord);
create index if not exists chunks_hash on chunks(hash);
create table if not exists edges(
  from_chunk integer not null, from_repo text not null, from_path text not null,
  to_path text not null, kind text not null default 'link');
create index if not exists edges_to on edges(from_repo, to_path);
create index if not exists edges_from on edges(from_chunk);
create table if not exists vec_cache(hash text primary key, model text not null, vector blob not null, created_at text);
create virtual table if not exists chunks_fts using fts5(heading, text, tokenize='trigram');
"""


def vec_to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def blob_to_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


class Store:
    def __init__(self, path: str | os.PathLike, readonly: bool = False):
        self.path = Path(path)
        if readonly:
            self.db = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path, check_same_thread=False)
            self.db.execute("pragma journal_mode=wal")
            self.db.executescript(SCHEMA)
        self.db.row_factory = sqlite3.Row

    # ---------------------------------------------------------------- meta
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        r = self.db.execute("select value from meta where key=?", (key,)).fetchone()
        return r[0] if r else default

    def set_meta(self, key: str, value) -> None:
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        self.db.execute("insert into meta(key,value) values(?,?) on conflict(key) do update set value=excluded.value", (key, value))

    def rules(self) -> dict[str, str]:
        rows = self.db.execute("select key, value from meta where key like 'rule:%'").fetchall()
        return {r[0][5:]: r[1] for r in rows}

    # ---------------------------------------------------------------- files
    def known_files(self, repo: str) -> dict[str, tuple[float, int, str]]:
        rows = self.db.execute("select path, mtime, size, hash from files where repo=?", (repo,)).fetchall()
        return {r[0]: (r[1], r[2], r[3]) for r in rows}

    def delete_file(self, repo: str, path: str) -> None:
        ids = [r[0] for r in self.db.execute("select id from chunks where repo=? and path=?", (repo, path))]
        if ids:
            q = ",".join("?" * len(ids))
            self.db.execute(f"delete from chunks_fts where rowid in ({q})", ids)
            self.db.execute(f"delete from edges where from_chunk in ({q})", ids)
            self.db.execute(f"delete from chunks where id in ({q})", ids)
        self.db.execute("delete from files where repo=? and path=?", (repo, path))

    def delete_repo(self, repo: str) -> None:
        self.db.execute("delete from chunks_fts where rowid in (select id from chunks where repo=?)", (repo,))
        self.db.execute("delete from edges where from_repo=?", (repo,))
        self.db.execute("delete from chunks where repo=?", (repo,))
        self.db.execute("delete from files where repo=?", (repo,))

    REPO_META_PREFIXES = ("head:", "rule:", "url:", "hint:")

    def delete_repo_meta(self, repo: str) -> None:
        """Drop the per-repo meta rows (head / rule / url / hint) of a repo that left the config."""
        for pre in self.REPO_META_PREFIXES:
            self.db.execute("delete from meta where key=?", (pre + repo,))

    def repos_in_db(self) -> set[str]:
        """Every repo the database still knows about: indexed files or per-repo meta rows."""
        out = {r[0] for r in self.db.execute("select distinct repo from files")}
        out |= {r[0] for r in self.db.execute("select distinct repo from chunks")}
        for (k,) in self.db.execute("select key from meta"):
            for pre in self.REPO_META_PREFIXES:
                if k.startswith(pre) and len(k) > len(pre):
                    out.add(k[len(pre):])
        return out

    def stale_repos(self, configured) -> list[str]:
        """Repos present in the database but no longer in the config. The indexer never touches them, so their
        chunks, rule and url stay searchable until `idx rebuild --repo R` removes them."""
        return sorted(self.repos_in_db() - set(configured))

    # ---------------------------------------------------------------- cache
    def cached_vectors(self, hashes: list[str], model: str, dim: int | None = None) -> dict[str, bytes]:
        """Cached vectors for these text hashes. With `dim`, a vector of the wrong width counts as a miss
        (protects against changing embedding.dim without purging the cache)."""
        out: dict[str, bytes] = {}
        want = dim * 4 if dim else None
        for i in range(0, len(hashes), 500):
            part = hashes[i:i + 500]
            q = ",".join("?" * len(part))
            for r in self.db.execute(f"select hash, vector from vec_cache where model=? and hash in ({q})", [model, *part]):
                if want is None or len(r[1]) == want:
                    out[r[0]] = r[1]
        return out

    def put_cache(self, items: list[tuple[str, bytes]], model: str) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.db.executemany(
            "insert or replace into vec_cache(hash, model, vector, created_at) values(?,?,?,?)",
            [(h, model, v, now) for h, v in items])
        self.db.commit()

    def purge_cache(self) -> None:
        self.db.execute("delete from vec_cache")

    # ---------------------------------------------------------------- chunks
    def insert_file_chunks(self, repo: str, domain_of, path: str, file_hash: str, mtime: float, size: int,
                           chunks, vectors: dict[str, bytes], links: list[tuple[int, str]]) -> None:
        """chunks: list[Chunk]; vectors: chunk.hash -> blob; links: (chunk_ord, to_path)."""
        self.delete_file(repo, path)
        ids: list[int] = []
        for c in chunks:
            cur = self.db.execute(
                "insert into chunks(repo,domain,path,ord,heading,text,hash,vector,depth) values(?,?,?,?,?,?,?,?,?)",
                (repo, domain_of(path), path, c.ord, c.heading, c.text, c.hash, vectors.get(c.hash), c.depth))
            ids.append(cur.lastrowid)
            self.db.execute("insert into chunks_fts(rowid, heading, text) values(?,?,?)", (cur.lastrowid, c.heading, c.text))
        from .chunker import parent_index
        for i, cid in enumerate(ids):
            pi = parent_index(chunks, i)
            self.db.execute("update chunks set parent_id=?, prev_id=?, next_id=? where id=?",
                            (ids[pi] if pi is not None else None,
                             ids[i - 1] if i > 0 else None,
                             ids[i + 1] if i + 1 < len(ids) else None, cid))
        for ord_, to_path in links:
            if 0 <= ord_ < len(ids):
                self.db.execute("insert into edges(from_chunk, from_repo, from_path, to_path, kind) values(?,?,?,?,'link')",
                                (ids[ord_], repo, path, to_path))
        self.db.execute(
            "insert into files(repo,path,hash,mtime,size,nchunks,indexed_at) values(?,?,?,?,?,?,?)",
            (repo, path, file_hash, mtime, size, len(chunks), time.strftime("%Y-%m-%dT%H:%M:%S")))

    def commit(self) -> None:
        self.db.commit()

    # ---------------------------------------------------------------- stats
    def stats(self) -> dict:
        out: dict = {"repos": {}}
        for r in self.db.execute("select repo, count(*) f, sum(nchunks) c from files group by repo"):
            out["repos"][r[0]] = {"files": r[1], "chunks": r[2] or 0}
        out["chunks"] = self.db.execute("select count(*) from chunks").fetchone()[0]
        out["missing_vectors"] = self.db.execute("select count(*) from chunks where vector is null").fetchone()[0]
        out["edges"] = self.db.execute("select count(*) from edges").fetchone()[0]
        out["cache"] = self.db.execute("select count(*) from vec_cache").fetchone()[0]
        out["meta"] = {r[0]: r[1] for r in self.db.execute("select key, value from meta where key not like 'rule:%'")}
        return out

    # ---------------------------------------------------------------- publish
    def publish(self, out_dir: str | os.PathLike) -> Path:
        """VACUUM INTO a clean copy (without vec_cache), atomically replace index.db, write its sha256.

        Also publishes a delta: the version number in meta is incremented, the new copy is compared file by
        file with the previous index.db into deltas/<version>.db.gz, and index.manifest.json is written last
        (clients read it first, see delta.py). Order: delta -> index.db -> manifest, so every file the
        manifest references is in place by the time a client can see it."""
        from . import delta as D
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / "index.db.tmp"
        final = out_dir / "index.db"
        if tmp.exists():
            tmp.unlink()
        old = sqlite3.connect(f"file:{final.as_posix()}?mode=ro", uri=True) if final.exists() else None
        old_ver = D.get_version(old) if old else 0
        self.set_meta("version", str(old_ver + 1))
        self.db.commit()
        self.db.execute("vacuum into ?", (tmp.as_posix(),))
        # The published copy drops vec_cache: vectors already live in chunks.vector, the cache only matters to
        # the indexer (it was about 30% of the file). The empty table stays so `idx status` works on it.
        pub = sqlite3.connect(tmp)
        pub.execute("delete from vec_cache")
        pub.commit()
        pub.execute("vacuum")
        new_sig = D.content_sig(pub)
        entry = None
        if old:
            ddir = out_dir / "deltas"
            ddir.mkdir(exist_ok=True)
            raw = ddir / f"{old_ver + 1}.db"
            stats = D.make_delta(old, pub, raw)
            gz = ddir / f"{old_ver + 1}.db.gz"
            D.gzip_file(raw, gz)
            raw.unlink()
            entry = {"from": old_ver, "from_sig": D.content_sig(old), "sig": new_sig,
                     "file": f"deltas/{gz.name}", "sha256": D.sha256_file(gz), "size": gz.stat().st_size, **stats}
            old.close()
        pub.close()
        sha = D.sha256_file(tmp)
        os.replace(tmp, final)
        (out_dir / "index.db.sha256").write_text(sha + "  index.db\n", encoding="ascii")
        D.write_manifest(out_dir, old_ver + 1, sha, new_sig, final.stat().st_size, entry)
        return final

    def close(self) -> None:
        self.db.close()
