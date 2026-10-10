"""Incremental publishing and pulling.

Why: with frequent publishes, a client on a slow link would otherwise download the whole database (hundreds
of MB) every time, while a typical round changes a dozen files and a few hundred chunks (a few MB, mostly
vectors).

How: the indexer always replaces a file's chunks wholesale and chunks.id is an integer primary key (VACUUM
keeps it), so a delta is a per-file comparison of the previous and the new published database. Changed and
removed files, with their chunks (including vectors), edges and files rows, go into a small SQLite file
together with the complete meta table. The client deletes and re-inserts those files on a copy and replaces
meta wholesale.

Consistency is checked with a content signature, not the file sha: after applying deltas the client's file
bytes differ from the publisher's (no VACUUM). Signature = a strong digest per file (see file_sigs) + all of
meta, hashed with sha256. Each delta records its starting signature `from_sig`; a client whose local
signature does not match must not apply it and falls back to a full download.

The signature algorithm is a wire format shared by publisher and clients running different versions of this
code. Changing it forces every client into a full download.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sqlite3
from pathlib import Path

KEEP_DELTAS = 40   # deltas kept on the publisher; usually 1-5 MB each gzipped, 40 covers days of frequent publishes

_DELTA_SCHEMA = """
create table meta(key text primary key, value text);
create table del(repo text not null, path text not null);
create table files(repo text, path text, hash text, mtime real, size integer, nchunks integer, indexed_at text);
create table chunks(id integer primary key, repo text, domain text, path text, ord integer, heading text, text text,
                    hash text, vector blob, depth integer, parent_id integer, prev_id integer, next_id integer);
create table edges(from_chunk integer, from_repo text, from_path text, to_path text, kind text);
"""
_CHUNK_COLS = "id, repo, domain, path, ord, heading, text, hash, vector, depth, parent_id, prev_id, next_id"
_FILE_COLS = "repo, path, hash, mtime, size, nchunks, indexed_at"
_EDGE_COLS = "from_chunk, from_repo, from_path, to_path, kind"


def file_sigs(db: sqlite3.Connection) -> dict[tuple[str, str], str]:
    """(repo, path) -> digest of everything the database holds for that file: files.hash + every chunk's
    id / domain / ord / heading / embed-text hash / structural edges / **vector bytes** + its link edges.

    files.hash (the source file hash) alone is not enough: after a model change (rebuild --purge-cache), a
    domain mapping change, or a chunking change followed by a rebuild, the source file is untouched while its
    vectors / domain / chunks all changed. Comparing only files.hash would leave those files out of the delta
    and the client would pair new meta with old vectors while the signatures still matched.
    min(id) is not enough either: without AUTOINCREMENT, deleting a tail range and inserting again reuses ids.
    About 0.5 s for 15k chunks."""
    per: dict[tuple[str, str], "hashlib._Hash"] = {}
    for r in db.execute("select repo, path, id, domain, ord, heading, hash, depth, parent_id, prev_id, next_id, vector "
                        "from chunks order by repo, path, ord, id"):
        r = tuple(r)
        h = per.get((r[0], r[1]))
        if h is None:
            h = per[(r[0], r[1])] = hashlib.sha1()
        h.update(repr(r[2:11]).encode("utf-8"))
        h.update(r[11] or b"")
    for r in db.execute("select from_repo, from_path, from_chunk, to_path, kind from edges "
                        "order by from_repo, from_path, from_chunk, to_path, kind"):
        r = tuple(r)
        h = per.get((r[0], r[1]))
        if h is None:
            h = per[(r[0], r[1])] = hashlib.sha1()
        h.update(repr(("edge",) + r[2:]).encode("utf-8"))
    out = {}
    for r in db.execute("select repo, path, hash from files"):
        k = (r[0], r[1])
        out[k] = f"{r[2]}:{per[k].hexdigest() if k in per else '-'}"
    return out


def _sig(db: sqlite3.Connection, skip_meta: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    for k, v in sorted(file_sigs(db).items()):
        h.update(json.dumps([k[0], k[1], v], ensure_ascii=False).encode())
    for k, v in sorted(tuple(r) for r in db.execute("select key, value from meta")):
        if not k.startswith(skip_meta):
            h.update(json.dumps([k, v], ensure_ascii=False).encode())
    return h.hexdigest()


def content_sig(db: sqlite3.Connection) -> str:
    """Client-side check: all of meta counts; after applying deltas it must equal the published database."""
    return _sig(db, ())


# Meta keys ignored when deciding whether to publish: last_* change every run, head:* change when a repo only
# touched files that are not indexed, version is written by publish itself.
VOLATILE_META = ("last_run", "last_report", "head:", "version")


def state_sig(db: sqlite3.Connection) -> str:
    """Publish decision: does the work database differ from the last published one in anything clients read?"""
    return _sig(db, VOLATILE_META)


def get_version(db: sqlite3.Connection) -> int:
    r = db.execute("select value from meta where key='version'").fetchone()
    try:
        return int(r[0]) if r else 0
    except (TypeError, ValueError):
        return 0


def make_delta(old: sqlite3.Connection, new: sqlite3.Connection, out_path: str | os.PathLike) -> dict:
    """Write the old -> new difference to out_path (an SQLite file). Returns counts."""
    so, sn = file_sigs(old), file_sigs(new)
    changed = [k for k, v in sn.items() if so.get(k) != v]
    removed = [k for k in so if k not in sn]
    out_path = Path(out_path)
    if out_path.exists():
        out_path.unlink()
    d = sqlite3.connect(out_path)
    try:
        d.executescript(_DELTA_SCHEMA)
        d.executemany("insert into del(repo, path) values(?,?)", changed + removed)
        nchunks = 0
        for repo, path in changed:
            d.executemany(f"insert into files({_FILE_COLS}) values(?,?,?,?,?,?,?)",
                          new.execute(f"select {_FILE_COLS} from files where repo=? and path=?", (repo, path)).fetchall())
            rows = new.execute(f"select {_CHUNK_COLS} from chunks where repo=? and path=?", (repo, path)).fetchall()
            d.executemany(f"insert into chunks({_CHUNK_COLS}) values({','.join('?' * 13)})", rows)
            nchunks += len(rows)
            d.executemany(f"insert into edges({_EDGE_COLS}) values(?,?,?,?,?)",
                          new.execute(f"select {_EDGE_COLS} from edges where from_repo=? and from_path=?", (repo, path)).fetchall())
        d.executemany("insert into meta(key, value) values(?,?)", new.execute("select key, value from meta").fetchall())
        d.commit()
    finally:
        d.close()
    return {"changed": len(changed), "removed": len(removed), "chunks": nchunks}


def gzip_file(src: str | os.PathLike, dst: str | os.PathLike) -> None:
    with open(src, "rb") as fi, gzip.open(dst, "wb", compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)


def sha256_file(p: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def apply_delta(db: sqlite3.Connection, delta_path: str | os.PathLike) -> dict:
    """Apply one delta to db in a single transaction. delta_path may be gzipped (.gz)."""
    src = str(delta_path)
    tmp = None
    if src.endswith(".gz"):
        tmp = src[:-3]
        with gzip.open(src, "rb") as fi, open(tmp, "wb") as fo:
            shutil.copyfileobj(fi, fo, 1 << 20)
        src = tmp
    try:
        d = sqlite3.connect(f"file:{Path(src).as_posix()}?mode=ro", uri=True)
        try:
            dels = d.execute("select repo, path from del").fetchall()
            files = d.execute(f"select {_FILE_COLS} from files").fetchall()
            chunks = d.execute(f"select {_CHUNK_COLS} from chunks").fetchall()
            edges = d.execute(f"select {_EDGE_COLS} from edges").fetchall()
            meta = d.execute("select key, value from meta").fetchall()
        finally:
            d.close()
        with db:
            for repo, path in dels:
                ids = [r[0] for r in db.execute("select id from chunks where repo=? and path=?", (repo, path))]
                for i in range(0, len(ids), 500):
                    part = ids[i:i + 500]
                    q = ",".join("?" * len(part))
                    db.execute(f"delete from chunks_fts where rowid in ({q})", part)
                    db.execute(f"delete from edges where from_chunk in ({q})", part)
                    db.execute(f"delete from chunks where id in ({q})", part)
                db.execute("delete from files where repo=? and path=?", (repo, path))
            db.executemany(f"insert into files({_FILE_COLS}) values(?,?,?,?,?,?,?)", files)
            db.executemany(f"insert into chunks({_CHUNK_COLS}) values({','.join('?' * 13)})", chunks)
            db.executemany("insert into chunks_fts(rowid, heading, text) values(?,?,?)", [(c[0], c[5], c[6]) for c in chunks])
            db.executemany(f"insert into edges({_EDGE_COLS}) values(?,?,?,?,?)", edges)
            db.execute("delete from meta")
            db.executemany("insert into meta(key, value) values(?,?)", meta)
        return {"deleted": len(dels), "files": len(files), "chunks": len(chunks)}
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def write_manifest(out_dir: Path, version: int, sha: str, sig: str, size: int, new_delta: dict | None) -> None:
    """index.manifest.json, the first thing clients read. Deltas are keyed by target version; from_sig is the
    signature a client must have to apply one."""
    mf = out_dir / "index.manifest.json"
    deltas: dict = {}
    # No delta (first publish, or index.db was deleted and published again) means versions restart at 1: the
    # old chain is void and its files are removed, otherwise old "2".."40" would collide with new numbers.
    if new_delta and mf.exists():
        try:
            deltas = json.loads(mf.read_text(encoding="utf-8")).get("deltas", {})
        except ValueError:
            deltas = {}
    if new_delta:
        deltas[str(version)] = new_delta
    keep = sorted((int(k) for k in deltas), reverse=True)[:KEEP_DELTAS]
    deltas = {str(k): deltas[str(k)] for k in sorted(keep)}
    ddir = out_dir / "deltas"
    if ddir.exists():   # remove deltas the manifest no longer references
        live = {v["file"] for v in deltas.values()}
        for p in ddir.iterdir():
            if f"deltas/{p.name}" not in live:
                p.unlink()
    body = {"version": version, "sha256": sha, "sig": sig, "size": size, "deltas": deltas}
    tmp = out_dir / "index.manifest.json.tmp"
    tmp.write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, mf)
