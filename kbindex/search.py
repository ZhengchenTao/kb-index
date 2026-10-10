"""Search: vector + BM25 legs fused with RRF; the threshold only applies to the vector leg.

The threshold is a cosine (query <-> chunk). After RRF there is no cosine left, so:
  vector leg: candidates with cosine < threshold are dropped
  BM25 leg:   exact keyword hits are kept independently (identifier queries live on this leg)
  both empty: return "not found" honestly

Connections: the vector matrix and meta are cached in memory keyed on the db file's mtime; SQLite connections
are **opened per call and closed right after**. Holding the file open would prevent replacing index.db on
Windows while the server runs.

Thread safety (the HTTP server runs synchronous tools in a thread pool, several sessions call concurrently):
  - every public method **opens its connection first, then calls _ensure**. On Windows index.db cannot be
    replaced while a connection is open, so the vectors loaded for this call and the db file are always the
    same version (no "new db + old vector ids" mismatch).
  - _ensure is double-checked under a lock and reloads once; ids and V are swapped atomically as one tuple.
  - the lazy graph / doc_status / agents_paths caches are harmless to compute twice and take no lock; the
    embedding HTTP call is outside the lock, so a slow query never queues other sessions.

Client settings come from meta (written by the indexer from the `client:` section of the config):
  client:workspace       {"nt": "...", "posix": "..."} or a string: where local checkouts live
  client:agents_files    list of per-directory instruction file names (default ["AGENTS.md"])
  client:freshness_note  appended to "not found" notes (e.g. how often the index is refreshed)
  hint:<repo>            clone hint for a repo that is not a plain git clone
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass

import numpy as np

from .embed import from_meta
from .config import DEFAULT_PREAMBLE
from .store import Store, blob_to_vec

_TOKEN_SPLIT = re.compile(r"[\s,，。、；;：:？?！!（）()【】\[\]「」《》\"'`]+")

# ---- archived / deprecated signals (bilingual word lists; prefer misses over false alarms) -------------
# a whole directory name: archive/, _retired/, ...
_ARCHIVE_DIR = re.compile(r"(?i)^(archive|archived|_archive|deprecated|legacy|old|_retired|retired|已废弃|废弃|归档)$")
# file name contains a marker: "...-old-version.md", "...(rolled back).md", "deprecated-plan.md"
_ARCHIVE_FILE = re.compile(r"(已废弃|废弃|旧版|过时|已回滚|已作废|已退役|deprecated|legacy|superseded|obsolete)")
# a marker in the section heading: only explicit words, delimited by whitespace / punctuation / emoji or the
# end of the heading, so that phrases which merely contain the word ("reuse retired code") do not count
_SECTION_MARK = re.compile(
    r"(?:^|(?<=[^一-鿿\w]))(已废弃|已作废|已退役|已下线|已停用|已回滚|旧版|旧方案|历史方案|不再使用|归档备查|deprecated|superseded|retired|obsolete)"
    r"(?:$|(?=[^一-鿿\w]))|(已废弃|已作废|已退役|已下线|已停用)(?=[，,。）)：:]|$)")
_FRONTMATTER_STATUS = re.compile(r"(?m)^status\s*:\s*([A-Za-z_\-]+)\s*$")


def workspace_root(meta: dict | None = None) -> str | None:
    """Where local checkouts live: $KB_WORKSPACE, else meta client:workspace (per os.name), else None.
    With no workspace configured, results carry no local_path / clone hints."""
    ws = os.environ.get("KB_WORKSPACE")
    if ws:
        return ws
    raw = (meta or {}).get("client:workspace")
    if not raw:
        return None
    try:
        val = json.loads(raw)
    except ValueError:
        val = raw
    if isinstance(val, dict):
        val = val.get(os.name) or val.get("default")
    if not val or not isinstance(val, str):
        return None
    return os.path.expandvars(os.path.expanduser(val))


def _json_meta(meta: dict, key: str, default):
    raw = meta.get(key)
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except ValueError:
        return default


@dataclass
class Hit:
    chunk_id: int
    repo: str
    domain: str
    path: str
    heading: str
    snippet: str
    score: float
    vec_score: float | None = None
    bm25: bool = False
    prev_id: int | None = None
    next_id: int | None = None
    extra: dict | None = None   # relation annotations (links / status), see Searcher.annotate

    def to_dict(self) -> dict:
        d = {
            "repo": self.repo, "domain": self.domain, "path": self.path, "heading": self.heading,
            "score": round(self.score, 4), "chunk_id": self.chunk_id, "snippet": self.snippet,
        }
        if self.vec_score is not None:
            d["vec_score"] = round(self.vec_score, 3)
        if self.bm25:
            d["bm25_hit"] = True
        if self.extra:
            d.update(self.extra)
        return d


class Searcher:
    """Read-only. Keeps the vector matrix in memory and reloads it when the db file changes."""

    def __init__(self, db_path: str | os.PathLike, embedder=None):
        self.db_path = str(db_path)
        self._embedder = embedder
        self._loaded_mtime = -1.0
        self._lock = threading.Lock()
        self._vec: tuple[np.ndarray, np.ndarray] = (np.zeros(0, dtype=np.int64), np.zeros((0, 1), dtype=np.float32))
        self.meta: dict[str, str] = {}
        self._agents: dict[str, set] | None = None   # repo -> instruction file paths; invalidated with the db
        self._graph = None
        self._docstat = None
        self._ensure()

    @property
    def ids(self) -> np.ndarray:
        return self._vec[0]

    @property
    def V(self) -> np.ndarray:
        return self._vec[1]

    # ---------------------------------------------------------------- load
    def _ensure(self) -> None:
        try:
            m = os.path.getmtime(self.db_path)
        except OSError:
            raise RuntimeError(f"index.db not found: {self.db_path}")
        if m == self._loaded_mtime:
            return
        with self._lock:
            if m == self._loaded_mtime:   # another thread reloaded while we waited for the lock
                return
            st = Store(self.db_path, readonly=True)
            try:
                db = st.db
                meta = {r[0]: r[1] for r in db.execute("select key, value from meta")}
                rows = db.execute("select id, vector from chunks where vector is not null").fetchall()
            finally:
                st.close()
            if rows:
                vec = (np.array([r[0] for r in rows], dtype=np.int64), np.stack([blob_to_vec(r[1]) for r in rows]))
            else:
                vec = (np.zeros(0, dtype=np.int64), np.zeros((0, int(meta.get("dim", 1536))), dtype=np.float32))
            del rows
            self.meta = meta
            self._vec = vec
            self._agents = None
            self._graph = None
            self._docstat = None
            self._loaded_mtime = m

    # ---------------------------------------------------------------- relations
    def graph(self, db: sqlite3.Connection) -> tuple[dict, dict]:
        """Edges folded into two maps: (repo, path) -> {to_path: n} and (repo, to_path) -> {from_path: n}.
        A few milliseconds for thousands of edges; invalidated with the db."""
        if self._graph is None:
            out: dict[tuple[str, str], dict[str, int]] = {}
            inc: dict[tuple[str, str], dict[str, int]] = {}
            for r in db.execute("select from_repo, from_path, to_path, count(*) n from edges group by from_repo, from_path, to_path"):
                out.setdefault((r[0], r[1]), {})[r[2]] = r[3]
                inc.setdefault((r[0], r[2]), {})[r[1]] = r[3]
            self._graph = (out, inc)
        return self._graph

    def doc_status(self, db: sqlite3.Connection) -> dict[tuple[str, str], str]:
        """`status:` from YAML frontmatter (e.g. active / draft / implemented)."""
        if self._docstat is None:
            m: dict[tuple[str, str], str] = {}
            for r in db.execute("select repo, path, substr(text, 1, 600) t from chunks where ord=0"):
                mm = _FRONTMATTER_STATUS.search(r[2])
                if mm:
                    m[(r[0], r[1])] = mm.group(1).lower()
            self._docstat = m
        return self._docstat

    def annotate(self, db: sqlite3.Connection, repo: str, path: str, heading: str = "", top: int = 3) -> dict:
        """Relation annotations for one hit: links (who links here / where it links to) + status (archived).

        Why inline them into search / grep instead of relying on neighbors: in practice agents almost never call
        neighbors during real tasks; once they have a path they open the file. Relation signals only get used
        when they come back with the hit. Costs about 30-60 tokens per hit."""
        out_m, in_m = self.graph(db)
        outs = out_m.get((repo, path), {})
        ins = in_m.get((repo, path), {})
        d: dict = {}
        links: dict = {}
        if ins:
            links["in"] = len(ins)
            links["in_top"] = [p for p, _ in sorted(ins.items(), key=lambda x: -x[1])[:top]]
        if outs:
            links["out"] = len(outs)
            links["out_top"] = [p for p, _ in sorted(outs.items(), key=lambda x: -x[1])[:top]]
        if links:
            d["links"] = links
        reasons: list[str] = []
        parts = path.split("/")
        for seg in parts[:-1]:
            if _ARCHIVE_DIR.match(seg):
                reasons.append(f"directory {seg}/")
                break
        if _ARCHIVE_FILE.search(parts[-1]):
            reasons.append("file name")
        last = heading.split(" > ")[-1] if heading else ""
        if last and _SECTION_MARK.search(last):
            reasons.append(f"section heading \"{last[:40]}\"")
        # "Linked from a deprecated-plans list" is NOT a signal: list-style documents mostly link to current
        # documents.
        if reasons:
            d["status"] = "archived"
            d["status_reason"] = "; ".join(reasons)
        fs = self.doc_status(db).get((repo, path))
        if fs and fs != "active":
            d["doc_status"] = fs
        return d

    def _open(self) -> Store:
        """One read-only connection per call; the caller closes it in finally."""
        return Store(self.db_path, readonly=True)

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = from_meta(self.meta)
        return self._embedder

    @property
    def threshold(self) -> float:
        return float(self.meta.get("threshold", 0.70))

    @property
    def rrf_k(self) -> int:
        return int(self.meta.get("rrf_k", 60))

    def rules(self) -> dict[str, str]:
        return {k[5:]: v for k, v in self.meta.items() if k.startswith("rule:") and v}

    def catalog(self) -> dict[str, list[str]]:
        """Distinct repos and domains in the database (for tool descriptions)."""
        st = self._open()
        try:
            rows = st.db.execute("select distinct repo, domain from chunks").fetchall()
        finally:
            st.close()
        return {"repos": sorted({r[0] for r in rows}), "domains": sorted({r[1] for r in rows})}

    def _note(self, text: str) -> str:
        extra = self.meta.get("client:freshness_note", "")
        return f"{text} {extra}" if extra else text

    def excluded(self, repos=None) -> list[str]:
        """The .indexignore patterns the indexer published, minus generic noise (dot directories, `**/` rules).
        Returned when nothing matched, so the caller can tell "not in the corpus" from "excluded from the
        index, look in the local checkout"."""
        try:
            pats = json.loads(self.meta.get("ignore", "[]"))
        except ValueError:
            return []
        out = []
        for p in pats:
            if p.startswith("**/") or re.search(r"/\.[^/]+/?$", p):
                continue
            if repos and p.split("/", 1)[0] not in repos:
                continue
            out.append(p)
        return out

    @property
    def agents_files(self) -> list[str]:
        files = _json_meta(self.meta, "client:agents_files", None)
        if isinstance(files, list) and files and all(isinstance(f, str) for f in files):
            return files
        return ["AGENTS.md"]

    def agents_paths(self, db: sqlite3.Connection) -> dict[str, set]:
        """Repo-relative paths of every instruction file (AGENTS.md by default) in each repo, taken from the
        index rather than the file system, so it works for repos that are not checked out locally."""
        if self._agents is None:
            out: dict[str, set] = {}
            names = self.agents_files
            for repo, path in db.execute("select distinct repo, path from chunks"):
                if path.rsplit("/", 1)[-1] in names:
                    out.setdefault(repo, set()).add(path)
            self._agents = out
        return self._agents

    def agents_chain(self, db: sqlite3.Connection, repo: str, path: str) -> list[str]:
        """Which instruction files apply to a path: the root one plus one per directory level down to the file,
        listed only when they exist. Reading only the root file misses rules of partitioned repos."""
        have = self.agents_paths(db).get(repo, set())
        names = self.agents_files
        chain, parts = [], path.split("/")[:-1]
        for i in range(len(parts) + 1):
            for name in names:
                cand = "/".join(parts[:i] + [name])
                if cand in have and cand != path:
                    chain.append(cand)
        return chain

    def agents_chains(self, db: sqlite3.Connection, pairs) -> dict[str, list[str]]:
        """Merge the chains of several hits: repo -> de-duplicated chain, root first."""
        out: dict[str, list[str]] = {}
        for repo, path in pairs:
            for a in self.agents_chain(db, repo, path):
                lst = out.setdefault(repo, [])
                if a not in lst:
                    lst.append(a)
        for lst in out.values():
            lst.sort(key=lambda x: x.count("/"))
        return out

    def repo_info(self, repos) -> dict[str, dict]:
        """For every repo in the hits: its rule, canonical clone URL and, when a workspace root is configured,
        where it should live on this machine (<workspace>/<repo>) and whether it is there."""
        ws = workspace_root(self.meta)
        out: dict[str, dict] = {}
        for r in dict.fromkeys(repos):
            info: dict = {"rule": self.meta.get(f"rule:{r}", "")}
            url = self.meta.get(f"url:{r}", "")
            hint = self.meta.get(f"hint:{r}", "")
            if ws:
                local = os.path.join(ws, r)
                info["local_path"] = local
                info["local_exists"] = os.path.isdir(local)
            if url:
                info["clone_url"] = url
            if ws and not info["local_exists"]:
                if hint:
                    info["clone_hint"] = hint
                elif url:
                    info["clone_hint"] = f"git clone {url} \"{info['local_path']}\""
            out[r] = info
        return out

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _row(db: sqlite3.Connection, cid: int) -> sqlite3.Row | None:
        return db.execute(
            "select id, repo, domain, path, heading, text, prev_id, next_id, parent_id from chunks where id=?", (cid,)).fetchone()

    @staticmethod
    def _filter_sql(repos, domains) -> tuple[str, list]:
        cond, args = [], []
        if repos:
            cond.append(f"repo in ({','.join('?' * len(repos))})")
            args += list(repos)
        if domains:
            cond.append(f"domain in ({','.join('?' * len(domains))})")
            args += list(domains)
        return (" and " + " and ".join(cond)) if cond else "", args

    def _allowed_ids(self, db: sqlite3.Connection, repos, domains) -> np.ndarray | None:
        if not repos and not domains:
            return None
        where, args = self._filter_sql(repos, domains)
        rows = db.execute(f"select id from chunks where 1=1{where}", args).fetchall()
        return np.array([r[0] for r in rows], dtype=np.int64)

    @staticmethod
    def _snippet(text: str, n: int = 280) -> str:
        t = " ".join(text.split())
        return t if len(t) <= n else t[:n] + "…"

    # ---------------------------------------------------------------- legs
    def vector_leg(self, db: sqlite3.Connection, query: str, k: int, repos=None, domains=None,
                   min_score: float | None = None) -> list[tuple[int, float]]:
        ids, V = self._vec
        if len(ids) == 0:
            return []
        q = self.embedder.embed_query(query)
        s = V @ q
        allowed = self._allowed_ids(db, repos, domains)
        if allowed is not None:
            mask = np.isin(ids, allowed)
            s = np.where(mask, s, -1.0)
        idx = np.argsort(-s)[:k]
        floor = self.threshold if min_score is None else min_score
        return [(int(ids[i]), float(s[i])) for i in idx if s[i] >= floor]

    @staticmethod
    def fts_expr(query: str) -> str:
        terms = [t for t in _TOKEN_SPLIT.split(query) if len(t) >= 3]
        if not terms and len(query.strip()) >= 3:
            terms = [query.strip()]
        return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)

    def bm25_leg(self, db: sqlite3.Connection, query: str, k: int, repos=None, domains=None) -> list[tuple[int, float]]:
        expr = self.fts_expr(query)
        if not expr:
            return []
        where, args = self._filter_sql(repos, domains)
        try:
            rows = db.execute(
                f"select f.rowid, f.rank from chunks_fts f join chunks c on c.id=f.rowid "
                f"where chunks_fts match ?{where} order by f.rank limit ?", [expr, *args, k]).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(int(r[0]), float(-r[1])) for r in rows]

    # ---------------------------------------------------------------- search
    def search(self, query: str, k: int = 5, repos=None, domains=None, expand: bool = False) -> dict:
        t0 = time.time()
        st = self._open()
        try:
            self._ensure()
            return self._search(st.db, query, k, repos, domains, expand, t0)
        finally:
            st.close()

    def _search(self, db: sqlite3.Connection, query: str, k: int, repos, domains, expand: bool, t0: float) -> dict:
        vec = self.vector_leg(db, query, max(k * 4, 20), repos, domains)
        bm = self.bm25_leg(db, query, max(k * 4, 20), repos, domains)
        if not vec and not bm:
            return {"found": False, "query": query, "hits": [], "threshold": self.threshold,
                    "note": self._note("No hits on either leg: vector similarity below the threshold and no exact "
                                       "keyword match. Rephrase, or use grep for identifiers. If the target may be under "
                                       "a path listed in `excluded` (never indexed), read the local checkout directly."),
                    "excluded": self.excluded(repos)}
        K = self.rrf_k
        fused: dict[int, float] = {}
        for rank, (cid, _) in enumerate(vec, 1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (K + rank)
        for rank, (cid, _) in enumerate(bm, 1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (K + rank)
        vs = dict(vec)
        bms = {cid for cid, _ in bm}
        top = sorted(fused.items(), key=lambda x: -x[1])[:k]
        hits: list[Hit] = []
        seen_paths: set[tuple[str, str]] = set()
        for cid, sc in top:
            r = self._row(db, cid)
            if not r:
                continue
            text = r["text"]
            if expand:
                parts = []
                for nid in (r["prev_id"], cid, r["next_id"]):
                    if nid:
                        rr = self._row(db, nid)
                        if rr:
                            parts.append(rr["text"])
                text = "\n…\n".join(parts)
            hits.append(Hit(cid, r["repo"], r["domain"], r["path"], r["heading"], self._snippet(text, 600 if expand else 280),
                            sc, vs.get(cid), cid in bms, r["prev_id"], r["next_id"],
                            self.annotate(db, r["repo"], r["path"], r["heading"])))
            # several hits in one document: only the first carries links, later ones keep status (saves tokens)
            key = (r["repo"], r["path"])
            if key in seen_paths and hits[-1].extra:
                hits[-1].extra.pop("links", None)
            seen_paths.add(key)
        rules = self.rules()
        notes = {h.repo: rules[h.repo] for h in hits if h.repo in rules}
        return {"found": True, "query": query, "hits": [h.to_dict() for h in hits], "threshold": self.threshold,
                "vector_hits": len(vec), "bm25_hits": len(bm), "elapsed_ms": int((time.time() - t0) * 1000),
                "repo_rules": notes, "repos": self.repo_info([h.repo for h in hits]),
                "agents_chain": self.agents_chains(db, [(h.repo, h.path) for h in hits])}

    # ---------------------------------------------------------------- grep
    def grep(self, pattern: str, k: int = 20, repos=None, domains=None, ignore_case: bool = True) -> dict:
        try:
            rx = re.compile(pattern, re.I if ignore_case else 0)
        except re.error as e:
            return {"error": f"bad regex: {e}"}
        st = self._open()
        try:
            self._ensure()
            return self._grep(st.db, rx, pattern, k, repos, domains)
        finally:
            st.close()

    def _grep(self, db: sqlite3.Connection, rx: re.Pattern, pattern: str, k: int, repos, domains) -> dict:
        where, args = self._filter_sql(repos, domains)
        # Pre-filter with FTS on the longest literal fragment of the pattern (>= 3 chars), then apply the regex;
        # without such a fragment scan the whole table (well under a second for tens of thousands of rows)
        lit = max(re.split(r"[\\\[\]().*+?^$|{}]+", pattern), key=len, default="")
        rows = None
        if len(lit) >= 3:
            try:
                rows = db.execute(
                    f"select c.id, c.repo, c.domain, c.path, c.heading, c.text from chunks_fts f join chunks c on c.id=f.rowid "
                    f"where chunks_fts match ?{where} limit 2000", ['"' + lit.replace('"', '""') + '"', *args]).fetchall()
            except sqlite3.OperationalError:
                rows = None
        if rows is None:
            rows = db.execute(
                f"select id, repo, domain, path, heading, text from chunks where 1=1{where}", args).fetchall()
        out = []
        for r in rows:
            for ln in r["text"].split("\n"):
                if rx.search(ln):
                    m = {"repo": r["repo"], "domain": r["domain"], "path": r["path"], "heading": r["heading"],
                         "line": ln.strip()[:300], "chunk_id": r["id"]}
                    # grep returns ~20 hits: only status and the inbound link count, no path lists (saves tokens)
                    a = self.annotate(db, r["repo"], r["path"], r["heading"], top=0)
                    if "links" in a and "in" in a["links"]:
                        m["links_in"] = a["links"]["in"]
                    for key in ("status", "status_reason", "doc_status"):
                        if key in a:
                            m[key] = a[key]
                    out.append(m)
                    break
            if len(out) >= k:
                break
        # Rules travel with content: grep returns rule / repo_info for the repos it hit, like the other tools.
        # Otherwise "grep once and quote the line" would silently bypass a repo's rule.
        rules = self.rules()
        hit_repos = [m["repo"] for m in out]
        if not out:
            return {"pattern": pattern, "matches": [], "truncated": False, "excluded": self.excluded(repos),
                    "note": self._note("No match. Paths listed in `excluded` are never indexed; the target may be "
                                       "there, search the local checkout directly.")}
        return {"pattern": pattern, "matches": out, "truncated": len(out) >= k,
                "repo_rules": {r: rules[r] for r in dict.fromkeys(hit_repos) if r in rules},
                "repos": self.repo_info(hit_repos),
                "agents_chain": self.agents_chains(db, [(m["repo"], m["path"]) for m in out])}

    # ---------------------------------------------------------------- neighbors
    def neighbors(self, path: str, repo: str | None = None, k: int = 10) -> dict:
        st = self._open()
        try:
            self._ensure()
            return self._neighbors(st.db, path, repo, k)
        finally:
            st.close()

    def _neighbors(self, db: sqlite3.Connection, path: str, repo: str | None, k: int) -> dict:
        cond = "path=?" + (" and repo=?" if repo else "")
        args = [path] + ([repo] if repo else [])
        own = db.execute(f"select id, repo, domain, vector from chunks where {cond}", args).fetchall()
        if not own:
            like = db.execute("select distinct repo, path from chunks where path like ? limit 10", (f"%{path}%",)).fetchall()
            return {"found": False, "path": path, "suggestions": [dict(r) for r in like]}
        repo = own[0]["repo"]
        out_links = db.execute(
            "select to_path, count(*) n from edges where from_repo=? and from_path=? group by to_path order by n desc limit ?",
            (repo, path, k)).fetchall()
        in_links = db.execute(
            "select from_path, count(*) n from edges where from_repo=? and to_path=? group by from_path order by n desc limit ?",
            (repo, path, k)).fetchall()
        # semantic neighbours: this file's centroid against other chunks (auxiliary, weaker than the link graph)
        vecs = [blob_to_vec(r["vector"]) for r in own if r["vector"] is not None]
        similar = []
        ids, V = self._vec
        if vecs and len(ids) > 0:
            c = np.mean(np.stack(vecs), axis=0)
            c /= (np.linalg.norm(c) or 1.0)
            s = V @ c
            best: dict[tuple[str, str], float] = {}
            for i in np.argsort(-s)[:200]:
                r = self._row(db, int(ids[i]))
                if not r or (r["repo"] == repo and r["path"] == path):
                    continue
                key = (r["repo"], r["path"])
                if key not in best:
                    best[key] = float(s[i])
                if len(best) >= k:
                    break
            similar = [{"repo": a, "path": b, "score": round(v, 3)} for (a, b), v in best.items()]
        rules = self.rules()

        def _entry(p: str, n: int) -> dict:
            e = {"path": p, "n": n}
            a = self.annotate(db, repo, p, top=0)
            for key in ("status", "status_reason", "doc_status"):
                if key in a:
                    e[key] = a[key]
            return e

        me = self.annotate(db, repo, path, top=0)
        return {"found": True, "repo": repo, "path": path, "domain": own[0]["domain"],
                **{k: v for k, v in me.items() if k != "links"},
                "links_out": [_entry(r[0], r[1]) for r in out_links],
                "links_in": [_entry(r[0], r[1]) for r in in_links],
                "similar": similar,
                "rule": rules.get(repo, ""), "repo_info": self.repo_info([repo])[repo],
                "agents_chain": self.agents_chain(db, repo, path)}

    # ---------------------------------------------------------------- read_doc
    def read_doc(self, path: str, repo: str | None = None, max_chars: int = 40000) -> dict:
        st = self._open()
        try:
            self._ensure()
            return self._read_doc(st.db, path, repo, max_chars)
        finally:
            st.close()

    def _read_doc(self, db: sqlite3.Connection, path: str, repo: str | None, max_chars: int) -> dict:
        cond = "path=?" + (" and repo=?" if repo else "")
        args = [path] + ([repo] if repo else [])
        rows = db.execute(f"select repo, domain, heading, text, depth from chunks where {cond} order by repo, ord", args).fetchall()
        if not rows:
            like = db.execute("select distinct repo, path from chunks where path like ? limit 10", (f"%{path}%",)).fetchall()
            return {"found": False, "path": path, "suggestions": [dict(r) for r in like]}
        repo = rows[0]["repo"]
        rows = [r for r in rows if r["repo"] == repo]
        parts: list[str] = []
        last_heading = None
        preamble = self.meta.get("preamble_label") or DEFAULT_PREAMBLE
        for r in rows:
            if r["heading"] != last_heading:
                title = r["heading"].split(" > ")[-1]
                if title and title != preamble and r["depth"] > 0:
                    parts.append("#" * min(int(r["depth"]), 4) + " " + title)
                last_heading = r["heading"]
            parts.append(r["text"])
        body = "\n\n".join(parts)
        truncated = len(body) > max_chars
        rules = self.rules()
        f = db.execute("select hash, indexed_at from files where repo=? and path=?", (repo, path)).fetchone()
        return {"found": True, "repo": repo, "path": path, "domain": rows[0]["domain"],
                "rule": rules.get(repo, ""), "repo_info": self.repo_info([repo])[repo],
                "agents_chain": self.agents_chain(db, repo, path),
                "indexed_at": f["indexed_at"] if f else None,
                "note": "Reassembled from the index (very short fragments are dropped), not the verbatim file; "
                        "read the local checkout when you have it.",
                "text": body[:max_chars], "truncated": truncated}
