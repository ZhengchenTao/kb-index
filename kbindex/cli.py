"""idx - indexer and debugging CLI.

  idx index   [--repo R ...] [--force] [--dry-run] [--no-publish] [--publish-always]
                                                      incremental index; publishes only when something changed
  idx rebuild [--repo R ...] [--purge-cache]          drop chunks and rebuild; the vector cache is kept (free)
  idx status                                          files / chunks / last run / failures
  idx query   "some question" [-k 5] [--repo R] [--domain D] [--expand] [-v]
                                                      try a query and print both legs' scores
  idx grep    PATTERN [-k 20]
  idx neighbors PATH
  idx read    PATH
  idx ignore  [--repo R]                              which files the exclusion rules hit
  idx calibrate --queries FILE                        measure the vector threshold for the current model
  idx publish                                         VACUUM INTO index.db + sha256 + delta + manifest
  idx sig         --db PATH                           version and content signature of a database
  idx apply-delta --db PATH DELTA.db.gz ...           apply deltas in order, print the resulting signature

Databases: the indexer commands (index / rebuild / status / publish) use --db > $KB_DB > ./work.db and publish
into $KB_OUT (default: the work database's directory). The query commands (query / grep / neighbors / read /
calibrate) use --db > $KB_DB > ./index.db, falling back to ./work.db.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .config import IgnoreRules, load_config
from .store import Store

_DB_ARG: str | None = None


def _utf8_stdout() -> None:
    """Windows consoles default to a legacy code page; print UTF-8 regardless."""
    if os.name == "nt":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _db_path() -> str:
    return _DB_ARG or os.environ.get("KB_DB", "work.db")


def _read_db_path() -> str:
    if _DB_ARG:
        return _DB_ARG
    if os.environ.get("KB_DB"):
        return os.environ["KB_DB"]
    return "index.db" if os.path.exists("index.db") else "work.db"


def _out_dir() -> str:
    return os.environ.get("KB_OUT") or os.path.dirname(os.path.abspath(_db_path()))


def cmd_index(a) -> int:
    from .indexer import run
    cfg = load_config(a.config)
    ignore = IgnoreRules.for_config(cfg, a.indexignore)
    store = Store(_db_path())
    rep = run(cfg, ignore, store, repos=a.repo, force=a.force, dry_run=a.dry_run)
    print(json.dumps({"per_repo": rep.per_repo, "embedded": rep.embedded, "cached": rep.cached,
                      "failures": rep.failures, "elapsed_s": round(rep.elapsed)}, ensure_ascii=False, indent=2))
    if not a.dry_run and not a.no_publish and not rep.failures:
        # Publish only when the work database differs from the last published one in anything clients read
        # (every publish is a new version every client pulls). Compare the two databases' state, not "what
        # changed in this run": if a previous run wrote changes but failed to publish (a source failed, or
        # the publish crashed), this run changes nothing and a per-run check would strand those changes.
        if a.publish_always or _needs_publish(store):
            p = store.publish(_out_dir())
            print(f"published {p} ({os.path.getsize(p) / 1e6:.1f} MB)")
        else:
            print("unchanged vs published index.db, not published")
    store.close()
    return 1 if rep.failures else 0


def _needs_publish(store: Store) -> bool:
    import sqlite3
    from .delta import state_sig
    final = os.path.join(_out_dir(), "index.db")
    if not os.path.exists(final):
        return True
    old = sqlite3.connect(f"file:{os.path.abspath(final)}?mode=ro", uri=True)
    try:
        return state_sig(store.db) != state_sig(old)
    finally:
        old.close()


def cmd_rebuild(a) -> int:
    from .indexer import run
    cfg = load_config(a.config)
    ignore = IgnoreRules.for_config(cfg, a.indexignore)
    store = Store(_db_path())
    repos = a.repo or [s.repo for s in cfg.sources]
    for r in repos:
        store.delete_repo(r)
    if a.purge_cache:
        store.purge_cache()
    store.commit()
    rep = run(cfg, ignore, store, repos=repos)
    print(json.dumps({"per_repo": rep.per_repo, "embedded": rep.embedded, "cached": rep.cached,
                      "failures": rep.failures, "elapsed_s": round(rep.elapsed)}, ensure_ascii=False, indent=2))
    if not a.no_publish and not rep.failures:
        p = store.publish(_out_dir())
        print(f"published {p} ({os.path.getsize(p) / 1e6:.1f} MB)")
    store.close()
    return 1 if rep.failures else 0


def cmd_status(a) -> int:
    store = Store(_db_path(), readonly=True)
    s = store.stats()
    print(json.dumps(s, ensure_ascii=False, indent=2))
    return 0


def _searcher():
    from .search import Searcher
    return Searcher(_read_db_path())


def cmd_query(a) -> int:
    s = _searcher()
    res = s.search(a.query, k=a.k, repos=a.repo, domains=a.domain, expand=a.expand)
    if not res["found"]:
        print(f"not found (threshold {res['threshold']}): {res['note']}")
        if res.get("excluded"):
            print("excluded (never indexed): " + ", ".join(res["excluded"]))
        return 0
    print(f"vector_hits={res['vector_hits']} bm25_hits={res['bm25_hits']} {res['elapsed_ms']}ms")
    for i, h in enumerate(res["hits"], 1):
        flag = ("V" if "vec_score" in h else "-") + ("B" if h.get("bm25_hit") else "-")
        vs = f" cos={h['vec_score']:.3f}" if "vec_score" in h else ""
        print(f"{i}. [{flag}] rrf={h['score']:.4f}{vs}  {h['repo']}/{h['path']}  § {h['heading'][:70]}")
        if a.verbose:
            print("     " + h["snippet"][:300])
    if res["repo_rules"]:
        for r, rule in res["repo_rules"].items():
            print(f"  rule[{r}]: {rule}")
    return 0


def cmd_grep(a) -> int:
    res = _searcher().grep(a.pattern, k=a.k, repos=a.repo, domains=a.domain)
    if "error" in res:
        print(res["error"])
        return 1
    for m in res["matches"]:
        print(f"{m['repo']}/{m['path']} § {m['heading'][:50]}\n    {m['line']}")
    print(f"({len(res['matches'])} matches{', truncated' if res['truncated'] else ''})")
    if not res["matches"] and res.get("excluded"):
        print("excluded (never indexed): " + ", ".join(res["excluded"]))
    return 0


def cmd_neighbors(a) -> int:
    print(json.dumps(_searcher().neighbors(a.path, a.repo_one), ensure_ascii=False, indent=2))
    return 0


def cmd_read(a) -> int:
    res = _searcher().read_doc(a.path, a.repo_one)
    if not res["found"]:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 1
    print(f"# {res['repo']}/{res['path']}  (indexed {res['indexed_at']})")
    if res["rule"]:
        print(f"> rule: {res['rule']}")
    print(res["text"])
    return 0


def cmd_ignore(a) -> int:
    from .sources import open_source
    cfg = load_config(a.config)
    ignore = IgnoreRules.for_config(cfg, a.indexignore)
    print(f"patterns ({len(ignore.patterns)}):")
    for p in ignore.patterns:
        print(f"  {p}")
    for src in cfg.sources:
        if a.repo and src.repo not in a.repo:
            continue
        entries = open_source(src).list_files()
        hit = [(e.rel, ignore.match(src.repo, e.rel)) for e in entries]
        ign = [(r, p) for r, p in hit if p]
        print(f"\n{src.repo}: {len(entries)} files, {len(ign)} ignored")
        for r, p in ign:
            print(f"  - {r}    <- {p}")
    return 0


def calibrate(searcher, inside: list[str], outside: list[str], k: int = 5) -> dict:
    """Top-1 vector cosine for queries that should hit (`inside`) and queries that should not (`outside`),
    with the threshold disabled. A usable threshold lies between max(outside) and min(inside)."""
    st = searcher._open()
    try:
        searcher._ensure()

        def top1(q: str) -> dict:
            hits = searcher.vector_leg(st.db, q, k, min_score=-1.0)
            if not hits:
                return {"query": q, "top1": -1.0, "path": None}
            row = st.db.execute("select repo, path from chunks where id=?", (hits[0][0],)).fetchone()
            return {"query": q, "top1": round(hits[0][1], 4), "path": f"{row[0]}/{row[1]}" if row else None}

        a = [top1(q) for q in inside]
        b = [top1(q) for q in outside]
    finally:
        st.close()
    lo = max((x["top1"] for x in b), default=None)
    hi = min((x["top1"] for x in a), default=None)
    out = {"model": searcher.meta.get("model"), "current_threshold": searcher.threshold,
           "inside": a, "outside": b, "inside_min": hi, "outside_max": lo}
    if lo is not None and hi is not None:
        out["gap"] = round(hi - lo, 4)
        out["suggested_threshold"] = round((hi + lo) / 2, 3) if hi > lo else None
        t = searcher.threshold
        out["at_current"] = {"inside_dropped": sum(1 for x in a if x["top1"] < t),
                             "outside_admitted": sum(1 for x in b if x["top1"] >= t)}
    return out


def cmd_calibrate(a) -> int:
    import yaml
    with open(a.queries, encoding="utf-8") as f:
        q = yaml.safe_load(f) or {}
    inside, outside = list(q.get("in") or []), list(q.get("out") or [])
    if not inside or not outside:
        print("the queries file needs two non-empty lists: `in` (should be found) and `out` (should not)")
        return 2
    res = calibrate(_searcher(), inside, outside, k=a.k)
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    for tag, rows in (("in ", res["inside"]), ("out", res["outside"])):
        for r in rows:
            print(f"[{tag}] top1={r['top1']:.3f}  {r['query'][:48]:<48}  {r['path'] or '-'}")
    print(f"\nmodel {res['model']}, current threshold {res['current_threshold']}")
    print(f"in   min top1 {res['inside_min']:.3f}")
    print(f"out  max top1 {res['outside_max']:.3f}")
    if res.get("suggested_threshold") is not None:
        print(f"gap {res['gap']:.3f} -> suggested threshold {res['suggested_threshold']}")
    else:
        print("the two groups overlap: no absolute threshold separates them; add queries or reconsider the model")
    if "at_current" in res:
        c = res["at_current"]
        print(f"at the current threshold: {c['inside_dropped']} in-corpus queries dropped, "
              f"{c['outside_admitted']} out-of-corpus queries admitted")
    return 0


def cmd_publish(a) -> int:
    store = Store(_db_path())
    p = store.publish(_out_dir())
    print(f"published {p} ({os.path.getsize(p) / 1e6:.1f} MB)")
    return 0


def cmd_sig(a) -> int:
    import sqlite3
    from .delta import content_sig, get_version
    db = sqlite3.connect(f"file:{os.path.abspath(a.db)}?mode=ro", uri=True)
    try:
        print(json.dumps({"version": get_version(db), "sig": content_sig(db)}))
    finally:
        db.close()
    return 0


def cmd_apply_delta(a) -> int:
    import sqlite3
    from .delta import apply_delta, content_sig, get_version
    db = sqlite3.connect(a.db)
    try:
        for d in a.deltas:
            apply_delta(db, d)
        print(json.dumps({"version": get_version(db), "sig": content_sig(db)}))
    finally:
        db.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    global _DB_ARG
    _utf8_stdout()
    ap = argparse.ArgumentParser(prog="idx", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="sources.yaml (default $KB_CONFIG or ./sources.yaml)")
    ap.add_argument("--indexignore", default=None, help=".indexignore (default: see kbindex/config.py)")
    ap.add_argument("--db", dest="main_db", default=None, help="database (default: $KB_DB, see above)")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("index"); p.add_argument("--repo", action="append"); p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--no-publish", action="store_true")
    p.add_argument("--publish-always", action="store_true"); p.set_defaults(fn=cmd_index)
    p = sub.add_parser("rebuild"); p.add_argument("--repo", action="append"); p.add_argument("--purge-cache", action="store_true")
    p.add_argument("--no-publish", action="store_true"); p.set_defaults(fn=cmd_rebuild)
    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("query"); p.add_argument("query"); p.add_argument("-k", type=int, default=5)
    p.add_argument("--repo", action="append"); p.add_argument("--domain", action="append")
    p.add_argument("--expand", action="store_true"); p.add_argument("-v", "--verbose", action="store_true"); p.set_defaults(fn=cmd_query)
    p = sub.add_parser("grep"); p.add_argument("pattern"); p.add_argument("-k", type=int, default=20)
    p.add_argument("--repo", action="append"); p.add_argument("--domain", action="append"); p.set_defaults(fn=cmd_grep)
    p = sub.add_parser("neighbors"); p.add_argument("path"); p.add_argument("--repo", dest="repo_one"); p.set_defaults(fn=cmd_neighbors)
    p = sub.add_parser("read"); p.add_argument("path"); p.add_argument("--repo", dest="repo_one"); p.set_defaults(fn=cmd_read)
    p = sub.add_parser("ignore"); p.add_argument("--repo", action="append"); p.set_defaults(fn=cmd_ignore)
    p = sub.add_parser("calibrate"); p.add_argument("--queries", required=True, help="YAML with `in:` and `out:` lists")
    p.add_argument("-k", type=int, default=5); p.add_argument("--json", action="store_true"); p.set_defaults(fn=cmd_calibrate)
    p = sub.add_parser("publish"); p.set_defaults(fn=cmd_publish)
    p = sub.add_parser("sig"); p.add_argument("--db", required=True); p.set_defaults(fn=cmd_sig)
    p = sub.add_parser("apply-delta"); p.add_argument("--db", required=True); p.add_argument("deltas", nargs="+")
    p.set_defaults(fn=cmd_apply_delta)

    a = ap.parse_args(argv)
    _DB_ARG = a.main_db
    if not a.cmd:
        ap.print_help()
        return 2
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
