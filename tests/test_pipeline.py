"""End to end on the example corpus with the offline hash provider: index -> publish -> search / grep /
neighbors / read_doc -> edit -> delta -> apply_delta -> signatures match."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import time

import numpy as np
import pytest

from conftest import meta_of, run_cli
from kbindex import delta as D
from kbindex.config import IgnoreRules, load_config
from kbindex.indexer import run
from kbindex.search import Searcher
from kbindex.store import Store


def test_index_and_publish(built):
    out = built["out"]
    assert (out / "index.db").exists() and (out / "index.db.sha256").exists()
    mf = json.loads((out / "index.manifest.json").read_text(encoding="utf-8"))
    assert mf["version"] == 1 and mf["deltas"] == {}
    db = sqlite3.connect(out / "index.db")
    files = {(r[0], r[1]) for r in db.execute("select repo, path from files")}
    assert ("handbook", "sql/reports/daily.sql") in files
    assert not any(p.startswith("private/") for r, p in files if r == "notes")          # excluded
    assert db.execute("select count(*) from vec_cache").fetchone()[0] == 0              # cache stripped
    assert db.execute("select count(*) from chunks where vector is null").fetchone()[0] == 0
    domains = dict(db.execute("select path, domain from chunks where repo='notes'").fetchall())
    assert domains["recipes/sourdough.md"] == "life" and domains["home-network.md"] == "personal"
    m = meta_of(out / "index.db")
    assert m["provider"] == "hash" and m["preamble_label"] == "(preamble)"
    assert json.loads(m["client:workspace"])["posix"] == "~/src"
    assert m["hint:notes"].startswith("notes is a synced folder")
    assert "notes/private/" in json.loads(m["ignore"])
    assert m["version"] == "1"


def test_second_run_does_not_publish(built, capsys):
    assert run_cli("--config", str(built["config"]), "--db", str(built["work"]), "index") == 0
    assert "unchanged vs published index.db, not published" in capsys.readouterr().out


def test_search_grep_neighbors_read(built, monkeypatch, tmp_path):
    ws = tmp_path / "ws"
    (ws / "handbook").mkdir(parents=True)
    monkeypatch.setenv("KB_WORKSPACE", str(ws))
    s = Searcher(built["index"])

    res = s.search("restore the latest dump", k=3)
    assert res["found"] and res["hits"][0]["path"] == "ops/backups.md"
    hit = res["hits"][0]
    assert hit["doc_status"] == "draft"
    assert res["agents_chain"]["handbook"] == ["AGENTS.md", "ops/AGENTS.md"]
    assert res["repo_rules"]["handbook"].startswith("Published to the team site")
    info = res["repos"]["handbook"]
    assert info["local_exists"] is True and info["local_path"] == os.path.join(str(ws), "handbook")

    g = s.grep("listens on port 8081")
    assert g["matches"][0]["path"] == "deploy.md" and "repo_rules" in g and "agents_chain" in g
    none = s.grep("quarterly-tax-reminder")
    assert none["matches"] == [] and "notes/private/" in none["excluded"]
    assert "never indexed" in none["note"] and "rebuilt every few hours" in none["note"]
    assert "**/.git/" not in none["excluded"]                                  # generic noise filtered

    nb = s.neighbors("deploy.md")
    ins = {e["path"]: e for e in nb["links_in"]}
    assert ins["archive/legacy-deploy.md"]["status"] == "archived"
    assert "directory archive/" in ins["archive/legacy-deploy.md"]["status_reason"]
    assert {e["path"] for e in nb["links_out"]} == {"ops/backups.md"}
    assert nb["agents_chain"] == ["AGENTS.md"]

    notes = s.repo_info(["notes"])["notes"]
    assert notes["local_exists"] is False and notes["clone_hint"].startswith("notes is a synced folder")

    rd = s.read_doc("deploy.md")
    assert rd["found"] and "(preamble)" not in rd["text"] and "## Rollback" in rd["text"]
    assert rd["text"].startswith("The deployment pipeline builds")


def test_section_status(built):
    s = Searcher(built["index"])
    st = s._open()
    try:
        s._ensure()
        a = s.annotate(st.db, "handbook", "deploy.md", "deploy > Deployment > Manual rollout over SSH (deprecated)")
        b = s.annotate(st.db, "handbook", "deploy.md", "deploy > Deployment > Rollback")
    finally:
        st.close()
    assert a["status"] == "archived" and "section heading" in a["status_reason"]
    assert "status" not in b


def test_no_workspace_means_no_local_paths(built):
    s = Searcher(built["index"])
    s.meta = {k: v for k, v in s.meta.items() if k != "client:workspace"}
    info = s.repo_info(["handbook"])["handbook"]
    assert set(info) == {"rule"}


def test_not_found(built):
    s = Searcher(built["index"])
    res = s.search("zz", k=3)                       # under 3 characters: no BM25 term; hash vectors below threshold
    assert res["found"] is False and "notes/private/" in res["excluded"]


def test_delta_roundtrip(built):
    out, corpus = built["out"], built["corpus"]
    client = out / "client.db"
    shutil.copy(out / "index.db", client)
    before = D.content_sig(sqlite3.connect(client))

    p = corpus / "docs" / "handbook" / "deploy.md"
    time.sleep(0.01)
    p.write_text(p.read_text(encoding="utf-8") + "\n## Canary\n\nTen percent of traffic goes to the new image first.\n",
                 encoding="utf-8")
    (corpus / "docs" / "notes" / "home-network.md").unlink()
    assert run_cli("--config", str(built["config"]), "--db", str(built["work"]), "index") == 0

    mf = json.loads((out / "index.manifest.json").read_text(encoding="utf-8"))
    assert mf["version"] == 2
    d = mf["deltas"]["2"]
    assert d["from"] == 1 and d["from_sig"] == before and d["changed"] == 1 and d["removed"] == 1
    assert (out / d["file"]).exists()

    db = sqlite3.connect(client)
    D.apply_delta(db, out / d["file"])
    assert D.content_sig(db) == mf["sig"] == D.content_sig(sqlite3.connect(out / "index.db"))
    assert D.get_version(db) == 2

    # the CLI pieces a pull script uses
    assert run_cli("sig", "--db", str(out / "index.db")) == 0


def test_extended_meta_switch(built):
    cfg = load_config(built["config"])
    cfg.extended_meta = False
    store = Store(built["work"])
    run(cfg, IgnoreRules.for_config(cfg), store)
    keys = {r[0] for r in store.db.execute("select key from meta")}
    store.close()
    assert not any(k.startswith(("client:", "hint:")) for k in keys)
    assert not {"provider", "api_key_env", "preamble_label"} & keys
    assert {"model", "dim", "threshold", "rrf_k", "ignore", "rule:handbook", "url:handbook"} <= keys


def test_vec_cache_width_check(tmp_path):
    st = Store(tmp_path / "w.db")
    st.put_cache([("h1", np.ones(4, dtype=np.float32).tobytes()), ("h2", np.ones(8, dtype=np.float32).tobytes())], "m")
    assert set(st.cached_vectors(["h1", "h2"], "m")) == {"h1", "h2"}
    assert set(st.cached_vectors(["h1", "h2"], "m", dim=8)) == {"h2"}
    st.close()


def test_git_source(tmp_path):
    repo = tmp_path / "wiki"
    repo.mkdir()
    (repo / "page.md").write_text("# Page\n\nA page stored in a git repository, long enough to index.\n", encoding="utf-8")
    (repo / "skip.txt").write_text("not included", encoding="utf-8")
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com")
    for cmd in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "init"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True, env=env)
    cfg_file = tmp_path / "s.yaml"
    cfg_file.write_text("embedding: {provider: hash, model: hash, dim: 32}\n"
                        "sources:\n  - {repo: wiki, type: git, path: wiki, url: 'https://git.example.com/wiki.git'}\n",
                        encoding="utf-8")
    assert run_cli("--config", str(cfg_file), "--db", str(tmp_path / "o" / "work.db"), "index") == 0
    m = meta_of(tmp_path / "o" / "index.db")
    assert m["url:wiki"] == "https://git.example.com/wiki.git" and len(m["head:wiki"]) == 40
    db = sqlite3.connect(tmp_path / "o" / "index.db")
    assert [r[0] for r in db.execute("select path from files")] == ["page.md"]


def test_calibrate(built, tmp_path, capsys):
    q = tmp_path / "q.yaml"
    q.write_text("in: [restore the latest dump]\nout: [croissant recipe]\n", encoding="utf-8")
    assert run_cli("--db", str(built["index"]), "calibrate", "--queries", str(q), "--json") == 0
    res = json.loads(capsys.readouterr().out)
    assert res["inside"][0]["path"] == "handbook/ops/backups.md"
    assert {"inside_min", "outside_max", "gap", "at_current"} <= set(res)
