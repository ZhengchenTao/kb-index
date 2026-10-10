from __future__ import annotations

import os
from pathlib import Path

import pytest

from kbindex.config import DEFAULT_PREAMBLE, IgnoreRules, Source, _glob_to_re, load_config, matches_any


@pytest.mark.parametrize("pat,path,hit", [
    ("notes/secret/", "notes/secret/a.md", True),
    ("notes/secret/", "notes/secret/x/y.md", True),
    ("notes/secret/", "notes/secrets/a.md", False),
    ("repo/*.md", "repo/a.md", True),
    ("repo/*.md", "repo/sub/a.md", False),
    ("repo/**/*.md", "repo/a.md", True),
    ("repo/**/*.md", "repo/sub/deep/a.md", True),
    ("**/node_modules/", "r/x/node_modules/p/index.md", True),
    ("todo.md", "r/deep/dir/todo.md", True),          # no slash: any depth
    ("todo.md", "r/deep/dir/mytodo.md", False),
    ("repo/a?.md", "repo/ab.md", True),
    ("repo/a?.md", "repo/a/b.md", False),
    ("handbook/api-tests/**/.env*", "handbook/api-tests/x/.env.local", True),
])
def test_glob(pat, path, hit):
    assert bool(_glob_to_re(pat).search(path)) is hit


def test_include_globs():
    assert matches_any("a/b.md", ["*.md"])
    assert matches_any("sql/x/y.sql", ["*.md", "sql/**/*.sql"])
    assert not matches_any("a/b.txt", ["*.md"])
    assert matches_any("anything", [])


def test_ignore_rules_skip_comments_and_blanks():
    r = IgnoreRules(["# comment", "", "notes/private/", "**/.git/"])
    assert r.patterns == ["notes/private/", "**/.git/"]
    assert r.match("notes", "private/todo.md") == "notes/private/"
    assert r.match("notes", "public.md") is None


def test_domain_of_longest_prefix():
    s = Source(repo="r", type="dir", path=".", domain="d0", domains={"a/": "d1", "a/b/": "d2"})
    assert s.domain_of("x.md") == "d0"
    assert s.domain_of("a/x.md") == "d1"
    assert s.domain_of("a/b/x.md") == "d2"


def test_load_config_paths_and_defaults(tmp_path):
    (tmp_path / "cfg").mkdir()
    cfg_file = tmp_path / "cfg" / "sources.yaml"
    cfg_file.write_text(
        "sources:\n"
        "  - {repo: rel, type: dir, path: docs}\n"
        "  - {repo: posix, type: git, path: /srv/git/x.git}\n"
        "  - {repo: home, type: dir, path: ~/notes}\n", encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.source("rel").path == str((tmp_path / "cfg" / "docs").resolve())
    assert cfg.source("posix").path == "/srv/git/x.git"            # POSIX-absolute stays, also on Windows
    assert cfg.source("home").path == os.path.expanduser("~/notes")
    assert cfg.preamble_label == DEFAULT_PREAMBLE == "(preamble)"
    assert cfg.provider == "gemini" and cfg.extended_meta is True
    assert cfg.cache_model == cfg.model
    with pytest.raises(KeyError):
        cfg.source("missing")


def test_cache_model_includes_document_prefix(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("embedding: {provider: openai, model: m, document_prefix: 'passage: '}\nsources: []\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.cache_model.startswith("m#doc:") and cfg.cache_model != "m"


def test_ignore_file_lookup(tmp_path, monkeypatch):
    d = tmp_path / "c"
    d.mkdir()
    (d / ".indexignore").write_text("a/default/\n", encoding="utf-8")
    (d / "custom.ignore").write_text("a/custom/\n", encoding="utf-8")
    (tmp_path / "env.ignore").write_text("a/env/\n", encoding="utf-8")
    (d / "s.yaml").write_text("sources: []\n", encoding="utf-8")
    cfg = load_config(d / "s.yaml")
    assert IgnoreRules.for_config(cfg).patterns == ["a/default/"]
    cfg.ignore_file = "custom.ignore"
    assert IgnoreRules.for_config(cfg).patterns == ["a/custom/"]
    monkeypatch.setenv("KB_INDEXIGNORE", str(tmp_path / "env.ignore"))
    assert IgnoreRules.for_config(cfg).patterns == ["a/env/"]
    assert IgnoreRules.for_config(cfg, d / ".indexignore").patterns == ["a/default/"]   # explicit path wins


def test_config_env_lookup(tmp_path, monkeypatch):
    p = tmp_path / "x.yaml"
    p.write_text("sources: []\nembedding: {provider: hash, dim: 8}\n", encoding="utf-8")
    monkeypatch.setenv("KB_CONFIG", str(p))
    cfg = load_config()
    assert cfg.provider == "hash" and cfg.dim == 8 and Path(cfg.path) == p
