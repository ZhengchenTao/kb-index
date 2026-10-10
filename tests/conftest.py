from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"


def pytest_configure(config):
    import kbindex
    here = os.path.normcase(str(ROOT))
    assert os.path.normcase(kbindex.__file__).startswith(here), \
        f"tests must import kbindex from this checkout, got {kbindex.__file__}"


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    for k in ("KB_CONFIG", "KB_INDEXIGNORE", "KB_DB", "KB_OUT", "KB_WORKSPACE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("KB_USAGE_LOG", str(tmp_path / "usage" / "embed-usage.jsonl"))


@pytest.fixture
def corpus(tmp_path) -> Path:
    """A private copy of examples/ (tests may edit files)."""
    dst = tmp_path / "examples"
    shutil.copytree(EXAMPLES, dst)
    return dst


def run_cli(*args: str) -> int:
    from kbindex import cli
    return cli.main(list(args))


@pytest.fixture
def built(corpus, tmp_path) -> dict:
    """Index the example corpus with the hash provider; returns paths."""
    out = tmp_path / "out"
    cfg = corpus / "sources.example.yaml"
    assert run_cli("--config", str(cfg), "--db", str(out / "work.db"), "index") == 0
    return {"config": cfg, "corpus": corpus, "out": out, "work": out / "work.db", "index": out / "index.db"}


def meta_of(db_path) -> dict:
    db = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        return dict(db.execute("select key, value from meta").fetchall())
    finally:
        db.close()
