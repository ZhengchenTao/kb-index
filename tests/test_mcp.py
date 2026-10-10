from __future__ import annotations

import asyncio
import json

import pytest

from conftest import run_cli
from kbindex.mcp_server import DEFAULT_PORT, build_server, default_db_path


def _tools(server):
    return {t.name: t for t in asyncio.run(server.list_tools())}


def _call(server, name, args):
    res = asyncio.run(server.call_tool(name, args))
    return json.loads(res.content[0].text)


def test_tools_and_schemas(built):
    server = build_server(str(built["index"]))
    tools = _tools(server)
    assert set(tools) == {"search", "grep", "neighbors", "read_doc"}
    props = {n: set(t.input_schema["properties"]) for n, t in tools.items()}
    assert props == {"search": {"query", "k", "repos", "domains", "expand"},
                     "grep": {"pattern", "k", "repos", "domains"},
                     "neighbors": {"path", "repo"},
                     "read_doc": {"path", "repo"}}
    assert "handbook, notes" in tools["search"].description
    assert "life, personal, team" in tools["search"].description


def test_default_instructions(built):
    server = build_server(str(built["index"]))
    text = server.instructions
    assert "(handbook, notes)" in text and "AGENTS.md" in text
    assert "local_path" in text and "rebuilt every few hours" in text


def test_tool_calls(built):
    server = build_server(str(built["index"]))
    res = _call(server, "search", {"query": "restore the latest dump", "k": 2})
    assert res["found"] and res["hits"][0]["path"] == "ops/backups.md"
    g = _call(server, "grep", {"pattern": "8081", "repos": ["handbook"]})
    assert g["matches"][0]["path"] == "deploy.md"
    nb = _call(server, "neighbors", {"path": "README.md", "repo": "handbook"})
    assert nb["found"] and {e["path"] for e in nb["links_out"]} >= {"deploy.md", "ops/backups.md"}
    rd = _call(server, "read_doc", {"path": "recipes/sourdough.md"})
    assert rd["repo"] == "notes" and rd["domain"] == "life"


def test_overrides_from_config(corpus, tmp_path):
    cfg = corpus / "sources.example.yaml"
    text = cfg.read_text(encoding="utf-8").replace(
        "  agents_files: [AGENTS.md]\n",
        "  agents_files: [AGENTS.md]\n  instructions: Custom instructions.\n"
        "  tool_descriptions:\n    grep: Custom grep text.\n")
    cfg.write_text(text, encoding="utf-8")
    out = tmp_path / "o"
    assert run_cli("--config", str(cfg), "--db", str(out / "work.db"), "index") == 0
    server = build_server(str(out / "index.db"))
    assert server.instructions == "Custom instructions."
    tools = _tools(server)
    assert tools["grep"].description == "Custom grep text."
    assert tools["search"].description.startswith("Hybrid semantic")


def test_db_path_resolution(monkeypatch):
    monkeypatch.delenv("KB_DB", raising=False)
    assert default_db_path() == "index.db"
    monkeypatch.setenv("KB_DB", "/x/env.db")
    assert default_db_path() == "/x/env.db"
    assert default_db_path("/x/arg.db") == "/x/arg.db"
    assert DEFAULT_PORT == 47291


def test_missing_db_fails_early(tmp_path):
    with pytest.raises(RuntimeError, match="index.db not found"):
        build_server(str(tmp_path / "nope.db"))
