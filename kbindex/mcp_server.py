"""kb-index-mcp: read-only MCP server with four tools: search / grep / neighbors / read_doc.

Two ways to run it:
  python -m kbindex.mcp_server --db index.db                       stdio, one process per client session
  python -m kbindex.mcp_server --db index.db --http [HOST:]PORT     streamable HTTP, one shared process per machine

Database: --db > $KB_DB > ./index.db. The API key for query embeddings is read from the environment variable
named in the database (GEMINI_API_KEY for databases that do not say).
There is no LLM inside: query -> one embedding HTTP call -> dot product -> one FTS5 query -> JSON.
Tools return locations (path + snippet + score), not whole documents.

Server instructions and tool descriptions come from the database (`client:instructions`,
`client:tool_descriptions`) when the indexer published them, otherwise from the English defaults below.
They are read at start-up; a new database with different text takes effect on restart.
"""
from __future__ import annotations

import os

# Must run before numpy is imported (via .search). numpy's bundled OpenBLAS pre-commits a buffer per thread
# (up to 24) as soon as it loads; the memory is never touched but counts against the commit limit. Measured on
# a 28-core machine: ~900 MB private commit per process with the default, ~175 MB single-threaded. A 13k x 1536
# dot product goes from 1.2 ms to 2.4 ms, invisible next to the ~900 ms embedding round trip.
# setdefault: an explicit value from the environment wins.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import logging
import socket
import sys

from mcp.server.mcpserver import MCPServer

from .search import Searcher

DEFAULT_PORT = 47291

DEFAULT_INSTRUCTIONS = """Cross-repository knowledge base search{repo_list}.
Usage: call search to locate, then open the original file with your own file tools; use read_doc only when the repo is not checked out on this machine.
Use grep, not search, for exact identifiers (port numbers, container names, function names, paths).
neighbors shows what a document links to and what links to it (relations written by the authors: more reliable than semantic similarity, and they tell current documents from deprecated ones).
repo_rules / rule in every result is that repo's one-line rule (for example "public site: no secrets"). Check it before quoting content. It is a one-line rule, not the repo's instructions; you need both.
agents_chain lists, per repo, the instruction files ({agents_files}) that apply to the hit paths, root first. Read them before writing code, changing configuration or drawing conclusions: local_exists=true -> read <local_path>/<entry>; otherwise read_doc(entry, repo=...).
repos / repo_info give each repo's canonical clone_url{workspace_note}.
found=false means this search found no match, not that the corpus contains nothing. Check query terms, filters and `excluded`; consult local files when needed.{freshness}
Every search / grep hit carries relation annotations: links.in / links.in_top = who links here (documents linked from many overviews are the main ones), links.out_top = where it links to.
status="archived" + status_reason marks an archived / deprecated / superseded document or section: do not cite it as current, follow links.in_top to the current document. doc_status is the frontmatter status (draft / implemented ...).
When asked "is this current or deprecated / who else mentions it", bring status and links into the answer instead of judging by content alone."""

DEFAULT_TOOL_DOCS = {
    "search": (
        "Hybrid semantic + BM25 search. Returns the top-k locations (repo / path / heading / snippet / score).\n"
        "repos: optional filter{repos}.\n"
        "domains: optional filter{domains}.\n"
        "expand=true also returns the chunks before and after each hit (more context, more tokens).\n"
        "Every hit carries links (who links here / where it links to) and status "
        "(archived = deprecated / archived / old version; do not cite as current)."
    ),
    "grep": (
        "Case-insensitive regex match. Use it for ports, container names, paths and identifiers; returns each "
        "matching line and its document.\n"
        "Results carry repo_rules / repos for the repos that matched.\n"
        "**Before quoting a grepped line, check repo_rules and read that repo's instruction files** - a rule is "
        "one line, not a summary of the instructions; you need both."
    ),
    "neighbors": (
        "The full relation graph of one document: documents it links to and documents linking to it "
        "(author-made edges, each with status), plus semantically similar documents (auxiliary).\n"
        "path is the repo-relative path from a search hit. search / grep already include a top-3 links summary; "
        "use this only when you need the full picture."
    ),
    "read_doc": (
        "Read-only access to a document reassembled from the index, for repos that are not checked out locally "
        "(not the verbatim file).\n"
        "Prefer reading the local file when you have the repo. The result carries the repo's rule."
    ),
}


def default_db_path(arg: str | None = None) -> str:
    """--db > $KB_DB > ./index.db. Never written back to the environment: $KB_DB also decides where usage is
    recorded (see usage.py), and passing --db must not change that."""
    return arg or os.environ.get("KB_DB") or "index.db"


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


_log = logging.getLogger("kb-index")


def _call(tool: str, arg: str, fn) -> str:
    """Every failure and every miss leaves a line in the --log file. Without it a failing query-side embedding
    (quota exhausted, network down, key missing) only shows up as one tool error inside one client session and
    leaves no trace on the machine. The exception is still raised to the client."""
    try:
        res = fn()
    except Exception as e:
        _log.error("%s failed  %r  %s: %s", tool, arg[:80], type(e).__name__, str(e)[:300])
        raise
    if isinstance(res, dict) and (res.get("found") is False or "error" in res
                                  or (tool == "grep" and not res.get("matches"))):
        _log.warning("%s miss  %r", tool, arg[:80])
    return _j(res)


def _meta_json(meta: dict, key: str):
    raw = meta.get(key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def server_texts(searcher: Searcher) -> tuple[str, dict[str, str]]:
    """(instructions, {tool: description}) for this database."""
    meta = searcher.meta
    cat = searcher.catalog()
    instructions = meta.get("client:instructions") or DEFAULT_INSTRUCTIONS.format(
        repo_list=f" ({', '.join(cat['repos'])})" if cat["repos"] else "",
        agents_files=", ".join(searcher.agents_files),
        workspace_note=("; local_path / local_exists say where it belongs on this machine, and clone_hint how to get "
                        "it: clone missing repos into local_path, not elsewhere" if _has_workspace(searcher) else ""),
        freshness=(" " + meta["client:freshness_note"]) if meta.get("client:freshness_note") else "",
    )
    docs = {
        "search": DEFAULT_TOOL_DOCS["search"].format(
            repos=f": {', '.join(cat['repos'])}" if cat["repos"] else "",
            domains=f": {', '.join(cat['domains'])}" if cat["domains"] else ""),
        "grep": DEFAULT_TOOL_DOCS["grep"],
        "neighbors": DEFAULT_TOOL_DOCS["neighbors"],
        "read_doc": DEFAULT_TOOL_DOCS["read_doc"],
    }
    override = _meta_json(meta, "client:tool_descriptions")
    if isinstance(override, dict):
        docs.update({k: v for k, v in override.items() if k in docs and isinstance(v, str) and v})
    return instructions, docs


def _has_workspace(searcher: Searcher) -> bool:
    from .search import workspace_root
    return workspace_root(searcher.meta) is not None


def build_server(db: str | Searcher) -> MCPServer:
    """Create the MCP server for one database. Raises if the database cannot be opened."""
    s = db if isinstance(db, Searcher) else Searcher(db)
    instructions, docs = server_texts(s)
    server = MCPServer("kb-index", instructions=instructions)

    @server.tool(description=docs["search"])
    def search(query: str, k: int = 5, repos: list[str] | None = None, domains: list[str] | None = None,
               expand: bool = False) -> str:
        return _call("search", query, lambda: s.search(query, k=k, repos=repos, domains=domains, expand=expand))

    @server.tool(description=docs["grep"])
    def grep(pattern: str, k: int = 20, repos: list[str] | None = None, domains: list[str] | None = None) -> str:
        return _call("grep", pattern, lambda: s.grep(pattern, k=k, repos=repos, domains=domains))

    @server.tool(description=docs["neighbors"])
    def neighbors(path: str, repo: str | None = None) -> str:
        return _call("neighbors", path, lambda: s.neighbors(path, repo))

    @server.tool(description=docs["read_doc"])
    def read_doc(path: str, repo: str | None = None) -> str:
        return _call("read_doc", path, lambda: s.read_doc(path, repo))

    server.searcher = s   # type: ignore[attr-defined]
    return server


def _serve_http(server: MCPServer, host: str, port: int, db_path: str) -> None:
    """One shared process per machine.

    - stateless_http: no session ids. A restart (code update, crash + supervisor) is invisible to connected
      clients; the next call simply reaches the new process. With sessions, the new process would reject old
      session ids and every client would have to reconnect.
    - json_response: all four tools are request/response, no SSE stream needed.
    - The socket is created here with SO_EXCLUSIVEADDRUSE: on Windows a second instance could otherwise bind the
      same port and the two would race for requests. With it the duplicate fails to bind and exits, so the port
      doubles as a single-instance lock.
    - On a loopback host the mcp library enables DNS rebinding protection (only 127.0.0.1 / localhost Host
      headers). Any local process can call the server; it exposes the whole corpus, so bind loopback only.
    """
    import anyio
    import uvicorn

    fam = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(fam, socket.SOCK_STREAM)
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        sock.bind((host, port))
    except OSError as e:
        print(f"kb-index-mcp: cannot bind {host}:{port} ({e}); another instance already running?", file=sys.stderr)
        sys.exit(2)
    sock.listen(128)
    sock.set_inheritable(True)

    app = server.streamable_http_app(stateless_http=True, json_response=True, host=host)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    print(f"kb-index-mcp: serving http://{host}:{port}/mcp  (pid {os.getpid()}, db {db_path})",
          file=sys.stderr, flush=True)
    anyio.run(uvicorn.Server(config).serve, [sock])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="kb-index-mcp")
    ap.add_argument("--db", metavar="FILE", help="index.db to serve (default $KB_DB or ./index.db)")
    ap.add_argument("--http", metavar="[HOST:]PORT", nargs="?", const=str(DEFAULT_PORT),
                    help=f"streamable HTTP mode (default 127.0.0.1:{DEFAULT_PORT}); stdio when omitted")
    ap.add_argument("--log", metavar="FILE", help="append stdout / stderr to this file (for runs without a console)")
    args = ap.parse_args(argv)

    if args.log:
        # The server creates its own log directory instead of trusting whoever launched it: a launcher running
        # inside a packaged desktop app on Windows may have its writes under %LOCALAPPDATA% redirected into the
        # app's private store, invisible to the scheduled task that runs the server.
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
        f = open(args.log, "a", encoding="utf-8", buffering=1)
        sys.stderr = f
        if args.http is not None:   # in stdio mode stdout is the protocol channel
            sys.stdout = f
        # Logging handlers may already be bound to the original stderr (None under pythonw), which turns every
        # record into "--- Logging error ---". Rebind to the file; WARNING and up (INFO logs every call).
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                            stream=f, force=True)

    db_path = default_db_path(args.db)
    try:
        server = build_server(db_path)   # load at start-up so a missing file fails early
    except Exception as e:
        print(f"kb-index-mcp: {e}", file=sys.stderr)
        sys.exit(1)

    if args.http is None:
        server.run("stdio")
        return
    host, _, port = args.http.rpartition(":")
    _serve_http(server, host.strip("[]") or "127.0.0.1", int(port), db_path)


if __name__ == "__main__":
    main()
