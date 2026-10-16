# kb-index

English | [简体中文](README.zh-CN.md)

Hybrid search over several Markdown repositories, packed into one SQLite file and served to AI coding
agents over MCP.

I keep notes, work documentation and a few project wikis in separate repositories, each with its own
`AGENTS.md` rules. Agents kept asking "which repo is this in, and which rules apply?", and kept missing
documents that used different words than the question. kb-index indexes all of them into a single
`index.db` and gives agents four read-only tools. Every result carries the rules of the repository it came
from, so the rules travel with the content instead of depending on which directory the agent was started in.

```
 repositories (git, bare or not / plain directories)
        │  idx index     (incremental; one container run on a schedule)
        ▼
   ┌──────────┐   SQLite: chunks + vectors + FTS5 + link graph + meta (rules, settings)
   │ index.db │   published with a sha256, a manifest and per-file deltas
   └──────────┘
        │  copied to each machine however you like (scp, a share, `ssh host cat`, ...)
        ▼
   kb-index-mcp  (one process per machine, HTTP on loopback, or stdio)
        │
        ▼
   Claude Code / Codex / any MCP client
```

Mental model: Markdown files are the source code, `index.db` is the build artifact, the indexer is the
compiler. Writing documents needs no extra step; if the database breaks, delete it and rebuild.

## What it does

- **Hybrid retrieval.** A vector leg (cosine similarity) and a BM25 leg (SQLite FTS5, trigram tokenizer, so
  CJK text works) fused with reciprocal rank fusion. The similarity threshold only cuts the vector leg:
  identifiers like `8081` or a container name are found by BM25 even when the embedding is useless. When
  both legs are empty the answer is an honest `found: false`.
- **Rules travel with content.** Each source has a one-line `rule`; every tool returns the rules of the
  repositories it hit, plus `agents_chain`: the `AGENTS.md` files (root and every directory level down to
  the hit) the agent should read before acting.
- **Link graph for free.** `[[wiki links]]` and relative Markdown links become edges. Search hits carry
  `links` (who links here, where it links to), and `status: archived` for documents under `archive/`,
  files named like `old` / `deprecated`, or sections titled "deprecated" / "已废弃" (English and Chinese
  word lists).
- **Knows what it does not know.** The `.indexignore` patterns are published with the index; a search that
  finds nothing returns them as `excluded`, so the agent can tell "not in the corpus" from "deliberately not
  indexed, look in the local checkout".
- **Incremental everything.** Git sources are diffed by blob sha (unchanged files are never read),
  directories by mtime + size + hash. Vectors are cached by text hash, so re-chunking or rebuilding never
  pays twice. Publishing writes per-file deltas, so clients usually download a few MB instead of the whole
  file.
- **Pluggable embeddings.** Gemini, any OpenAI-compatible endpoint (OpenAI, Ollama, LM Studio, vLLM, TEI,
  most cloud vendors), or an offline `hash` provider for trying things out without a key. No SDKs, no torch.

## Quick start

Requires Python 3.12+ and git.

```bash
git clone https://github.com/ZhengchenTao/kb-index.git && cd kb-index
pip install -e .

# index the bundled example corpus with the offline hash provider (no API key)
idx --config examples/sources.example.yaml --db /tmp/kb/work.db index
idx --db /tmp/kb/index.db query "how do I roll back a release" -v
idx --db /tmp/kb/index.db grep 8081
idx --db /tmp/kb/index.db neighbors deploy.md
idx --db /tmp/kb/index.db grep quarterly-tax-reminder      # excluded file: no match, `excluded` explains why

# serve it
kb-index-mcp --db /tmp/kb/index.db                          # stdio
kb-index-mcp --db /tmp/kb/index.db --http 127.0.0.1:47291   # shared HTTP server
```

On Windows the `Scripts` directory is often not on PATH; `python -m kbindex.cli` and
`python -m kbindex.mcp_server` work everywhere.

For real use copy `examples/sources.example.yaml` somewhere outside this repository (your configuration
lists private paths and rules; it belongs with your deployment), pick a real embedding provider, and point
`sources` at your repositories.

### Connect an MCP client

Claude Code (`~/.claude.json` or `claude mcp add`):

```json
"kb-index": { "type": "http", "url": "http://127.0.0.1:47291/mcp" }
```

Codex (`~/.codex/config.toml`):

```toml
[mcp_servers.kb-index]
url = "http://127.0.0.1:47291/mcp"
```

Or stdio: `{"command": "kb-index-mcp", "args": ["--db", "/path/to/index.db"]}`.

The server needs the query-side API key in its environment (the variable named by `embedding.api_key_env`,
`GEMINI_API_KEY` by default). Nothing else: rules, thresholds, exclusions and client settings come from the
database.

## Tools

| Tool | Use it for | Returns |
|---|---|---|
| `search(query, k, repos?, domains?, expand?)` | questions in natural language | top-k locations (repo / path / heading / snippet / scores), `links`, `status`, `repo_rules`, `repos`, `agents_chain`; `excluded` when nothing matched |
| `grep(pattern, k, repos?, domains?)` | ports, container names, paths, identifiers (regex, case-insensitive) | matching lines with their documents, same rule / repo / chain fields |
| `neighbors(path, repo?)` | the full link graph of one document | links out / in (each with `status`), semantically similar documents |
| `read_doc(path, repo?)` | reading a document from a repository that is not checked out locally | the document reassembled from chunks (not verbatim), with its rule |

Tools return locations, not whole documents; agents are told to open the real file when they have it.

## Configuration

The indexer reads `sources.yaml` (`--config` > `$KB_CONFIG` > `./sources.yaml`). Clients never read it:
everything they need is written into the `meta` table of `index.db`.

```yaml
embedding:
  provider: gemini            # gemini | openai | hash
  model: gemini-embedding-001
  dim: 1536
  batch: 50                   # texts per embedding request
  api_key_env: GEMINI_API_KEY # name of the variable holding the key; the key itself is never stored
  base_url: ""                # openai provider: e.g. https://api.openai.com/v1, http://localhost:11434/v1
  query_prefix: ""            # asymmetric models: instruction prepended to queries
  document_prefix: ""         # ... and to documents

search:
  threshold: 0.69             # cosine floor of the vector leg; model specific, measure with `idx calibrate`
  rrf_k: 60

chunk:
  min: 30                     # drop shorter chunks
  max: 1200                   # split longer ones by paragraph, then by line
  preamble_label: "(preamble)"

ignore_file: .indexignore     # relative to this file (default: .indexignore next to it)
extended_meta: true           # see "Changing a live deployment"

client:                       # shipped to MCP servers through index.db; all optional
  workspace: { nt: 'C:\src', posix: '~/src' }   # where checkouts live -> local_path / local_exists / clone_hint
  agents_files: [AGENTS.md]   # instruction file names used for agents_chain
  freshness_note: "The index is rebuilt every two hours."
  instructions: "..."         # replaces the built-in MCP server instructions
  tool_descriptions: { search: "..." }   # replaces built-in tool descriptions

sources:
  - repo: handbook            # name shown to agents; also the directory name under client.workspace
    type: git                 # git (bare or not; reads `git ls-tree` / `git show`, no working tree needed) | dir
    path: /srv/git/team/handbook.git   # absolute, or relative to this file
    ref: HEAD
    include: ["*.md", "sql/**/*.sql"]  # glob patterns; non-Markdown files become one chunk (split if long)
    domain: team              # free-form label for filtering
    domains: { "ops/": ops }  # path prefix -> domain, longest prefix wins
    rule: "Published to the team site: no credentials, use placeholders."
    url: https://git.example.com/team/handbook.git   # returned to clients without a local checkout
    clone_hint: ""            # free text instead of `git clone <url>` (for sources that are not git repos)
```

Environment: `KB_DB` (work database, default `./work.db`), `KB_OUT` (publish directory, default the work
database's directory), `KB_INDEXIGNORE`, `KB_WORKSPACE` (overrides `client.workspace` on one machine),
`KB_USAGE_LOG` (see below).

### `.indexignore`

gitignore-style patterns matched against `repo/path/inside/repo`: `**` crosses directories, `*` does not
cross `/`, a trailing `/` means a directory, a pattern without `/` matches a file name at any depth.
`idx ignore` lists what each pattern hits. **Anything indexed can be read by every client of the database.**
Keep secrets, credentials and personal data out with rules here, and remember that file names leak too.

## Commands

```
idx index [--repo R] [--force] [--dry-run] [--no-publish] [--publish-always]
idx rebuild [--repo R] [--purge-cache]   # drop chunks and re-chunk; cached vectors are reused unless purged
idx status | ignore | publish
idx query "..." [-k 5] [--repo R] [--domain D] [--expand] [-v]   # prints both legs' scores
idx grep PATTERN | neighbors PATH | read PATH
idx calibrate --queries q.yaml          # measure the threshold, see below
idx sig --db FILE                        # version + content signature (for pull scripts)
idx apply-delta --db FILE DELTA.db.gz ...
```

`idx index` publishes only when something clients read actually changed (chunks, vectors, rules, URLs,
thresholds, exclusions, client settings); otherwise it prints `unchanged vs published index.db, not
published`. Every publish is a new version every client pulls, so a no-op run must not create one.

Removing a source from `sources.yaml` does not delete its data: a typo or a source commented out for a moment
must not wipe anything. Until it is cleaned up its chunks stay searchable and its `rule` / `url` stay in every
answer, so each `idx index` prints `stale source: R is in the database but not in the config ... Remove it
with: idx rebuild --repo R`. That `rebuild` drops the repo's chunks, files and per-repo meta (`head:` /
`rule:` / `url:` / `hint:`) and publishes. `rebuild --repo` with a name that is neither configured nor in the
database is refused.

### Calibrating the threshold

The cosine threshold depends entirely on the model (and its dimension). Write a few questions your corpus
answers and a few it does not, including generic technical ones (that is where false positives come from):

```yaml
in:  [how do I roll back a release, how long are database dumps kept]
out: [how to bake croissants, what is a python decorator]
```

`idx calibrate --queries q.yaml` prints the top-1 cosine of each with the threshold disabled and suggests the
midpoint between the highest `out` and the lowest `in`. With `gemini-embedding-001` at 1536 dimensions, 20
real queries gave in-corpus >= 0.727 and out-of-corpus <= 0.649, hence 0.69. Re-run after changing models.

## Distributing the database

The indexer writes, into `$KB_OUT`:

```
index.db              the published database (vector cache stripped)
index.db.sha256
deltas/<N>.db.gz      changes from version N-1 to N, one SQLite file per publish (the last 40 are kept)
index.manifest.json   {"version": N, "sha256": ..., "sig": ..., "size": ...,
                       "deltas": {"N": {"from": N-1, "from_sig": ..., "sig": ..., "file": "deltas/N.db.gz",
                                         "sha256": ..., "size": ..., "changed": ..., "removed": ..., "chunks": ...}}}
```

How you move files is up to you (for example `ssh host cat`, which also works when the server cannot open
connections to the clients). A pull script should:

1. Read `index.manifest.json`. If the local copy's `idx sig --db local.db` equals `sig`, stop.
2. Otherwise walk `deltas` from the local version + 1 up to `version`; every entry's `from_sig` must equal
   the signature reached so far. If the chain is complete, download those files, check each `sha256`, run
   `idx apply-delta --db copy.db <files...>` on a copy, and replace the local file only if the printed
   signature equals `sig`.
3. Any break (copy too old, diverged, a hash mismatch): download `index.db` and check `index.db.sha256`.

The signature is not a file hash (a delta-patched file differs byte-wise from the published one): it is a
sha256 over a per-file digest of everything stored for that file (source hash, chunk ids, domains,
headings, text hashes, structural edges, vector bytes, link edges) plus every meta row. The algorithm is a
wire format between publisher and clients: changing it forces every client into a full download.

Replace the database atomically (write a temp file, then rename). The MCP server opens a connection per
call and reloads its in-memory vectors when the file's mtime changes, so no restart is needed. On Windows a
rename fails while a call holds the file open; retry for a few seconds.

## Running it

**Indexer**: `Dockerfile` builds an image whose entrypoint is `idx`; mount the configuration at `/config`,
repositories read-only, and a data volume at `/data` (see `examples/docker-compose.example.yml`). Run
`docker compose run --rm kb-index index` from cron or a scheduler. Mount git servers' bare repositories
read-only and never run git write commands in them: objects created by another user can leave the git
server unable to write.

**MCP server**: one HTTP process per machine (`--http`, loopback only) shared by every agent session works
better than one stdio process per session:

- `OPENBLAS_NUM_THREADS=1` is set before numpy loads. OpenBLAS otherwise pre-commits a buffer per thread;
  on a 28-core machine that was ~900 MB of committed memory per process, and a dozen sessions exhausted the
  commit limit. Single-threaded it is ~175 MB, and the dot product is still milliseconds next to a
  ~900 ms embedding request.
- Stateless streamable HTTP with JSON responses: a restarted server is invisible to connected clients.
- The port is a single-instance lock (`SO_EXCLUSIVEADDRUSE` on Windows); a duplicate exits with code 2.
- On a loopback address DNS-rebinding protection is on. Any local process can call the server, and the
  server exposes the whole corpus: never bind it to a public interface.
- `--log FILE` records every failed call and every miss, so "the agent said kb found nothing" can be traced
  (quota exhausted, network down, key missing).
- Supervise it with whatever your OS offers (systemd, launchd, a scheduled task that restarts it).

**Usage accounting**: Gemini embedding responses carry no token counts and API keys cannot query usage, so
each request appends a line to `embed-usage.jsonl` (`$KB_USAGE_LOG`, else next to `$KB_DB`, else nothing):
`{"ts", "date", "kind": "index"|"query", "model", "tokens", "items", "chars", "exact"}`. The file is trimmed
by month, never by line count, so a monthly total computed from it cannot shrink.

## Changing a live deployment

What changes the index, and what it costs:

| Change | Effect |
|---|---|
| `rule`, `url`, `clone_hint`, `client.*`, `.indexignore`, `threshold` | meta only: one publish of a small delta |
| `domain` / `domains` | chunks rewritten on the next `idx rebuild --repo R`, vectors reused from the cache (free) |
| removing a source | nothing until `idx rebuild --repo R` (each `idx index` reminds you); then a delta without that repo |
| `chunk.min` / `max` / `preamble_label`, chunker code | new chunk texts: re-embedding (paid) and large deltas |
| `model`, `dim`, `provider`, `document_prefix` | everything re-embedded: `idx rebuild --purge-cache`, then recalibrate |

`extended_meta: false` writes only the meta keys of earlier versions (`model`, `dim`, `threshold`,
`rrf_k`, `ignore`, `rule:*`, `url:*`). It exists for upgrading a running deployment: deploy the new code
with it off, check that the indexer reports `unchanged ..., not published` (proof that nothing clients see
changed), then turn it on.

## Design notes

- **SQLite + numpy instead of a vector database.** One file, no daemon, trivially copied. A brute-force dot
  product over 16k x 1536 float32 takes a few milliseconds; sqlite-vec or an ANN index can wait until the
  corpus is ten times larger.
- **Chunk along headings.** Documentation is already structured; headings are natural semantic borders and
  make readable hit labels (`file > H1 > H2`). Long sections are split by paragraph, then hard-cut at
  line boundaries (tables and code blocks have no blank lines); tiny chunks are dropped.
- **Two phases per run.** All missing vectors are embedded and committed to the cache batch by batch
  before any chunk is written, so a crash halfway costs nothing on the next run.
- **Relations in the hit, not in a separate tool.** In practice agents almost never call a "neighbors" tool
  during real work; once they have a path they open the file. Link and archive signals are only used when
  they arrive with the hit.
- **Asymmetric embeddings.** Gemini uses different task types for documents and queries; query-to-document
  and document-to-document cosines are on different scales, which is why the threshold must be measured with
  real queries.

## Development

```bash
pip install -e ".[test]"
python -m pytest
```

`tests/golden_examples.json` freezes the chunker's output on the example corpus. Chunking is a contract
with every existing deployment (a change re-embeds their corpus); if a change is intended, regenerate it
with `PYTHONPATH=. python tests/test_chunker.py --update` and say so in the commit message.

## Security

- `index.db` contains the full text of everything indexed. Treat it with the sensitivity of the most
  sensitive document in it.
- The MCP server is read-only and should listen on loopback only.
- API keys are read from the environment only; `index.db` stores the variable name, never the value.

## License

MIT
