"""Chunking: Markdown is split along its heading hierarchy; other text files (.sql, .bru, ...) become one chunk,
split by paragraph / line when too long.

Two pitfalls handled here:
1. Large tables and long code blocks contain no blank lines, so splitting on `\\n\\n` cannot cut them
   -> fall back to hard cuts at line boundaries.
2. A heading directly followed by another heading yields a tiny junk chunk -> minimum length filter.

Everything in this module shapes chunk text and therefore chunk hashes: changing it re-embeds the corpus.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field

from .config import DEFAULT_PREAMBLE

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)")


@dataclass
class Chunk:
    heading: str            # "file name > H1 > H2"
    text: str
    depth: int = 0          # heading level; 0 for the preamble
    ord: int = 0            # position within the file
    links: list[str] = field(default_factory=list)   # raw link targets (unresolved)

    @property
    def embed_text(self) -> str:
        return (self.heading + "\n" + self.text)[:6000]

    @property
    def hash(self) -> str:
        return hashlib.sha1(self.embed_text.encode("utf-8")).hexdigest()


def _split_long(text: str, max_len: int) -> list[str]:
    """Split on blank lines; cut paragraphs that are still too long at line boundaries; then pack small
    paragraphs together up to max_len."""
    if len(text) <= max_len:
        return [text]
    out: list[str] = []
    acc = ""
    for p in text.split("\n\n"):
        while len(p) > max_len:
            cut = p.rfind("\n", 0, max_len)
            if cut < max_len // 5:
                cut = max_len
            if acc.strip():
                out.append(acc.strip())
                acc = ""
            out.append(p[:cut].strip())
            p = p[cut:]
        if len(acc) + len(p) > max_len and acc.strip():
            out.append(acc.strip())
            acc = ""
        acc += p + "\n\n"
    if acc.strip():
        out.append(acc.strip())
    return out


def split_markdown(content: str, rel: str, min_len: int = 30, max_len: int = 1200,
                   preamble_label: str = DEFAULT_PREAMBLE) -> list[Chunk]:
    name = os.path.basename(rel)
    if name.endswith(".md"):
        name = name[:-3]
    lines = content.split("\n")
    raw: list[tuple[list[str], int, list[str]]] = []   # (stack, depth, buf)
    stack: list[str] = [name]
    depth = 0
    buf: list[str] = []
    infence = False
    started = False

    def flush():
        if buf and "\n".join(buf).strip():
            raw.append((list(stack), depth, list(buf)))

    for ln in lines:
        if ln.strip().startswith("```"):
            infence = not infence
        m = None if infence else HEADING_RE.match(ln)
        if m:
            flush()
            lvl, title = len(m.group(1)), m.group(2).strip()
            stack = stack[:lvl]          # stack[0] is always the file name
            while len(stack) < lvl:
                stack.append("")
            stack.append(title)
            depth = lvl
            buf = []
            started = True
        else:
            if not started:
                started = True
                stack = [name, preamble_label]
                depth = 0
            buf.append(ln)
    flush()

    chunks: list[Chunk] = []
    for st, dp, bl in raw:
        heading = " > ".join(h for h in st if h)
        body = "\n".join(bl).strip()
        if len(body) < min_len:
            continue
        for piece in _split_long(body, max_len):
            if len(piece) >= min_len:
                chunks.append(Chunk(heading=heading, text=piece, depth=dp))
    for i, c in enumerate(chunks):
        c.ord = i
    return chunks


def split_plain(content: str, rel: str, min_len: int = 30, max_len: int = 1200) -> list[Chunk]:
    """Non-Markdown: the whole file is one chunk, split by paragraph / line when too long. The heading is the
    relative path, which keeps the directory semantics (e.g. sql/reports/daily.sql)."""
    body = content.strip()
    if len(body) < min_len:
        return []
    chunks = [Chunk(heading=rel, text=p, depth=1) for p in _split_long(body, max_len) if len(p) >= min_len]
    for i, c in enumerate(chunks):
        c.ord = i
    return chunks


def split_file(content: str, rel: str, min_len: int = 30, max_len: int = 1200,
               preamble_label: str = DEFAULT_PREAMBLE) -> list[Chunk]:
    if rel.lower().endswith(".md"):
        return split_markdown(content, rel, min_len, max_len, preamble_label)
    return split_plain(content, rel, min_len, max_len)


def parent_index(chunks: list[Chunk], i: int) -> int | None:
    """Index of the closest preceding chunk with a smaller depth, or None."""
    d = chunks[i].depth
    for j in range(i - 1, -1, -1):
        if chunks[j].depth < d:
            return j
    return None
