"""Link extraction: `[[wiki links]]` and `[text](relative.md)`, resolved to repo-relative paths.

These edges are a human-curated graph that costs nothing to compute.
"""
from __future__ import annotations

import os
import posixpath
import re

WIKI_RE = re.compile(r"\[\[([^\]\|#]+)(?:#[^\]\|]*)?(?:\|[^\]]*)?\]\]")
MD_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+?)(?:\s+\"[^\"]*\")?\)")


def extract_raw_links(text: str) -> list[str]:
    out: list[str] = []
    for m in WIKI_RE.finditer(text):
        out.append("wiki:" + m.group(1).strip())
    for m in MD_RE.finditer(text):
        t = m.group(1).strip()
        if t.startswith(("http://", "https://", "mailto:", "#", "data:")):
            continue
        t = t.split("#", 1)[0]
        if t and t.lower().endswith(".md"):
            out.append("md:" + t)
    return out


class Resolver:
    """Resolve raw links to repo-relative paths. Wiki links follow Obsidian: exact path first, then a unique
    match on the file name alone."""

    def __init__(self, paths: list[str]):
        self.paths = set(paths)
        self.by_name: dict[str, list[str]] = {}
        for p in paths:
            base = posixpath.basename(p)
            stem = base[:-3] if base.endswith(".md") else base
            self.by_name.setdefault(stem, []).append(p)
            self.by_name.setdefault(base, []).append(p)

    def resolve(self, raw: str, from_rel: str) -> str | None:
        kind, target = raw.split(":", 1)
        target = target.replace("\\", "/").strip("/")
        if kind == "wiki":
            cands = [target, target + ".md"]
            for c in cands:
                if c in self.paths:
                    return c
            stem = posixpath.basename(target)
            hits = self.by_name.get(stem) or self.by_name.get(stem + ".md") or []
            if len(hits) == 1:
                return hits[0]
            if hits:
                # several files share the name: prefer one under the linking file's directory
                fd = posixpath.dirname(from_rel)
                same = [h for h in hits if h.startswith(fd + "/")] if fd else []
                return same[0] if len(same) == 1 else hits[0]
            return None
        # relative Markdown link
        base = posixpath.dirname(from_rel)
        joined = posixpath.normpath(posixpath.join(base, target)) if not target.startswith("/") else target.lstrip("/")
        joined = joined.replace("\\", "/")
        if joined in self.paths:
            return joined
        try:
            from urllib.parse import unquote
            j2 = unquote(joined)
            if j2 in self.paths:
                return j2
        except Exception:
            pass
        return None


def _norm(p: str) -> str:
    return os.path.normpath(p).replace("\\", "/")
