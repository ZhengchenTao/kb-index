"""Source readers: git repositories (bare or not) via `git ls-tree` / `git show`; plain directories via an
mtime + content-hash scan.

A git source needs no working tree: the blob sha is the content hash, so unchanged files are never read.
Mount repositories read-only. **Never run a write operation inside a bare repository owned by a git server**
(not even `git remote update`): objects created by another user can leave the server unable to write.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import Source, matches_any


@dataclass
class FileEntry:
    rel: str
    hash: str          # git blob sha, or the same algorithm over the file bytes
    mtime: float = 0.0
    size: int = 0


class GitSource:
    def __init__(self, src: Source):
        self.src = src
        self.path = src.path

    def _git(self, *args: str, binary: bool = False) -> bytes | str:
        cmd = ["git", "-c", "safe.directory=*", "-C", self.path, *args]
        r = subprocess.run(cmd, capture_output=True, check=True)
        return r.stdout if binary else r.stdout.decode("utf-8", "replace")

    def head(self) -> str:
        return self._git("rev-parse", self.src.ref).strip()  # type: ignore[union-attr]

    def list_files(self) -> list[FileEntry]:
        out = self._git("ls-tree", "-r", "-z", "--long", self.src.ref)
        entries: list[FileEntry] = []
        for rec in out.split("\0"):  # type: ignore[union-attr]
            if not rec:
                continue
            meta, _, path = rec.partition("\t")
            parts = meta.split()
            if len(parts) < 4 or parts[1] != "blob":
                continue
            sha, size = parts[2], parts[3]
            if not matches_any(path, self.src.include):
                continue
            if self.src.only and not any(path.startswith(p) for p in self.src.only):
                continue
            entries.append(FileEntry(rel=path, hash=sha, size=int(size) if size.isdigit() else 0))
        return entries

    def read(self, rel: str) -> str:
        data = self._git("show", f"{self.src.ref}:{rel}", binary=True)
        return data.decode("utf-8", "replace")  # type: ignore[union-attr]

    def describe(self) -> str:
        return f"git {self.path}@{self.src.ref}"


class DirSource:
    def __init__(self, src: Source):
        self.src = src
        self.root = Path(src.path)

    def head(self) -> str:
        return ""

    def list_files(self, known: dict[str, tuple[float, int, str]] | None = None) -> list[FileEntry]:
        """known: rel -> (mtime, size, hash). If mtime and size are unchanged the old hash is reused (no read)."""
        known = known or {}
        entries: list[FileEntry] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in filenames:
                full = Path(dirpath) / fn
                rel = full.relative_to(self.root).as_posix()
                if not matches_any(rel, self.src.include):
                    continue
                if self.src.only and not any(rel.startswith(p) for p in self.src.only):
                    continue
                st = full.stat()
                k = known.get(rel)
                if k and abs(k[0] - st.st_mtime) < 1e-6 and k[1] == st.st_size:
                    entries.append(FileEntry(rel, k[2], st.st_mtime, st.st_size))
                    continue
                data = full.read_bytes()
                h = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()   # same algorithm as a git blob sha
                entries.append(FileEntry(rel, h, st.st_mtime, st.st_size))
        return entries

    def read(self, rel: str) -> str:
        return (self.root / rel).read_text(encoding="utf-8", errors="replace")

    def describe(self) -> str:
        return f"dir {self.root}"


def open_source(src: Source):
    if src.type == "git":
        return GitSource(src)
    if src.type == "dir":
        return DirSource(src)
    raise ValueError(f"unknown source type {src.type!r} for {src.repo}")
