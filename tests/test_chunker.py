from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kbindex.chunker import parent_index, split_file, split_markdown, split_plain
from kbindex.links import extract_raw_links

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).with_name("golden_examples.json")


def test_preamble_label_default_and_custom():
    md = "Intro text long enough to be kept as its own chunk.\n\n# Title\n\nBody text that is also long enough to keep."
    a = split_markdown(md, "dir/doc.md")
    assert a[0].heading == "doc > (preamble)" and a[0].depth == 0
    assert a[1].heading == "doc > Title" and a[1].depth == 1
    b = split_markdown(md, "dir/doc.md", preamble_label="(intro)")
    assert b[0].heading == "doc > (intro)"
    assert [c.text for c in a] == [c.text for c in b]


def test_heading_stack_and_skipped_levels():
    md = ("# A\n\nfirst section body, long enough to keep around.\n\n"
          "### C\n\nthird level directly under the first, long enough.\n\n"
          "## B\n\nsecond level body text that is long enough to keep.")
    cs = split_markdown(md, "x.md")
    assert [c.heading for c in cs] == ["x > A", "x > A > C", "x > A > B"]
    assert [c.depth for c in cs] == [1, 3, 2]
    assert [c.ord for c in cs] == [0, 1, 2]
    assert parent_index(cs, 1) == 0 and parent_index(cs, 2) == 0 and parent_index(cs, 0) is None


def test_headings_inside_code_fences_are_text():
    md = "# Real\n\n```sh\n# not a heading, just a shell comment in a code block\necho hi\n```\n"
    cs = split_markdown(md, "x.md")
    assert len(cs) == 1 and cs[0].heading == "x > Real" and "# not a heading" in cs[0].text


def test_min_length_filter_and_long_split():
    para = "word " * 100               # ~500 chars
    md = "# Tiny\n\nshort\n\n# Big\n\n" + "\n\n".join([para] * 6)
    cs = split_markdown(md, "x.md", min_len=30, max_len=1200)
    assert all(c.heading == "x > Big" for c in cs)
    assert len(cs) >= 3 and all(len(c.text) <= 1200 for c in cs)


def test_hard_cut_without_blank_lines():
    table = "\n".join(f"| row {i} | " + "x" * 50 + " |" for i in range(100))   # no blank line anywhere
    cs = split_markdown("# T\n\n" + table, "t.md", max_len=1200)
    assert len(cs) > 1 and all(len(c.text) <= 1200 for c in cs)


def test_plain_files_use_path_as_heading():
    cs = split_plain("select 1;\n" * 10, "sql/reports/daily.sql")
    assert cs[0].heading == "sql/reports/daily.sql" and cs[0].depth == 1
    assert split_file("select 1;\n" * 10, "sql/a.SQL")[0].heading == "sql/a.SQL"
    assert split_plain("tiny", "a.sql") == []


def test_embed_text_and_hash():
    c = split_markdown("# H\n\n" + "y" * 7000, "a.md", max_len=10000)[0]
    assert c.embed_text == ("a > H\n" + "y" * 7000)[:6000]
    assert c.hash == hashlib.sha1(c.embed_text.encode("utf-8")).hexdigest()


def _snapshot() -> dict:
    out = {}
    base = ROOT / "examples" / "docs"
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(base).as_posix()
        repo, _, inner = rel.partition("/")
        chunks = split_file(p.read_text(encoding="utf-8"), inner)
        out[rel] = [[c.heading, hashlib.sha1(c.text.encode("utf-8")).hexdigest(), c.depth, c.ord, c.hash,
                     extract_raw_links(c.text)] for c in chunks]
    return out


def test_golden_snapshot_of_examples():
    """Chunking is a frozen contract: any change here re-embeds every deployment's corpus.
    If a change is intended, regenerate with `PYTHONPATH=. python tests/test_chunker.py --update` and say so in the commit."""
    assert GOLDEN.exists(), "run PYTHONPATH=. python tests/test_chunker.py --update"
    assert _snapshot() == json.loads(GOLDEN.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import sys
    import kbindex
    assert Path(kbindex.__file__).resolve().is_relative_to(ROOT), f"run from the repo root with PYTHONPATH=. ({kbindex.__file__})"
    if "--update" in sys.argv:
        GOLDEN.write_text(json.dumps(_snapshot(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("updated", GOLDEN)
