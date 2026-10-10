from __future__ import annotations

from kbindex.links import Resolver, extract_raw_links


def test_extract():
    text = ("See [[deploy]], [[ops/backups|backups]], [[deploy#Rollback]] and [guide](../guide.md#intro). "
            "Not these: ![img](pic.md) [web](https://example.com/a.md) [anchor](#top) [file](notes.txt) "
            '[titled](rel/page.md "Title")')
    assert extract_raw_links(text) == [
        "wiki:deploy", "wiki:ops/backups", "wiki:deploy",
        "md:../guide.md", "md:rel/page.md",
    ]


def test_resolve_wiki_and_relative():
    r = Resolver(["README.md", "deploy.md", "ops/backups.md", "a/x.md", "b/x.md", "docs/My Page.md"])
    assert r.resolve("wiki:deploy", "README.md") == "deploy.md"
    assert r.resolve("wiki:ops/backups", "README.md") == "ops/backups.md"
    assert r.resolve("wiki:backups", "README.md") == "ops/backups.md"          # unique file name
    assert r.resolve("wiki:x", "a/y.md") == "a/x.md"                             # ambiguous: same directory tree wins
    assert r.resolve("wiki:missing", "README.md") is None
    assert r.resolve("md:../README.md", "ops/backups.md") == "README.md"
    assert r.resolve("md:My%20Page.md", "docs/index.md") == "docs/My Page.md"
    assert r.resolve("md:nope.md", "README.md") is None
