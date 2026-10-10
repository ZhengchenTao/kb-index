"""Embedding usage accounting: one JSON line per embedding request, appended to a local file.

Why record it ourselves: the Gemini embedding endpoints return no usage metadata (only generateContent has
`usageMetadata`), and an AI Studio API key cannot query any usage endpoint (Cloud Monitoring and billing
export need OAuth / a service account, and report requests rather than tokens). Client-side accounting is
the only option. Other providers are recorded the same way with estimated token counts.

The file is a buffer for an external collector (for example a cron job or a metrics agent that tails it and
alerts on a monthly budget); this module neither knows nor cares who reads it.

Location: $KB_USAGE_LOG, else <directory of $KB_DB>/embed-usage.jsonl, else nothing is recorded.

The line format is an interface (collectors parse it): ts, date, kind, model, tokens, items, chars, exact.

**Never raises**: losing a day of usage data is fine, a failed search or index run because of accounting is not.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

# In HTTP mode concurrent searches run in a thread pool; append + rotate must be serialised.
_lock = threading.Lock()


def log_path() -> Path | None:
    p = os.environ.get("KB_USAGE_LOG")
    if p:
        return Path(p)
    db = os.environ.get("KB_DB")
    if db:
        return Path(db).parent / "embed-usage.jsonl"
    return None


def record(kind: str, model: str, tokens: int, items: int, chars: int, exact: bool) -> None:
    """kind: index (batch document embedding, the bulk) / query (one per search).

    exact=True means tokens came from the provider's token counter; False means a local estimate. The query
    side always estimates: an extra round trip on every search would add straight to latency, and query volume
    is a rounding error next to indexing.
    """
    p = log_path()
    if p is None:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "date": time.strftime("%Y-%m-%d"),
            "kind": kind,
            "model": model,
            "tokens": int(tokens),
            "items": int(items),
            "chars": int(chars),
            "exact": bool(exact),
        }
        with _lock:
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            _rotate(p)
    except Exception:
        pass  # see the module docstring: accounting must never break the main path


MAX_BYTES = 2 * 1024 * 1024
KEEP_LINES = 20000


def _keep_from() -> str:
    """Retention watermark = the 1st of last month. Current + previous month covers any monthly rollup."""
    t = time.localtime()
    y, m = (t.tm_year - 1, 12) if t.tm_mon == 1 else (t.tm_year, t.tm_mon - 1)
    return f"{y:04d}-{m:02d}-01"


def _rotate(p: Path) -> None:
    """Trim old lines once the file exceeds 2 MB.

    **Trim by month, not by line count**: a collector may sum "tokens this month" from this file to alert on a
    monthly budget. Trimming a fixed number of lines would drop early-month lines in a busy month and make the
    monthly total shrink, which silently disables the alert. Keeping current + previous month makes that
    impossible by construction. KEEP_LINES is only a fallback when dates cannot be parsed.

    Collectors are expected to aggregate by date and upsert, so re-reading old lines is harmless.
    Trimming happens here rather than in the collector because collectors may only have read access."""
    if p.stat().st_size <= MAX_BYTES:
        return
    keep_from = _keep_from()
    with open(p, encoding="utf-8") as f:
        lines = f.readlines()
    kept = []
    for ln in lines:
        try:
            if json.loads(ln).get("date", "") >= keep_from:
                kept.append(ln)
        except Exception:
            continue
    if not kept:                       # nothing parsed -> fall back to line count, never empty the file
        kept = lines[-KEEP_LINES:]
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(kept)
    os.replace(tmp, p)


def estimate_tokens(texts: list[str]) -> int:
    """Local estimate: roughly 1 token per CJK character and 1 token per 4 other characters.

    Used for queries (tens of tokens, being a few off does not matter) and for providers without a token
    counter."""
    total = 0
    for t in texts:
        cjk = sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff")
        total += cjk + max(0, (len(t) - cjk)) // 4
    return total
