"""data.gov.in OGD API client.

The key lives in ~/.config/aisehack/datagovin.key, OUTSIDE the repo, and is never
printed, logged or written into any output. Nothing in this module echoes it.

The API is paged and flaky: a 1000-row request times out where 250 succeeds, so the
page size is deliberately small and every call retries. [R2 e10]
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://api.data.gov.in/resource"
KEYFILE = Path.home() / ".config" / "aisehack" / "datagovin.key"
PAGE = 250


def _key():
    if not KEYFILE.exists():
        raise RuntimeError(f"no data.gov.in key at {KEYFILE}")
    return KEYFILE.read_text(encoding="utf8").strip()


def _get(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "aisehack-r3"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:                                   # noqa: BLE001
            if k == tries - 1:
                raise
            # never include the URL in the message: it carries the key
            print(f"  retry {k + 1}: {type(e).__name__}")
            time.sleep(2 * (k + 1))


def resource(rid, limit=None, filters=None):
    """Fetch a resource, paging until exhausted. Returns (records, metadata)."""
    out, off, meta = [], 0, None
    while True:
        q = {"api-key": _key(), "format": "json", "limit": PAGE, "offset": off}
        for k, v in (filters or {}).items():
            q[f"filters[{k}]"] = v
        j = _get(f"{BASE}/{rid}?{urllib.parse.urlencode(q)}")
        if meta is None:
            meta = {k: v for k, v in j.items() if k != "records"}
        recs = j.get("records", [])
        out.extend(recs)
        off += len(recs)
        if not recs or len(recs) < PAGE or (limit and len(out) >= limit):
            break
    return out, meta


def catalog(query, limit=40):
    """Search the catalogue for resources matching a phrase."""
    q = {"api-key": _key(), "format": "json", "limit": limit, "filters[title]": query}
    return _get(f"https://api.data.gov.in/catalog?{urllib.parse.urlencode(q)}")
