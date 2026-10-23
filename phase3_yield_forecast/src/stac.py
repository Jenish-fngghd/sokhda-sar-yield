"""Microsoft Planetary Computer STAC access -- witness data only.

Sentinel-1 RTC and Sentinel-2 L2A as public COGs, anonymous SAS token, open
Copernicus licence. urllib only: no pystac_client / planetary_computer dependency,
because the whole client surface we need is one POST and one token fetch.

WITNESS DISCIPLINE: nothing fetched here enters a shipped number without an explicit,
argued promotion that first names and freezes a replacement witness. [prompt 5.1]
"""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FARMS, VILLAGE  # noqa: E402  (after PROJ_LIB pop in common)

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SASAPI = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
_tokens = {}


def sign(href):
    """PC assets need a per-collection SAS token appended. Cached per collection."""
    acct = href.split("//")[1].split(".")[0]
    coll = href.split("/")[3]
    key = f"{acct}/{coll}"
    if key not in _tokens:
        with urllib.request.urlopen(f"{SASAPI}/{acct}/{coll}", timeout=60) as r:
            _tokens[key] = json.load(r)["token"]
    return f"{href}?{_tokens[key]}"


def search(collection, datetime, bbox, query=None, limit=100):
    body = {"collections": [collection], "bbox": list(map(float, bbox)),
            "datetime": datetime, "limit": limit}
    if query:
        body["query"] = query
    req = urllib.request.Request(f"{STAC}/search", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["features"]


def aoi_bbox(pad=0.01):
    """Village bbox in WGS84, padded, from the delivered shapefile."""
    import shapefile
    sf = shapefile.Reader(str(VILLAGE))
    x0, y0, x1, y1 = sf.bbox
    return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)


if __name__ == "__main__":
    print("village bbox:", aoi_bbox(0.0))
    print("padded      :", aoi_bbox())
