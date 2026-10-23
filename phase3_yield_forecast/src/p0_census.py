"""Phase 0: census every Sentinel-1 and Sentinel-2 scene over Sokhda for the season.

This is the *count* that settles the question 5.1 raises: Round 2 measured zero
S2 scenes under 20% cloud over Jun-Sep and concluded "there is no optical here".
That was a statement about the monsoon. Round 3's two new acquisitions sit in the
post-monsoon dry season. Before anything is assumed either way, count.

Kill criterion for the optical lead, written before the run: fewer than three S2
scenes under 20% cloud between 1 Oct and 30 Nov means there is no usable
late-season optical witness and the lead dies here.

Writes results/tables/p0_scene_census.csv -- metadata only, no pixels.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATES, TABLES, log, scene
from stac import aoi_bbox, search

WINDOW = "2025-05-01/2025-12-15"


def main():
    bbox = aoi_bbox()
    rows = []

    s2 = search("sentinel-2-l2a", WINDOW, bbox, limit=500)
    for it in s2:
        p = it["properties"]
        rows.append({"collection": "sentinel-2-l2a", "id": it["id"],
                     "datetime": p["datetime"], "cloud": round(p.get("eo:cloud_cover", -1), 3),
                     "extra": p.get("s2:mgrs_tile", "")})

    s1 = search("sentinel-1-rtc", WINDOW, bbox, limit=500)
    for it in s1:
        p = it["properties"]
        rows.append({"collection": "sentinel-1-rtc", "id": it["id"],
                     "datetime": p["datetime"], "cloud": "",
                     "extra": f"{p.get('sat:orbit_state','')}/relorbit{p.get('sat:relative_orbit','')}"
                              f"/{'+'.join(p.get('sar:polarizations', []))}"})

    rows.sort(key=lambda r: (r["collection"], r["datetime"]))
    out = TABLES / "p0_scene_census.csv"
    with out.open("w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, ["collection", "id", "datetime", "cloud", "extra"])
        w.writeheader()
        w.writerows(rows)

    n_s2 = sum(r["collection"] == "sentinel-2-l2a" for r in rows)
    n_s1 = len(rows) - n_s2
    clear = [r for r in rows if r["collection"] == "sentinel-2-l2a"
             and r["cloud"] != -1 and r["cloud"] < 20]
    late = [r for r in clear if "2025-10" <= r["datetime"][:7] <= "2025-11"]
    log("p0_census", s2=n_s2, s1=n_s1, s2_under20=len(clear), s2_under20_octnov=len(late),
        verdict="optical lead LIVES" if len(late) >= 3 else "optical lead DIES",
        out=str(out))

    print("\n-- S2 scenes under 20% cloud --")
    for r in clear:
        print(f"   {r['datetime'][:10]}  {r['cloud']:>7}%  {r['id']}")
    print("\n-- S2 monthly: scenes / min cloud --")
    for m in sorted({r["datetime"][:7] for r in rows if r["collection"] == "sentinel-2-l2a"}):
        mm = [r for r in rows if r["collection"] == "sentinel-2-l2a" and r["datetime"][:7] == m]
        print(f"   {m}  n={len(mm):>3}  min_cloud={min(r['cloud'] for r in mm):>7}%")
    print("\n-- S1 by relative orbit --")
    for k in sorted({r["extra"] for r in rows if r["collection"] == "sentinel-1-rtc"}):
        kk = [r for r in rows if r["extra"] == k]
        print(f"   {k:<40} n={len(kk):>3}  {kk[0]['datetime'][:10]} .. {kk[-1]['datetime'][:10]}")
    print("\n-- nearest S1/S2 to each Capella date --")
    for d in DATES:
        cd = scene(d)["utc"][:10]
        for coll in ("sentinel-1-rtc", "sentinel-2-l2a"):
            cc = [r for r in rows if r["collection"] == coll]
            near = min(cc, key=lambda r: abs(_days(r["datetime"][:10], cd)))
            print(f"   {d}  {coll:<16} {near['datetime'][:10]} "
                  f"({_days(near['datetime'][:10], cd):+d}d) cloud={near['cloud']}")


def _days(a, b):
    from datetime import date
    return (date(*map(int, a.split("-"))) - date(*map(int, b.split("-")))).days


if __name__ == "__main__":
    main()
