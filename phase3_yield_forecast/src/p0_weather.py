"""Phase 0: hourly meteorology at every Capella overpass, plus the season series.

Two of the six acquisitions are at 01:37 and 19:22 local -- pre-dawn and post-sunset,
against four morning/midday passes. Dew is a documented several-dB term at X-band, so
a "senescence" signal read off 29 October could be a dew signal. This fetches the
independent meteorological check: relative humidity, dew-point depression, soil
moisture and precipitation at the exact overpass hour.

Open-Meteo archive (ERA5-Land reanalysis), free, no key. It is ONE value for all 966
farms, so under the resolution-ceiling rule it may only be used for temporal context
and never to rank farms. [R2 e12 ceiling: 11 km and beyond explains 0.000]

Writes results/tables/p0_weather_overpass.csv and p0_weather_hourly.csv.
"""
import csv
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATES, TABLES, log, scene
from stac import aoi_bbox

API = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
          "soil_moisture_0_to_7cm", "soil_temperature_0_to_7cm", "wind_speed_10m",
          "surface_pressure", "cloud_cover"]


def _get(lat, lon, start, end, tries=4):
    q = urllib.parse.urlencode({"latitude": lat, "longitude": lon,
                                "start_date": start, "end_date": end,
                                "hourly": ",".join(HOURLY), "timezone": "UTC"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=180) as r:
                return json.load(r)
        except Exception as e:                      # noqa: BLE001
            if k == tries - 1:
                raise
            print(f"  retry {k+1} {start}..{end}: {type(e).__name__}")
            import time as _t
            _t.sleep(2 * (k + 1))


def fetch(lat, lon, start, end):
    """Month-by-month. The server drops a 9-variable, 7-month request mid-stream
    (WinError 10054) but serves the same query a month at a time without complaint."""
    d0 = datetime.fromisoformat(start).date()
    d1 = datetime.fromisoformat(end).date()
    out = None
    while d0 <= d1:
        nxt = (d0.replace(day=28) + timedelta(days=4)).replace(day=1)
        chunk = _get(lat, lon, d0.isoformat(), min(nxt - timedelta(days=1), d1).isoformat())
        if out is None:
            out = chunk
        else:
            for k, v in chunk["hourly"].items():
                out["hourly"][k].extend(v)
        d0 = nxt
    return out


def main():
    x0, y0, x1, y1 = aoi_bbox(0.0)
    lat, lon = round((y0 + y1) / 2, 4), round((x0 + x1) / 2, 4)
    j = fetch(lat, lon, "2025-05-01", "2025-12-15")
    h = j["hourly"]
    idx = {t: i for i, t in enumerate(h["time"])}

    with (TABLES / "p0_weather_hourly.csv").open("w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["time_utc"] + HOURLY)
        for i, t in enumerate(h["time"]):
            w.writerow([t] + [h[k][i] for k in HOURLY])

    rows = []
    for d in DATES:
        s = scene(d)
        t = datetime.fromisoformat(s["utc"].replace("Z", "+00:00")).replace(
            minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        key = t.strftime("%Y-%m-%dT%H:00")
        i = idx[key]
        r = {"date": d, "local": s["local"][:19], "utc_hour": key,
             "pointing": s["pointing"]}
        r.update({k: h[k][i] for k in HOURLY})
        # dew-point depression: small or negative => condensation on the canopy
        r["dewpoint_depression_C"] = round(r["temperature_2m"] - r["dew_point_2m"], 2)
        # rain in the 6 h and 24 h before the overpass -- R2's rice channel was a
        # 17.3 mm event in the six hours before the 19 June pass. [R2 e12]
        r["precip_6h_mm"] = round(sum(h["precipitation"][max(0, i - 5):i + 1]), 2)
        r["precip_24h_mm"] = round(sum(h["precipitation"][max(0, i - 23):i + 1]), 2)
        r["precip_72h_mm"] = round(sum(h["precipitation"][max(0, i - 71):i + 1]), 2)
        rows.append(r)

    cols = list(rows[0])
    out = TABLES / "p0_weather_overpass.csv"
    with out.open("w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)

    log("p0_weather", lat=lat, lon=lon, hours=len(h["time"]), out=str(out))
    print(f"\n{'date':<10}{'local':<18}{'look':<7}{'RH%':>6}{'dewdep':>8}"
          f"{'soilm':>8}{'p6h':>7}{'p24h':>7}{'p72h':>7}{'T':>7}")
    for r in rows:
        print(f"{r['date']:<10}{r['local'][11:16]:<18}{r['pointing']:<7}"
              f"{r['relative_humidity_2m']:>6}{r['dewpoint_depression_C']:>8}"
              f"{r['soil_moisture_0_to_7cm']:>8}{r['precip_6h_mm']:>7}"
              f"{r['precip_24h_mm']:>7}{r['precip_72h_mm']:>7}{r['temperature_2m']:>7}")


if __name__ == "__main__":
    main()
