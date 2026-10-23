"""Stage I1 -- SLC to calibrated, geocoded gamma0 on a common grid.

Chain, with the evidence tag for each step:

    complex SLC (slant range, ungeocoded)
      -> beta0 = scale_factor * |z|^2                       [B9, confirmed: the product's
                                                             own `radiometry` field reads
                                                             "beta_nought"]
      -> per-pixel incidence theta from ORBIT STATE VECTORS  [validated to 0.006 deg against
                                                             the vendor's annotated centre
                                                             incidence, all 4 dates]
      -> sigma0 = beta0 * sin(theta)
         gamma0  = sigma0 / cos(theta) = beta0 * tan(theta)  [O6]
      -> geocode with the 225 GCPs to EPSG:32643, resampling
         by AVERAGE so that geocoding and multilooking happen
         in one pass (averaging POWER is exactly multilooking)
      -> two grids: FINE 2 m (texture, within-field stats)
                    BASE 5 m (farm means, temporal trends)

Deliberately NOT done here: incidence normalisation to a reference angle and
water-point referencing. Both need all four dates on a common grid first, so they
live in the next stage.

Run:  python src/prep_r2.py            (all dates, both grids)
      python src/prep_r2.py --trial    (Oct only, BASE only -- quick check)
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_gcps, from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from scipy.interpolate import griddata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, DATES, FARMS, REF_INCIDENCE_DEG, log, slc_path

import geopandas as gpd
import pyproj

UTM = 32643
GRIDS = {"base": 5.0, "fine": 2.0}
AOI_BUFFER_M = 400.0      # margin around the farms, in metres
SLANT_MARGIN_PX = 200     # margin on the slant-range crop, in pixels
CHUNK_ROWS = 1024

_LL2ECEF = pyproj.Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)


def _t2s(t: str) -> float:
    return datetime.fromisoformat(t.replace("Z", "")[:26]).timestamp()


def scene_meta(date):
    js = json.loads(sorted(slc_path(date).parent.glob("*_extended.json"))[0]
                    .read_text(encoding="utf8"))
    c = js["collect"]
    return c, c["image"], c["image"]["image_geometry"]


def incidence_at(date, rows, cols, gcps):
    """Incidence angle (deg) at arbitrary (row, col), from orbit geometry.

    For each GCP we know both its image position (row, col) and its ground
    position (lon, lat, h). The sensor position at that GCP's azimuth time comes
    from interpolating the 108 state vectors. The incidence angle is then the
    angle between the line of sight and the local ellipsoid normal -- no
    approximation, no assumed constant.

    This reproduces the vendor's annotated centre incidence to <0.007 deg on all
    four dates, which is what licenses us to use it. Incidence is then
    interpolated from the 15x15 GCP lattice to the pixels we want; it varies
    smoothly and by only ~0.4 deg across a whole 27 km strip, so interpolation
    error is negligible against the 6.55 deg spread BETWEEN dates.
    """
    c, im, ig = scene_meta(date)
    sv = c["state"]["state_vectors"]
    st = np.array([_t2s(v["time"]) for v in sv])
    P = np.array([v["position"] for v in sv])
    t0, dt = _t2s(ig["first_line_time"]), ig["delta_line_time"]

    gr = np.array([p.row for p in gcps]); gc = np.array([p.col for p in gcps])
    X, Y, Z = _LL2ECEF.transform(np.array([p.x for p in gcps]),
                                 np.array([p.y for p in gcps]),
                                 np.array([p.z for p in gcps]))
    T = np.c_[X, Y, Z]
    S = np.c_[[np.interp(t0 + gr * dt, st, P[:, k]) for k in range(3)]].T

    los = S - T
    los /= np.linalg.norm(los, axis=1)[:, None]
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    up = np.c_[T[:, 0], T[:, 1], T[:, 2] / (1 - e2)]
    up /= np.linalg.norm(up, axis=1)[:, None]
    inc_gcp = np.degrees(np.arccos(np.sum(los * up, axis=1)))

    out = griddata((gr, gc), inc_gcp, (rows, cols), method="linear")
    nan = np.isnan(out)
    if nan.any():   # AOI can sit outside the GCP convex hull at the strip edge
        out[nan] = griddata((gr, gc), inc_gcp, (rows[nan], cols[nan]), method="nearest")
    return out, float(np.nanmean(inc_gcp))


def aoi_window(date, gcps, height, width, bounds_ll):
    """Slant-range window covering the AOI, from the GCP-derived affine.

    The affine is only an approximation of a curved SAR geometry, so the window
    is padded and clipped rather than trusted exactly. It exists purely to avoid
    reading a 27000 x 4700 strip when we need ~30% of it.
    """
    inv = ~from_gcps(gcps)
    lon0, lat0, lon1, lat1 = bounds_ll
    rc = [inv * (X, Y) for X in (lon0, lon1) for Y in (lat0, lat1)]
    cs = [p[0] for p in rc]; rs = [p[1] for p in rc]
    c0 = max(0, int(min(cs)) - SLANT_MARGIN_PX)
    c1 = min(width, int(max(cs)) + SLANT_MARGIN_PX)
    r0 = max(0, int(min(rs)) - SLANT_MARGIN_PX)
    r1 = min(height, int(max(rs)) + SLANT_MARGIN_PX)
    return Window(c0, r0, c1 - c0, r1 - r0)


def slant_gamma0(date):
    """Read the AOI window and return (gamma0, sigma0, incidence, shifted GCPs)."""
    _, im, _ = scene_meta(date)
    sc = float(im["scale_factor"])
    assert im["radiometry"] == "beta_nought", im["radiometry"]

    farms = gpd.read_file(FARMS)
    b = farms.buffer(AOI_BUFFER_M / 111320.0).total_bounds  # crude deg buffer, padded anyway

    with rasterio.open(slc_path(date)) as s:
        gcps, gcrs = s.gcps
        win = aoi_window(date, gcps, s.height, s.width, b)
        h, w = int(win.height), int(win.width)
        beta = np.empty((h, w), dtype="float32")
        for r in range(0, h, CHUNK_ROWS):
            n = min(CHUNK_ROWS, h - r)
            z = s.read(1, window=Window(win.col_off, win.row_off + r, w, n))
            # beta0 = scale * |z|^2  -- NOT scale^2. The R1 bug squared the scale
            # factor, which because sc differs per scene imposed a per-date offset
            # of up to 2.4 dB, the same size as the crop signal itself. [B9]
            beta[r:r + n] = (sc * (z.real.astype("float32") ** 2
                                   + z.imag.astype("float32") ** 2))

    rr, cc = np.meshgrid(np.arange(h, dtype="float32") + win.row_off,
                         np.arange(w, dtype="float32") + win.col_off, indexing="ij")
    inc, inc_mean = incidence_at(date, rr, cc, gcps)
    inc = inc.astype("float32")
    del rr, cc

    th = np.radians(inc)
    sigma0 = beta * np.sin(th)
    gamma0 = sigma0 / np.cos(th)

    # GCPs are given in FULL-image pixel coordinates; shift them into the window's
    # frame so the warp knows where the cropped array actually sits.
    shifted = []
    for p in gcps:
        q = rasterio.control.GroundControlPoint(
            row=p.row - win.row_off, col=p.col - win.col_off, x=p.x, y=p.y, z=p.z)
        shifted.append(q)

    log("prep.slant", date=date, win=f"{h}x{w}", scale=sc,
        inc_mean=round(inc_mean, 4), inc_min=round(float(inc.min()), 3),
        inc_max=round(float(inc.max()), 3),
        beta_med=round(float(np.median(beta)), 5))
    return gamma0, sigma0, inc, shifted, gcrs


def target_grid(res):
    farms = gpd.read_file(FARMS).to_crs(UTM)
    x0, y0, x1, y1 = farms.total_bounds
    x0 = np.floor((x0 - AOI_BUFFER_M) / res) * res
    y0 = np.floor((y0 - AOI_BUFFER_M) / res) * res
    x1 = np.ceil((x1 + AOI_BUFFER_M) / res) * res
    y1 = np.ceil((y1 + AOI_BUFFER_M) / res) * res
    w = int(round((x1 - x0) / res)); h = int(round((y1 - y0) / res))
    return from_origin(x0, y1, res, res), w, h


def geocode(src, gcps, gcrs, res, resampling=Resampling.average):
    """Slant range -> EPSG:32643. Average resampling IS the multilook.

    Averaging power over the several slant-range pixels that fall in each ground
    cell is precisely what multilooking does: speckle variance falls as 1/looks
    and ENL rises from the product's annotated 1.0. Doing it inside the warp
    avoids a separate resampling step and the extra interpolation blur that would
    come with it.
    """
    tf, w, h = target_grid(res)
    dst = np.full((h, w), np.nan, dtype="float32")
    reproject(source=src, destination=dst,
              gcps=gcps, src_crs=gcrs, src_nodata=np.nan,
              dst_transform=tf, dst_crs=f"EPSG:{UTM}", dst_nodata=np.nan,
              resampling=resampling)
    return dst, tf


def write(path, arr, tf, **tags):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", crs=f"EPSG:{UTM}", transform=tf,
                       nodata=np.nan, compress="deflate", predictor=2, tiled=True) as d:
        d.write(arr, 1)
        if tags:
            d.update_tags(**{k: str(v) for k, v in tags.items()})


def run(dates, grids):
    for date in dates:
        gamma0, sigma0, inc, gcps, gcrs = slant_gamma0(date)
        for name in grids:
            res = GRIDS[name]
            g, tf = geocode(gamma0, gcps, gcrs, res)
            s, _ = geocode(sigma0, gcps, gcrs, res)
            # incidence is a smooth geometric field, so bilinear is right for it
            # -- averaging an angle field would be fine too but bilinear is exact
            # enough and cheaper
            i, _ = geocode(inc, gcps, gcrs, res, Resampling.bilinear)
            write(CACHE / f"gamma0_{name}_{date}.tif", g, tf, date=date, quantity="gamma0")
            write(CACHE / f"sigma0_{name}_{date}.tif", s, tf, date=date, quantity="sigma0")
            write(CACHE / f"incidence_{name}_{date}.tif", i, tf, date=date, quantity="incidence_deg")
            valid = np.isfinite(g)
            log("prep.geocode", date=date, grid=name, res=res,
                shape=f"{g.shape[0]}x{g.shape[1]}",
                valid_pct=round(float(valid.mean()) * 100, 2),
                gamma0_db_med=round(float(10 * np.log10(np.nanmedian(g[valid]))), 2))
        del gamma0, sigma0, inc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", action="store_true", help="Oct 13 + BASE grid only")
    a = ap.parse_args()
    dates = ["20251013"] if a.trial else DATES
    grids = ["base"] if a.trial else list(GRIDS)
    log("prep.start", dates=len(dates), grids=grids)
    run(dates, grids)
    log("prep.done")


if __name__ == "__main__":
    main()
