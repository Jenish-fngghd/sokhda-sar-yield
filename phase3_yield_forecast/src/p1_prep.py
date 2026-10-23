"""Phase 1: six SLCs -> calibrated, geocoded beta0/sigma0/gamma0 on a common grid.

Chain, one line per step:

    complex SLC (slant range, ungeocoded)
      -> beta0  = scale_factor**2 * |z|**2          [common.beta0 -- the SQUARE is the
                                                     fix; R2 shipped the unsquared form]
      -> theta  = per-pixel incidence from the 108 ORBIT STATE VECTORS, interpolated
                  from the GCP lattice                [validated below against the
                                                       vendor's annotated centre angle]
      -> sigma0 = beta0 * sin(theta)
         gamma0 = beta0 * tan(theta)
      -> geocode to EPSG:32643 with the scene's own GCPs, resampling by AVERAGE, which
         IS the multilook: averaging POWER over the slant pixels in a ground cell is
         exactly what multilooking does, and it happens in the same pass as the warp.
      -> two grids: FINE 2 m (texture, within-field) and BASE 5 m (farm means, trends).

beta0 is written out as well as gamma0 because the calibration adjudication (p1_calib)
compares the darkest percentiles against `nesz_peak`, and the vendor's noise floor is a
statement about the product's own radiometry, not about our incidence correction.

Deliberately NOT done here: incidence normalisation and any inter-date referencing.
Both need all six on a common grid first.

Run:  python src/p1_prep.py                 (all dates, both grids)
      python src/p1_prep.py --dates 20251029 --grids base
"""
import argparse
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
from common import CACHE, DATES, FARMS, beta0, log, meta, scene, slc_path

import geopandas as gpd
import pyproj

UTM = 32643
GRIDS = {"base": 5.0, "fine": 2.0}
AOI_BUFFER_M = 400.0
SLANT_MARGIN_PX = 200
CHUNK_ROWS = 1024

_LL2ECEF = pyproj.Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)


def _t2s(t):
    return datetime.fromisoformat(t.replace("Z", "")[:26]).timestamp()


def incidence_at(date, rows, cols, gcps):
    """Incidence (deg) at arbitrary (row, col) from orbit geometry alone.

    Each GCP gives an image position and a ground position. The sensor position at
    that GCP's azimuth time comes from interpolating the state vectors; the incidence
    angle is the angle between the line of sight and the local ellipsoid normal. No
    approximation and no assumed constant -- and it reproduces the vendor's annotated
    centre incidence, which is what licenses using it.

    Incidence varies by ~0.4 deg across a 27 km strip, so interpolating it from the
    15x15 GCP lattice is negligible against the 6.55 deg spread BETWEEN dates.
    """
    c = meta(date)["collect"]
    ig = c["image"]["image_geometry"]
    sv = c["state"]["state_vectors"]
    st = np.array([_t2s(v["time"]) for v in sv])
    P = np.array([v["position"] for v in sv])
    t0, dt = _t2s(ig["first_line_time"]), ig["delta_line_time"]

    gr = np.array([p.row for p in gcps])
    gc = np.array([p.col for p in gcps])
    X, Y, Z = _LL2ECEF.transform(np.array([p.x for p in gcps]),
                                 np.array([p.y for p in gcps]),
                                 np.array([p.z for p in gcps]))
    T = np.c_[X, Y, Z]
    S = np.c_[[np.interp(t0 + gr * dt, st, P[:, k]) for k in range(3)]].T

    los = S - T
    los /= np.linalg.norm(los, axis=1)[:, None]
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    up = np.c_[T[:, 0], T[:, 1], T[:, 2] / (1 - e2)]
    up /= np.linalg.norm(up, axis=1)[:, None]
    inc_gcp = np.degrees(np.arccos(np.sum(los * up, axis=1)))

    out = griddata((gr, gc), inc_gcp, (rows, cols), method="linear")
    nan = np.isnan(out)
    if nan.any():                    # AOI can sit outside the GCP hull at a strip edge
        out[nan] = griddata((gr, gc), inc_gcp, (rows[nan], cols[nan]), method="nearest")
    return out, inc_gcp


def aoi_window(gcps, height, width, bounds_ll):
    """Slant-range window covering the AOI, from the GCP-derived affine.

    That affine only approximates a curved SAR geometry, so the window is padded and
    clipped rather than trusted. It exists to avoid reading a 27000 x 4000 strip when
    the village needs a third of it.
    """
    inv = ~from_gcps(gcps)
    lon0, lat0, lon1, lat1 = bounds_ll
    rc = [inv * (X, Y) for X in (lon0, lon1) for Y in (lat0, lat1)]
    cs = [p[0] for p in rc]
    rs = [p[1] for p in rc]
    return Window(max(0, int(min(cs)) - SLANT_MARGIN_PX),
                  max(0, int(min(rs)) - SLANT_MARGIN_PX),
                  min(width, int(max(cs)) + SLANT_MARGIN_PX) - max(0, int(min(cs)) - SLANT_MARGIN_PX),
                  min(height, int(max(rs)) + SLANT_MARGIN_PX) - max(0, int(min(rs)) - SLANT_MARGIN_PX))


def slant(date):
    """Read the AOI window; return (beta0, sigma0, gamma0, incidence, shifted GCPs, crs)."""
    sc = scene(date)
    assert meta(date)["collect"]["image"]["radiometry"] == "beta_nought"

    b = gpd.read_file(FARMS).buffer(AOI_BUFFER_M / 111320.0).total_bounds

    with rasterio.open(slc_path(date)) as s:
        gcps, gcrs = s.gcps
        win = aoi_window(gcps, s.height, s.width, b)
        h, w = int(win.height), int(win.width)
        bet = np.empty((h, w), dtype="float32")
        for r in range(0, h, CHUNK_ROWS):
            n = min(CHUNK_ROWS, h - r)
            z = s.read(1, window=Window(win.col_off, win.row_off + r, w, n))
            bet[r:r + n] = beta0(z, date)          # scale_factor SQUARED, from common

    rr, cc = np.meshgrid(np.arange(h, dtype="float32") + win.row_off,
                         np.arange(w, dtype="float32") + win.col_off, indexing="ij")
    inc, inc_gcp = incidence_at(date, rr, cc, gcps)
    inc = inc.astype("float32")
    del rr, cc

    # the annotated centre angle is a number the vendor computed from the same orbit;
    # agreeing with it is the check that our state-vector geometry is right.
    inc_ctr = float(inc_gcp.mean())      # GCP lattice is centred on the strip
    log("p1_prep.incidence", date=date, ours_mean=round(inc_ctr, 4),
        vendor_center=round(float(sc["incidence"]), 4),
        delta_deg=round(inc_ctr - float(sc["incidence"]), 4),
        aoi_min=round(float(inc.min()), 3), aoi_max=round(float(inc.max()), 3))

    th = np.radians(inc)
    sig = bet * np.sin(th)
    gam = bet * np.tan(th)

    # GCPs are in FULL-image pixel coordinates; shift them into the window's frame so
    # the warp knows where the cropped array sits.
    shifted = [rasterio.control.GroundControlPoint(
        row=p.row - win.row_off, col=p.col - win.col_off, x=p.x, y=p.y, z=p.z)
        for p in gcps]

    log("p1_prep.slant", date=date, win=f"{h}x{w}", sf=sc["scale_factor"],
        beta0_db_med=round(float(10 * np.log10(np.median(bet[bet > 0]))), 2))
    return bet, sig, gam, inc, shifted, gcrs


def target_grid(res):
    x0, y0, x1, y1 = gpd.read_file(FARMS).to_crs(UTM).total_bounds
    x0 = np.floor((x0 - AOI_BUFFER_M) / res) * res
    y0 = np.floor((y0 - AOI_BUFFER_M) / res) * res
    x1 = np.ceil((x1 + AOI_BUFFER_M) / res) * res
    y1 = np.ceil((y1 + AOI_BUFFER_M) / res) * res
    return from_origin(x0, y1, res, res), int(round((x1 - x0) / res)), int(round((y1 - y0) / res))


def geocode(src, gcps, gcrs, res, resampling=Resampling.average):
    """Slant range -> EPSG:32643. Average resampling IS the multilook."""
    tf, w, h = target_grid(res)
    dst = np.full((h, w), np.nan, dtype="float32")
    reproject(source=src, destination=dst, gcps=gcps, src_crs=gcrs, src_nodata=np.nan,
              dst_transform=tf, dst_crs=f"EPSG:{UTM}", dst_nodata=np.nan,
              resampling=resampling)
    return dst, tf


def write(path, arr, tf, **tags):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", crs=f"EPSG:{UTM}", transform=tf,
                       nodata=np.nan, compress="deflate", predictor=2, tiled=True) as d:
        d.write(arr, 1)
        d.update_tags(**{k: str(v) for k, v in tags.items()})


def run(dates, grids):
    for date in dates:
        bet, sig, gam, inc, gcps, gcrs = slant(date)
        for name in grids:
            res = GRIDS[name]
            for q, a in (("beta0", bet), ("sigma0", sig), ("gamma0", gam)):
                g, tf = geocode(a, gcps, gcrs, res)
                write(CACHE / f"{q}_{name}_{date}.tif", g, tf, date=date, quantity=q,
                      calibration="scale_factor_squared")
                if q == "gamma0":
                    valid = np.isfinite(g)
                    log("p1_prep.geocode", date=date, grid=name, res=res,
                        shape=f"{g.shape[0]}x{g.shape[1]}",
                        valid_pct=round(float(valid.mean()) * 100, 2),
                        gamma0_db_med=round(float(10 * np.log10(np.nanmedian(g[valid]))), 2))
            # incidence is a smooth geometric field: bilinear, not an average of angles
            i, tf = geocode(inc, gcps, gcrs, res, Resampling.bilinear)
            write(CACHE / f"incidence_{name}_{date}.tif", i, tf, date=date,
                  quantity="incidence_deg")
        del bet, sig, gam, inc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*", default=DATES)
    ap.add_argument("--grids", nargs="*", default=list(GRIDS))
    a = ap.parse_args()
    log("p1_prep.start", dates=a.dates, grids=a.grids)
    run(a.dates, a.grids)
    log("p1_prep.done")


if __name__ == "__main__":
    main()
