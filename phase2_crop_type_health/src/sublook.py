"""Stage I2b -- sub-look (zero temporal baseline) coherence. [D1]

Why this exists. The rubric names "temporal coherence" as a scorable SAR physics
axis, but our shortest repeat-pass baseline is 56 days at X-band, where the
literature expects near-total decorrelation over growing crops [B4]. Rather than
report only that failure, we exploit the fact that the SLC is COMPLEX: a single
acquisition can be split along its azimuth Doppler spectrum into two sub-apertures,
which view the scene from slightly different squint angles at the SAME instant.

Correlating those two sub-looks gives a coherence with ZERO temporal baseline, so
it cannot decorrelate through time at all. What it measures instead is angular /
structural stability:

    high sub-look coherence -> a few dominant, deterministic scatterers
                               (bare soil, harvested stubble, built-up, roads)
    low  sub-look coherence -> many randomly-placed scatterers whose interference
                               changes with viewing angle, i.e. a VOLUME -- a canopy

So it reads as a canopy-structure indicator, available on all four dates, and it
uses the phase that every amplitude-only approach discards [C3-T5].

Method per date:
  1. FFT along azimuth (axis 0 -- rows are azimuth, delta_line_time per row)
  2. find the occupied Doppler band from the mean power spectrum
  3. take the lower and upper half of that band as two sub-looks
  4. demodulate each to baseband, so the sub-look centre-frequency offset does not
     appear as a phase ramp across the estimation window
  5. IFFT, then boxcar coherence  |<s1 s2*>| / sqrt(<|s1|^2><|s2|^2>)
  6. geocode with the same GCPs as I1 and average per farm

Run:  python src/sublook.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, DATES, FARMS, RESULTS, log, slc_path
from prep_r2 import (UTM, aoi_window, geocode, target_grid, write)

import geopandas as gpd
from rasterio.windows import Window

COH_WIN = 9          # boxcar for the coherence estimate, in slant-range pixels
GAP_FRAC = 0.10      # gap between the two sub-bands, as a fraction of the band
COL_CHUNK = 512


def occupied_band(spec_profile, thresh=0.25):
    """Doppler bins carrying real signal, from the mean azimuth power spectrum.

    The processed bandwidth does not fill the PRF, so splitting the FULL spectrum
    in half would put most of one sub-look in noise. Threshold relative to peak
    instead and split only the occupied part.
    """
    p = spec_profile / spec_profile.max()
    idx = np.where(p >= thresh)[0]
    # spectrum is fftshifted, so the occupied band is contiguous about the centre
    return int(idx.min()), int(idx.max())


def sublook_coherence(z, band_frac, win=COH_WIN):
    """Two-sub-aperture coherence for one complex block. Azimuth is axis 0.

    `band_frac` is passed in rather than re-detected here. Detecting the occupied
    band per column chunk made the sub-look SEPARATION vary from chunk to chunk,
    and since coherence depends directly on that separation the result carried
    block artefacts -- which showed up as Moran's I flipping between +0.28 and
    -0.01 across dates. One band per date, applied to every chunk. [I1]
    """
    n = z.shape[0]
    Z = np.fft.fftshift(np.fft.fft(z, axis=0), axes=0)
    band = int(round(band_frac * n))
    lo = n // 2 - band // 2
    hi = lo + band
    gap = int(band * GAP_FRAC)
    half = (band - gap) // 2
    if half < 32:
        return np.full(z.shape, np.nan, dtype="float32")

    def look(a, b):
        S = np.zeros_like(Z)
        S[a:b] = Z[a:b]
        # demodulate: move this sub-band's centre to zero frequency, otherwise the
        # two looks differ by a linear phase ramp and the boxcar average below
        # would destroy coherence that is really there
        S = np.roll(S, -((a + b) // 2 - n // 2), axis=0)
        return np.fft.ifft(np.fft.ifftshift(S, axes=0), axis=0)

    s1 = look(lo, lo + half)
    s2 = look(hi - half, hi)

    num = uniform_filter((s1 * np.conj(s2)).real, (win, win)) + \
        1j * uniform_filter((s1 * np.conj(s2)).imag, (win, win))
    p1 = uniform_filter((s1 * np.conj(s1)).real, (win, win))
    p2 = uniform_filter((s2 * np.conj(s2)).real, (win, win))
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.abs(num) / np.sqrt(np.maximum(p1 * p2, 1e-20))
    return np.clip(g, 0, 1).astype("float32")


def run_date(date):
    farms = gpd.read_file(FARMS)
    b = farms.buffer(400 / 111320.0).total_bounds
    with rasterio.open(slc_path(date)) as s:
        gcps, gcrs = s.gcps
        win = aoi_window(date, gcps, s.height, s.width, b)
        h, w = int(win.height), int(win.width)
        # one band estimate for the whole date, from a representative central strip
        probe = s.read(1, window=Window(win.col_off + w // 2 - 128, win.row_off, 256,
                                        min(4096, h))).astype("complex64")
        P = np.fft.fftshift(np.fft.fft(probe, axis=0), axes=0)
        lo, hi = occupied_band(np.abs(P).mean(axis=1) ** 2)
        band_frac = (hi - lo) / P.shape[0]
        del probe, P

        coh = np.empty((h, w), dtype="float32")
        for c in range(0, w, COL_CHUNK):
            nc = min(COL_CHUNK, w - c)
            z = s.read(1, window=Window(win.col_off + c, win.row_off, nc, h)
                       ).astype("complex64")
            coh[:, c:c + nc] = sublook_coherence(z, band_frac)
        shifted = [rasterio.control.GroundControlPoint(
            row=p.row - win.row_off, col=p.col - win.col_off, x=p.x, y=p.y, z=p.z)
            for p in gcps]
    g, tf = geocode(coh, shifted, gcrs, 5.0)
    write(CACHE / f"sublook_coh_base_{date}.tif", g, tf, date=date, quantity="sublook_coherence")
    log("sublook.date", date=date, slant=f"{h}x{w}", band_frac=round(band_frac,4),
        med_slant=round(float(np.nanmedian(coh)), 4),
        med_geo=round(float(np.nanmedian(g)), 4))
    return g


def main():
    log("sublook.start")
    for d in DATES:
        run_date(d)

    # attach to the farm table
    import rasterio as rio
    from rasterio.features import rasterize
    from scipy import ndimage as ndi
    from farm_stats import farm_labels

    farms = gpd.read_file(FARMS).to_crs(UTM)
    farms["geometry"] = farms.geometry.make_valid()
    tf, w, h = target_grid(5.0)
    lab, _ = farm_labels(farms, tf, (h, w))
    n = len(farms)
    f = pd.read_csv(RESULTS / "farm_features.csv")
    for d in DATES:
        with rio.open(CACHE / f"sublook_coh_base_{d}.tif") as s:
            a = s.read(1)
        lv = np.where(np.isfinite(a), lab, 0)
        cnt = np.bincount(lv.ravel(), minlength=n + 1)[1:]
        tot = np.bincount(lv.ravel(), weights=np.nan_to_num(a).ravel(), minlength=n + 1)[1:]
        with np.errstate(invalid="ignore"):
            f[f"subcoh_{d}"] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
        log("sublook.farm", date=d, n=int((cnt > 0).sum()),
            med=round(float(np.nanmedian(f[f"subcoh_{d}"])), 4))
    sc = f[[f"subcoh_{d}" for d in DATES]].values
    f["subcoh_drop_jun_aug"] = np.nanmean(sc[:, :2], axis=1) - sc[:, 2]
    f.to_csv(RESULTS / "farm_features.csv", index=False)
    log("sublook.done", cols=f.shape[1])
    print(f[[f"subcoh_{d}" for d in DATES] + ["subcoh_drop_jun_aug"]].describe().round(4).to_string())


if __name__ == "__main__":
    main()
