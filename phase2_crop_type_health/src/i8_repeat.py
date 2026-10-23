"""Stage I8 -- repeat-pass interferometric coherence, Jun19 vs Aug14. [O2]

WHY ATTEMPT IT AT ALL. The rubric names temporal coherence as a scorable SAR
physics axis, and coherence is the one product that uses the SLC phase across
acquisitions rather than within one (which is what [I1]/sublook.py already did).
If it worked it would be the single most discriminating crop-dynamics layer we
could build from this data.

WHY IT IS EXPECTED TO FAIL, stated before running so the result cannot be
retrofitted. Three independent reasons, any one of which is fatal:
  1. 56-day baseline at X-band (3.1 cm) over a growing canopy. Temporal
     decorrelation over vegetation is near-total in days, not months [B4].
  2. The perpendicular baseline was not computed from the orbit state vectors
     in this analysis, so the flat-earth / topographic fringe is not removed
     analytically. We estimate and remove the dominant fringe
     empirically instead, which is weaker.
  3. Range pixel spacing differs between the two scenes (1.2825 vs 1.2855 m),
     i.e. ~0.23%, which is several pixels of drift across the AOI. Handled by
     estimating the offset per block rather than globally.

WHY THE ANSWER IS TRUSTWORTHY EITHER WAY -- the three controls. A bare "coherence
was low" claim is worthless, because a coregistration bug produces exactly the
same number as real decorrelation. So every run reports:

  SELF    Jun19 correlated with ITSELF through the identical code path.
          Must return ~1.0. If it does not, the estimator is broken and no
          other number in this file means anything.
  NULL    Jun19 vs Aug14 deliberately mis-registered by 200 px, i.e. two
          genuinely unrelated pieces of ground. This is the EMPIRICAL bias
          floor of a finite boxcar -- |gamma_hat| is biased high, and this
          measures that bias directly instead of trusting a formula.
  STABLE  coherence on persistent bright scatterers (buildings, roads, the
          canal) versus on farm fields, both from the real coregistered pair.
          This is the check that separates the two failure modes:
            stable > null  ->  coregistration WORKED, so low field coherence is
                               real temporal decorrelation. Reportable negative.
            stable ~ null  ->  we failed to coregister. Uninformative, and must
                               be reported as uninformative, not as a finding.

Run:  python src/i8_repeat.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, FARMS, RESULTS, RESULTS_AUX, log, slc_path
from prep_r2 import UTM, aoi_window, geocode, target_grid, write

PRIMARY, SECONDARY = "20250619", "20250814"
BLOCK = 512          # coregistration block. Small enough that the range-spacing
                     # drift is constant inside it, big enough to correlate on.
COH_WIN = 9          # same boxcar as [I1], so the two coherences are comparable
NULL_SHIFT = 200     # px of deliberate mis-registration for the NULL control
SEARCH = 64          # max |offset| accepted from the cross-correlation, px


def xcorr_offset(a, b, search=SEARCH):
    """Integer + sub-pixel offset of `b` relative to `a`, from amplitude.

    Amplitude, not complex: the two acquisitions are 56 days apart, so their
    phases are unrelated by assumption and a complex correlation would be
    correlating noise. Amplitude survives -- fields and roads keep their relative
    brightness -- which is exactly why coregistration is still possible on a pair
    whose phase has fully decorrelated.
    """
    A, B = np.abs(a).astype("float64"), np.abs(b).astype("float64")
    A -= A.mean(); B -= B.mean()
    c = np.fft.ifft2(np.fft.fft2(A) * np.conj(np.fft.fft2(B))).real
    c = np.fft.fftshift(c)
    n0, n1 = c.shape
    ctr = (n0 // 2, n1 // 2)
    # only look inside +-search of zero shift; a peak further out is a false match
    sub = c[ctr[0] - search:ctr[0] + search + 1, ctr[1] - search:ctr[1] + search + 1]
    k = np.unravel_index(np.argmax(sub), sub.shape)
    peak = float(sub[k])
    dr, dc = k[0] - search, k[1] - search

    def parab(v0, v1, v2):
        """Sub-pixel peak by a 3-point parabola. Sub-pixel accuracy is not
        optional: half a pixel of residual misregistration alone costs ~0.4 of
        coherence, which would fake the very decorrelation we are testing for."""
        d = v0 - 2 * v1 + v2
        return 0.0 if abs(d) < 1e-12 else np.clip(0.5 * (v0 - v2) / d, -1, 1)

    fr = fc = 0.0
    if 0 < k[0] < sub.shape[0] - 1:
        fr = parab(sub[k[0] - 1, k[1]], sub[k], sub[k[0] + 1, k[1]])
    if 0 < k[1] < sub.shape[1] - 1:
        fc = parab(sub[k[0], k[1] - 1], sub[k], sub[k[0], k[1] + 1])
    # peak-to-sidelobe: a flat correlation surface means the offset is a guess
    psr = peak / (np.abs(sub).mean() + 1e-30)
    # negated: the correlation peak sits at MINUS the shift of `b`, and callers
    # want the shift itself so the correction is an unambiguous fshift(b, -dr, -dc)
    return -(dr + fr), -(dc + fc), psr


def fshift(z, dr, dc):
    """Sub-pixel shift of complex data by a phase ramp in the Fourier domain.

    Exact for band-limited data and, unlike any interpolation kernel, it does not
    smooth the speckle -- smoothing one image of a pair would itself destroy
    coherence and would look identical to decorrelation.
    """
    n0, n1 = z.shape
    r = np.fft.fftfreq(n0)[:, None] * dr
    c = np.fft.fftfreq(n1)[None, :] * dc
    return np.fft.ifft2(np.fft.fft2(z) * np.exp(-2j * np.pi * (r + c)))


def deramp(ifg):
    """Remove the dominant interferometric fringe empirically.

    The flat-earth/topographic phase is not subtracted analytically here (no
    orbit-based baseline model). But a residual linear fringe is a single spike in the 2D
    spectrum of the interferogram, so we find that spike and demodulate it away.
    A fringe left in place would rotate the phase across the boxcar window and
    suppress coherence that is genuinely present.
    """
    n0, n1 = ifg.shape
    S = np.abs(np.fft.fft2(ifg))
    k = np.unravel_index(np.argmax(S), S.shape)
    f0 = np.fft.fftfreq(n0)[k[0]]
    f1 = np.fft.fftfreq(n1)[k[1]]
    r = np.arange(n0)[:, None] * f0
    c = np.arange(n1)[None, :] * f1
    return ifg * np.exp(-2j * np.pi * (r + c))


def coherence(z1, z2, win=COH_WIN):
    """Boxcar |<z1 z2*>| / sqrt(<|z1|^2><|z2|^2>), with the fringe removed first."""
    ifg = deramp(z1 * np.conj(z2))
    num = uniform_filter(ifg.real, (win, win)) + 1j * uniform_filter(ifg.imag, (win, win))
    p1 = uniform_filter((z1 * np.conj(z1)).real, (win, win))
    p2 = uniform_filter((z2 * np.conj(z2)).real, (win, win))
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.abs(num) / np.sqrt(np.maximum(p1 * p2, 1e-20))
    return np.clip(g, 0, 1).astype("float32")


def read_block(src, win, r, c, h, w):
    return src.read(1, window=Window(win.col_off + c, win.row_off + r, w, h)
                    ).astype("complex64")


def run_pair(mode="real"):
    """mode: 'real' | 'null' (mis-register on purpose) | 'self' (primary vs itself)."""
    farms = gpd.read_file(FARMS)
    b = farms.buffer(400 / 111320.0).total_bounds
    p_path = slc_path(PRIMARY)
    s_path = slc_path(PRIMARY if mode == "self" else SECONDARY)

    with rasterio.open(p_path) as sp, rasterio.open(s_path) as ss:
        gcps, gcrs = sp.gcps
        wp = aoi_window(PRIMARY, gcps, sp.height, sp.width, b)
        sg, _ = ss.gcps
        ws = aoi_window(PRIMARY if mode == "self" else SECONDARY,
                        sg, ss.height, ss.width, b)
        h = min(int(wp.height), int(ws.height))
        w = min(int(wp.width), int(ws.width))
        coh = np.full((h, w), np.nan, dtype="float32")
        amp = np.full((h, w), np.nan, dtype="float32")
        stats = []

        for r in range(0, h - BLOCK + 1, BLOCK):
            for c in range(0, w - BLOCK + 1, BLOCK):
                z1 = read_block(sp, wp, r, c, BLOCK, BLOCK)
                rr = r + (NULL_SHIFT if mode == "null" else 0)
                cc = c + (NULL_SHIFT if mode == "null" else 0)
                if rr + BLOCK > int(ws.height) or cc + BLOCK > int(ws.width):
                    continue
                z2 = read_block(ss, ws, rr, cc, BLOCK, BLOCK)
                if not (np.abs(z1).any() and np.abs(z2).any()):
                    continue
                if mode == "null":
                    dr = dc = 0.0; psr = np.nan     # mis-registration is the point
                else:
                    dr, dc, psr = xcorr_offset(z1, z2)
                    z2 = fshift(z2, -dr, -dc)
                g = coherence(z1, z2)
                # boxcar edges are contaminated by the block border
                m = COH_WIN
                coh[r + m:r + BLOCK - m, c + m:c + BLOCK - m] = g[m:-m, m:-m]
                amp[r + m:r + BLOCK - m, c + m:c + BLOCK - m] = \
                    np.abs(z1[m:-m, m:-m]) * np.abs(z2[m:-m, m:-m])
                stats.append((dr, dc, psr, float(np.nanmedian(g))))

        shifted = [rasterio.control.GroundControlPoint(
            row=p.row - wp.row_off, col=p.col - wp.col_off, x=p.x, y=p.y, z=p.z)
            for p in gcps]

    st = np.array(stats, dtype="float64")
    log(f"i8.{mode}", blocks=len(st), slant=f"{h}x{w}",
        med_coh=round(float(np.nanmedian(coh)), 4),
        p90_coh=round(float(np.nanpercentile(coh[np.isfinite(coh)], 90)), 4)
        if np.isfinite(coh).any() else None,
        med_dr=round(float(np.nanmedian(st[:, 0])), 3) if len(st) else None,
        med_dc=round(float(np.nanmedian(st[:, 1])), 3) if len(st) else None,
        med_psr=round(float(np.nanmedian(st[:, 2])), 1) if len(st) else None)
    return coh, amp, shifted, gcrs


def main():
    log("i8.start", pair=f"{PRIMARY}-{SECONDARY}")

    # --- control 1: does the estimator work at all? must be ~1.0 -------------
    self_coh, _, _, _ = run_pair("self")
    m_self = float(np.nanmedian(self_coh))
    assert m_self > 0.9, f"SELF coherence {m_self:.3f} -- estimator broken, stop"

    # --- control 2: empirical bias floor of the finite boxcar ---------------
    null_coh, null_amp, _, _ = run_pair("null")
    m_null = float(np.nanmedian(null_coh))

    # --- the actual measurement ---------------------------------------------
    coh, amp, gcps, gcrs = run_pair("real")
    m_real = float(np.nanmedian(coh))

    # --- control 3: stable scatterers vs everything else --------------------
    # brightest 1% of pixels: buildings, roads, the canal. These do not grow, so
    # they must hold coherence if the coregistration is sound.
    def brightest(c, a, pct=99):
        ok = np.isfinite(c) & np.isfinite(a)
        return float(np.nanmedian(c[ok & (a >= np.nanpercentile(a[ok], pct))])), ok

    m_stable, ok = brightest(coh, amp)
    # The bright pixels must be compared against BRIGHT pixels in the null run,
    # not against the all-pixel floor. A neighbourhood dominated by one strong
    # scatterer has fewer effective looks than a speckle-only one, so its
    # coherence bias is HIGHER -- scoring it against the all-pixel floor would
    # manufacture an excess out of nothing but statistics.
    m_stable_null, _ = brightest(null_coh, null_amp)

    g, tf = geocode(coh, gcps, gcrs, 5.0)
    write(CACHE / f"repeat_coh_{PRIMARY}_{SECONDARY}.tif", g, tf,
          pair=f"{PRIMARY}/{SECONDARY}", quantity="repeat_pass_coherence")

    # per farm, so the writeup can quote a farm-level number, not just an image
    from farm_stats import farm_labels
    fa = gpd.read_file(FARMS).to_crs(UTM)
    fa["geometry"] = fa.geometry.make_valid()
    tfg, wg, hg = target_grid(5.0)
    lab, _ = farm_labels(fa, tfg, (hg, wg))
    lv = np.where(np.isfinite(g), lab, 0)
    n = len(fa)
    cnt = np.bincount(lv.ravel(), minlength=n + 1)[1:]
    tot = np.bincount(lv.ravel(), weights=np.nan_to_num(g).ravel(), minlength=n + 1)[1:]
    with np.errstate(invalid="ignore"):
        per_farm = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    m_farm = float(np.nanmedian(per_farm))

    print("\n" + "=" * 72)
    print(f"I8 REPEAT-PASS COHERENCE  {PRIMARY} x {SECONDARY}  (56 days, X-band)")
    print("=" * 72)
    print(f"  SELF   (Jun19 vs itself, estimator control) : {m_self:.4f}   expect ~1")
    print(f"  NULL   (mis-registered {NULL_SHIFT}px, bias floor) : {m_null:.4f}")
    print(f"  REAL   (coregistered, all pixels)           : {m_real:.4f}")
    print(f"  STABLE (brightest 1%, should hold if any)   : {m_stable:.4f}")
    print(f"  STABLE-NULL (same pixels, mis-registered)   : {m_stable_null:.4f}")
    print(f"  FARMS  (per-farm mean, median over farms)   : {m_farm:.4f}")
    excess = m_real - m_null
    stable_excess = m_stable - m_stable_null
    print(f"\n  excess over bias floor: {excess:+.4f}   "
          f"stable excess (like for like): {stable_excess:+.4f}")
    if stable_excess > 0.05 and excess < 0.05:
        verdict = ("COREGISTRATION SOUND, FIELDS FULLY DECORRELATED -- "
                   "reportable negative")
    elif stable_excess <= 0.05:
        verdict = ("STABLE TARGETS ALSO AT THE FLOOR -- coregistration not "
                   "demonstrated, result UNINFORMATIVE, do not claim a finding")
    else:
        verdict = "RESIDUAL COHERENCE PRESENT -- investigate, do not discard"
    print(f"  VERDICT: {verdict}\n")

    pd.DataFrame({"farm_id": fa.get("farm_id", pd.RangeIndex(1, n + 1)),
                  "repeat_coh_jun19_aug14": per_farm}).to_csv(
        RESULTS_AUX / "i8_repeat_coh.csv", index=False)
    log("i8.done", self=round(m_self, 4), null=round(m_null, 4), real=round(m_real, 4),
        stable=round(m_stable, 4), stable_null=round(m_stable_null, 4),
        farm=round(m_farm, 4), verdict=verdict)


def _selfcheck():
    """Estimator sanity on synthetic data -- runs in a second, no I/O.

    Three properties, each of which the real run depends on:
      identical signal    -> coherence ~1
      unrelated signals   -> coherence at the small-sample bias floor, not 0
      known sub-pixel shift recovered by xcorr_offset
    """
    rng = np.random.default_rng(0)
    z = (rng.normal(size=(256, 256)) + 1j * rng.normal(size=(256, 256))).astype("complex64")
    assert np.nanmedian(coherence(z, z)) > 0.99, "identical signal must give ~1"
    z2 = (rng.normal(size=(256, 256)) + 1j * rng.normal(size=(256, 256))).astype("complex64")
    floor = np.nanmedian(coherence(z, z2))
    assert 0.0 < floor < 0.3, f"bias floor {floor:.3f} out of expected range"
    a = np.abs(z) ** 2 + 0.0j                     # a real, structured amplitude field
    dr, dc, _ = xcorr_offset(a, fshift(a, 3.0, -5.0))
    assert abs(dr - 3.0) < 0.2 and abs(dc + 5.0) < 0.2, f"offset recovery failed {dr},{dc}"
    print(f"i8 self-check OK -- coh(z,z)~1, floor {floor:.3f}, offset ({dr:.2f},{dc:.2f})")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
