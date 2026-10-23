"""T25 / P3-23: how much of a farm's index is its crop, and how much is its BOUNDARY?

Nine post-freeze approaches have interrogated the ranking's inputs -- which dates (T23), which
neighbourhood (T22), which noise realisation (T24). None of them asked what a farm's pixels
actually are. That is the gap this closes, and it lands squarely on the part of the index we
have so far been unable to test.

WHY IT BITES HERE. These parcels are tiny -- median 69 pixels on the peak date. A parcel of 69
pixels is roughly 8x8, so a one-pixel border is HALF of it. Every boundary pixel is a mixture
of the farm, the bund, the track, and the neighbour's crop, and geocoding spreads the mixture
further. Two of the five index parts are computed from exactly the statistic mixture corrupts:

    level    = mean gamma0 at peak canopy       -> pulled toward the neighbourhood mean
    uniform  = -within-farm CV at peak canopy   -> INFLATED by a mixed border, which reads
               as a non-uniform stand, which reads as an unhealthy field

`uniform` carries the LARGEST weight in the shipped index, 0.249. T24 could not perturb it --
a speckle model for a texture statistic needs a correlation length we do not have -- and said
so, which left the largest single component of the deliverable untested. This test reaches it
by geometry instead of by noise model, so it is the follow-up T24's own honest bound demanded.

The pipeline already knows this is a hazard: p1_features rasterises through a negative-buffer
ladder [-5, -2, 0] m and records which rung each farm landed on. But the bottom rung is
rasterised with all_touched=True -- every pixel the polygon so much as clips is counted -- and
no test has ever asked what that costs, or whether the farms on that rung are the same farms we
put on the attention list.

METHOD. One rasterisation, the SHIPPED one, split into two disjoint pixel sets per farm:

    core   pixels whose four neighbours all belong to the same farm (interior)
    ring   every other pixel of that farm (the one-pixel boundary shell)

and the index rebuilt from core pixels alone by the SHIPPED p3_build.health_parts. No farm is
dropped: a farm with no interior at all keeps its shipped values and is counted and reported.

CONTROLS:

  C1  KNOWN ANSWER. core + ring is the shipped pixel set by construction, so recomputing the
      features over their union must reproduce p1_farm_features.csv to floating point and the
      index to rho = 1.0000. If it does not, the harness is not measuring the shipped product
      and nothing below means anything. Runs first, aborts on failure.

  C2  POSITIVE CONTROL, KNOWN ANSWER. Multiply the ring pixels of the peak scene by a known
      +3.000 dB and re-extract. The recovered ring-minus-core difference must move by exactly
      +3.000 dB. This fails if the two masks overlap, leak, or address the wrong farms.

  C3  THE CONTROL THE CONCLUSION RESTS ON -- a MATCHED-N RANDOM SPLIT. Core has fewer pixels
      than the full parcel, and fewer pixels alone changes a mean and biases a CV. So the ring
      effect is measured against a null that discards exactly the same number of pixels from
      each farm AT RANDOM. If a random split moves the index as much as the boundary split
      does, the effect is sample size and not geometry, and this test is dead. It is the only
      control here that can kill the finding, and it is the reason the finding is worth
      anything.

  C4  Invariance. The village total is anchor x area; asserted per crop on every regime.

Writes results/tables/p3_edge_farm.csv, p3_edge_stats.csv.

Run:  py -3.12 -u src/p3_edge.py
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

os.environ.pop("PROJ_LIB", None)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CROPS, TABLES, log
from p1_features import UTM, farm_labels, labelled_stats, load_grid
from p3_build import LEVEL_DATES, derive_weights, health_index, health_parts, modifier

import geopandas as gpd
from common import FARMS

sys.stdout.reconfigure(encoding="utf8")
warnings.filterwarnings("ignore", category=RuntimeWarning)

RNG = np.random.default_rng(20260901)
DECILE = 0.10
PEAK = "20250814"
INJECT_DB = 3.0          # C2's known answer


def core_mask(lab):
    """Interior pixels: those whose four neighbours belong to the SAME farm.

    Done on the label image rather than by re-buffering the polygons, so core and ring are
    guaranteed to partition exactly the pixel set the shipped extraction used -- which is what
    makes C1 a real check and not a re-derivation. A farm one pixel wide has no interior; that
    is a fact about the parcel, not a failure, and it is counted rather than patched.
    """
    m = lab > 0
    c = m.copy()
    c[1:, :] &= lab[1:, :] == lab[:-1, :]
    c[:-1, :] &= lab[:-1, :] == lab[1:, :]
    c[:, 1:] &= lab[:, 1:] == lab[:, :-1]
    c[:, :-1] &= lab[:, :-1] == lab[:, 1:]
    c[0, :] = c[-1, :] = c[:, 0] = c[:, -1] = False      # image edge is not interior
    return c


def matched_split(lab, n, keep_cnt, rng):
    """Discard the same NUMBER of pixels per farm as the ring does, but at random.

    Without this the whole test is confounded: core has fewer pixels than the parcel, and a
    smaller sample moves a mean and biases a CV all on its own. The null keeps exactly
    keep_cnt[i] pixels of farm i, chosen uniformly, and its complement stands in for the ring.
    """
    idx = np.flatnonzero(lab.ravel() > 0)
    fl = lab.ravel()[idx] - 1
    order = np.lexsort((rng.random(idx.size), fl))
    idx, fl = idx[order], fl[order]
    starts = np.searchsorted(fl, np.arange(n))
    rank = np.arange(idx.size) - starts[fl]
    keep = rank < keep_cnt[fl]
    a = np.zeros(lab.size, dtype=lab.dtype)
    b = np.zeros(lab.size, dtype=lab.dtype)
    a[idx[keep]] = fl[keep] + 1
    b[idx[~keep]] = fl[~keep] + 1
    return a.reshape(lab.shape), b.reshape(lab.shape)


def features(G, mask_lab, n, base):
    """The feature columns the shipped index consumes, extracted over an arbitrary label set.

    Farms left with no pixels keep their SHIPPED values -- no farm is ever dropped, per the
    standing rule -- and the caller is handed the mask of which ones those were, so the
    substitution is reported instead of hidden.
    """
    out = base.copy()
    empty = np.zeros(n, dtype=bool)
    for d in LEVEL_DATES:
        mean, _, std, cnt = labelled_stats(G["gamma0"][d], mask_lab, n)
        with np.errstate(invalid="ignore", divide="ignore"):
            cv = std / np.where(mean > 0, mean, np.nan)
        ok = cnt > 1
        empty |= ~ok
        out.loc[ok, f"g0_lin_{d}"] = mean[ok]
        out.loc[ok, f"g0_db_{d}"] = 10.0 * np.log10(np.where(mean[ok] > 0, mean[ok], np.nan))
        out.loc[ok, f"cv_{d}"] = cv[ok]
        out.loc[ok, f"npix_{d}"] = cnt[ok]
    return out, empty


def index_of(f, crop):
    parts = health_parts(f, use_29oct=False)
    return health_index(parts, derive_weights(parts), crop)


def bottom(idx, k):
    return set(np.argsort(idx)[:k])


def check_invariance(idx, crop, area, tag):
    m = modifier(idx, crop, area)
    for c in CROPS:
        msk = (crop == c).to_numpy()
        if msk.sum() >= 3:
            assert abs(np.average(m[msk], weights=area[msk]) - 1.0) < 1e-9, tag


def main():
    shipped = pd.read_csv(TABLES / "p1_farm_features.csv")
    farms = gpd.read_file(FARMS).to_crs(UTM)
    farms["geometry"] = farms.geometry.make_valid()
    n = len(farms)
    assert n == len(shipped) == 966, (n, len(shipped))

    G, tf, shape = load_grid("base", quantities=("gamma0",))
    lab, level = farm_labels(farms, tf, shape)
    core = core_mask(lab)
    lab_core = np.where(core, lab, 0).astype(lab.dtype)
    lab_ring = np.where(core, 0, lab).astype(lab.dtype)

    crop = shipped_crop(shipped)
    area = shipped.area_ha.to_numpy(dtype="float64")
    k = max(1, int(round(DECILE * n)))

    full_cnt = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    core_cnt = np.bincount(lab_core.ravel(), minlength=n + 1)[1:]
    ring_frac = np.where(full_cnt > 0, 1.0 - core_cnt / np.maximum(full_cnt, 1), np.nan)
    no_core = core_cnt == 0

    print(f"{n} farms | attention list = bottom {k} | median parcel {np.median(full_cnt):.0f} px")
    print(f"ring fraction: median {np.nanmedian(ring_frac):.3f}  "
          f"p90 {np.nanpercentile(ring_frac, 90):.3f}  "
          f"farms with NO interior at all: {int(no_core.sum())}")
    print(f"buffer ladder rungs (-5/-2/0 m): "
          + " ".join(f"{int((level == i).sum())}" for i in range(3))
          + f"   |  on the all_touched rung: {int((level == 2).sum())}\n")

    # ---------------- C1: core + ring must BE the shipped extraction ----------------
    ref = index_of(shipped, crop)
    f_full, _ = features(G, lab, n, shipped)
    dmax = max(float(np.nanmax(np.abs(f_full[f"g0_db_{d}"] - shipped[f"g0_db_{d}"])))
               for d in LEVEL_DATES)
    cmax = max(float(np.nanmax(np.abs(f_full[f"cv_{d}"] - shipped[f"cv_{d}"])))
               for d in LEVEL_DATES)
    rho1 = spearmanr(index_of(f_full, crop), ref).statistic
    print(f"C1 union reproduces shipped (known answer 0 / 0 / 1.0000): "
          f"max |dB| {dmax:.2e}  max |cv| {cmax:.2e}  rho {rho1:.4f}")
    if not (dmax < 1e-9 and cmax < 1e-9 and abs(rho1 - 1.0) < 1e-12):
        raise SystemExit("C1 FAILED: the core/ring split does not partition the shipped "
                         "pixel set; nothing below measures the shipped product")

    # ---------------- the split ----------------
    f_core, core_empty = features(G, lab_core, n, shipped)
    f_ring, ring_empty = features(G, lab_ring, n, shipped)
    lvl_core = f_core[f"g0_db_{PEAK}"].to_numpy()
    lvl_ring = f_ring[f"g0_db_{PEAK}"].to_numpy()
    both = ~(core_empty | ring_empty)
    d_ring = lvl_ring - lvl_core

    # ---------------- C2: inject a known offset into the ring ----------------
    # Run in BOTH dtypes. As first written this check used the cache's native float32 and
    # failed its own 1e-9 threshold at 1.94e-07 -- which is float32 epsilon, not a leak. The
    # two candidate causes separate cleanly and are not assumed apart: a mask that addressed
    # the wrong pixels would miss by a fraction of the 3 dB injection, i.e. by ~1e-1, in ANY
    # dtype, whereas storage precision vanishes in float64. So both are run and both are
    # reported, and the float32 residual is required to be small enough to be precision and
    # nothing else. [P3-23]
    def inject(dtype):
        arr = G["gamma0"][PEAK].astype(dtype)
        Gi = {"gamma0": dict(G["gamma0"])}
        Gi["gamma0"][PEAK] = np.where(lab_ring > 0, arr * dtype(10.0 ** (INJECT_DB / 10.0)), arr)
        r = features(Gi, lab_ring, n, shipped)[0][f"g0_db_{PEAK}"].to_numpy()
        c = features(Gi, lab_core, n, shipped)[0][f"g0_db_{PEAK}"].to_numpy()
        return float(np.nanmax(np.abs(((r - c) - d_ring)[both] - INJECT_DB)))

    c2_32, c2 = inject(np.float32), inject(np.float64)
    print(f"C2 ring injection (known answer {INJECT_DB:.3f} dB): max recovery error "
          f"{c2:.2e} dB in float64, {c2_32:.2e} dB in the cache's native float32")
    if not (c2 < 1e-9 and c2_32 < 1e-4):
        raise SystemExit("C2 FAILED: the ring mask does not address the pixels it claims to; "
                         "the core/ring attribution is void")

    # ---------------- C3: the matched-N random null ----------------
    lab_a, lab_b = matched_split(lab, n, core_cnt, RNG)
    f_a, a_empty = features(G, lab_a, n, shipped)
    f_b, _ = features(G, lab_b, n, shipped)
    d_rand = (f_b[f"g0_db_{PEAK}"].to_numpy() - f_a[f"g0_db_{PEAK}"].to_numpy())

    i_core = index_of(f_core, crop)
    i_rand = index_of(f_a, crop)
    for tag, v in (("core", i_core), ("rand", i_rand)):
        check_invariance(v, crop, area, tag)

    ref_list = bottom(ref, k)
    ret_core = len(bottom(i_core, k) & ref_list) / k
    ret_rand = len(bottom(i_rand, k) & ref_list) / k
    rho_core = spearmanr(i_core, ref).statistic
    rho_rand = spearmanr(i_rand, ref).statistic
    chance = k / n

    print(f"\nC3 matched-N random null discards the same {int(np.nansum(full_cnt - core_cnt))} "
          f"pixels, chosen at random")
    print(f"  |level difference| at peak   boundary split {np.nanmedian(np.abs(d_ring[both])):.3f} dB"
          f"   random split {np.nanmedian(np.abs(d_rand[both])):.3f} dB")
    print(f"  rank rho vs shipped          core-only {rho_core:.3f}"
          f"        matched random {rho_rand:.3f}")
    print(f"  attention list retained      core-only {ret_core:.3f}"
          f"        matched random {ret_rand:.3f}   (chance {chance:.3f})")
    if not (np.nanmedian(np.abs(d_ring[both])) > np.nanmedian(np.abs(d_rand[both]))):
        raise SystemExit("C3 FAILED: discarding pixels at random moves the level as much as "
                         "discarding the boundary does; the effect is sample size, not "
                         "geometry, and this test is discarded")

    # ---------------- what it does to `uniform`, the largest-weight part ----------------
    cv_full = shipped[f"cv_{PEAK}"].to_numpy()
    cv_core = f_core[f"cv_{PEAK}"].to_numpy()
    cv_rand = f_a[f"cv_{PEAK}"].to_numpy()
    drop_core = np.nanmedian((cv_core - cv_full)[both] / cv_full[both])
    drop_rand = np.nanmedian((cv_rand - cv_full)[both] / cv_full[both])
    print(f"\nuniform (weight 0.249, untestable in T24) -- within-farm CV at peak:")
    print(f"  core-only      median change {100*drop_core:+.1f}%   rho vs shipped "
          f"{spearmanr(cv_core[both], cv_full[both]).statistic:.3f}")
    print(f"  matched random median change {100*drop_rand:+.1f}%   rho vs shipped "
          f"{spearmanr(cv_rand[both], cv_full[both]).statistic:.3f}")

    # ---------------- is the attention list an edge-contamination list? ----------------
    listed = np.zeros(n, dtype=bool)
    listed[np.fromiter(ref_list, dtype=int, count=len(ref_list))] = True
    rho_rf = spearmanr(ref, ring_frac, nan_policy="omit").statistic
    print(f"\nis the shipped list an edge list?  rho(index, ring fraction) = {rho_rf:+.3f}")
    print(f"  ring fraction  listed {np.nanmedian(ring_frac[listed]):.3f}"
          f"   unlisted {np.nanmedian(ring_frac[~listed]):.3f}")
    print(f"  no-interior farms  listed {int(no_core[listed].sum())}/{k}"
          f"   unlisted {int(no_core[~listed].sum())}/{n-k}")

    # Ring fraction is reported over EVERY farm in the bin; the level difference only over
    # farms that have both a core and a ring, since it is undefined otherwise. Filtering both
    # columns the same way put the table's 0-20 px ring fraction at 0.80 while the figure drew
    # 1.00 -- the 75 no-interior farms all sit in that bin and they ARE parcels, so dropping
    # them understates exactly the population this test is about. [caught in F9]
    bins = [(0, 20), (20, 50), (50, 150), (150, 10 ** 9)]
    print("\n  parcel size       n   ring frac    n(core&ring)  |ring-core| dB")
    for lo, hi in bins:
        m = (full_cnt >= lo) & (full_cnt < hi)
        mb = m & both
        print(f"  {lo:4d}-{hi if hi < 10**9 else 0:<6d} n={int(m.sum()):4d}"
              f"   {np.nanmedian(ring_frac[m]):.3f}"
              f"         n={int(mb.sum()):4d}      {np.nanmedian(np.abs(d_ring[mb])):.3f}")

    # Ring fraction is very nearly a deterministic function of parcel size (perimeter over
    # area), which the figure made obvious. So rho(index, ring fraction) is close to a
    # statement that the index does not track parcel SIZE either -- checked, not assumed.
    rho_rf_npix = spearmanr(ring_frac, full_cnt, nan_policy="omit").statistic
    rho_idx_npix = spearmanr(ref, full_cnt).statistic
    print(f"\n  rho(ring frac, npix) = {rho_rf_npix:+.3f}   "
          f"rho(index, npix) = {rho_idx_npix:+.3f}")

    pd.DataFrame({
        "farm_id": shipped.farm_id, "crop_type": crop, "area_ha": area,
        "npix_full": full_cnt, "npix_core": core_cnt, "ring_frac": ring_frac,
        "no_interior": no_core, "buffer_level": level,
        "health_index": np.round(ref, 2), "health_core_only": np.round(i_core, 2),
        "health_matched_random": np.round(i_rand, 2),
        "level_ring_minus_core_db": np.round(d_ring, 3),
        "cv_full": cv_full, "cv_core": cv_core,
        "on_attention_list": listed,
        "stays_core_only": [i in bottom(i_core, k) for i in range(n)],
    }).to_csv(TABLES / "p3_edge_farm.csv", index=False)

    stats = {
        "n_farms": n, "attention_k": k, "chance_floor": chance,
        "median_npix_full": float(np.median(full_cnt)),
        "median_ring_frac": float(np.nanmedian(ring_frac)),
        "farms_no_interior": int(no_core.sum()),
        "farms_all_touched_rung": int((level == 2).sum()),
        "c1_max_db_err": dmax, "c1_max_cv_err": cmax, "c1_rho": rho1,
        "c2_max_recovery_err_db": c2, "c2_max_recovery_err_db_float32": c2_32,
        "med_abs_level_diff_ring_db": float(np.nanmedian(np.abs(d_ring[both]))),
        "med_abs_level_diff_random_db": float(np.nanmedian(np.abs(d_rand[both]))),
        "rho_core_vs_shipped": rho_core, "rho_random_vs_shipped": rho_rand,
        "retention_core": ret_core, "retention_random": ret_rand,
        "cv_change_core_pct": float(100 * drop_core),
        "cv_change_random_pct": float(100 * drop_rand),
        "rho_index_vs_ring_frac": rho_rf, "rho_ring_frac_vs_npix": rho_rf_npix,
        "rho_index_vs_npix": rho_idx_npix,
    }
    pd.DataFrame([stats]).to_csv(TABLES / "p3_edge_stats.csv", index=False)
    log("p3_edge.done", **{k2: (round(v, 4) if isinstance(v, float) else v)
                           for k2, v in stats.items()})
    print(f"\nwrote p3_edge_farm.csv, p3_edge_stats.csv")


def shipped_crop(shipped):
    """Crop labels come from the submission, which is the frozen deliverable."""
    sub = pd.read_csv(TABLES.parent / "submission.csv")[["farm_id", "crop_type"]]
    m = shipped[["farm_id"]].merge(sub, on="farm_id", how="left")
    assert m.crop_type.notna().all(), "crop label missing for some farm"
    return m.crop_type


if __name__ == "__main__":
    main()
