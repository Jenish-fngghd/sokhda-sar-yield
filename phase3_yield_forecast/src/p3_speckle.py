"""T24 / P3-22: would the same 97 farms be on the attention list if we re-flew the mission?

The sub-village deliverable names individual farms. T22 asked whether that ranking is about
the farm or about its neighbourhood; T23 asked what each scene bought. Neither asked the
blunt question a farmer would ask:

    how much of my rank is the radar measuring my field, and how much is speckle?

This is not the ensemble spread already in §7. That resamples our MODELLING choices -- which
features, which weights, which dates. This resamples the MEASUREMENT: the same crop, the same
sensor, the same day, a different realisation of speckle. It is the one source of error we
can compute from first principles instead of arguing about, and it has never been propagated
into the ranking.

It matters here more than it would elsewhere because these parcels are TINY. Median farm:
69 pixels. A quarter of them sit under 29, and 159 farms have fewer than 20 pixels on the
peak-canopy date. The relative standard error of a speckle mean over N independent looks is
1/sqrt(N), so the median farm's own brightness is known to about 12% -- before any question
of whether the model is right.

METHOD. For a mean of N independent single-look intensity samples, the sample mean is
Gamma-distributed with shape N. So each farm-date is perturbed by a multiplicative
u ~ Gamma(k, 1/k), k = npix * F, applied consistently to the linear mean and the dB mean.
The index is then rebuilt with the SHIPPED p3_build.health_parts -- not a reimplementation --
and the bottom-decile attention list is recomputed. Repeat, and count how often each farm
comes back.

F is the number of independent looks per pixel and is swept rather than assumed, because
the honest value is bracketed rather than known:
    F = 2    optimistic
    F = 1    nominal -- one independent look per pixel
    F = 0.5  conservative: neighbouring pixels are correlated, so the effective independent
             sample count is below the pixel count

CONTROLS:

  C1  ZERO NOISE. With the perturbation switched off, every farm must be retained and the
      ranking must be identical -- retention exactly 1.000, rho exactly 1.000. Anything else
      means the harness perturbs something it should not, and every number below is void.
      Runs first, aborts on failure.

  C2  POSITIVE CONTROL, AS FIRST WRITTEN -- AND IT FAILED. Retention among listed farms was
      required to rise with pixel count; it came back rho = -0.008. The control was
      mis-specified, and the run below reports it as failed rather than deleting it.
      Retention among ALREADY-LISTED farms is dominated by MARGIN -- how far below the
      decile boundary a farm sits. A farm deep in the bottom returns whatever its size; a
      farm on the boundary flips whatever its size. Size and margin are close to
      independent, so the size effect is invisible in that statistic. The margin effect is
      measured below so this is shown rather than asserted.

  C2b POSITIVE CONTROL, CORRECTLY SPECIFIED. The underlying claim is that the noise model
      reaches the index in a SIZE-DEPENDENT way, so test that directly: the per-farm spread
      of the index across replicates must rise as pixel count falls. It does, decisively --
      rho(index SD, npix) = -0.62, with mean SD 12.3 points for farms under 20 pixels
      against 4.1 for farms over 150. This is the control the conclusion rests on.

  C3  CHANCE FLOOR. A 97-farm list drawn at random from 966 farms retains 97/966 = 10.0% by
      chance. Retention is reported against that floor, not against zero.

  C4  Invariance. The village total is anchor x area and cannot move; asserted per replicate.

HONEST BOUND, AND IT IS NOT A SMALL ONE. Only the LEVEL is perturbed, not the within-farm CV
that feeds `uniform` -- and `uniform` carries the LARGEST weight in the shipped index, 0.249.
So a quarter of the index is noise-free by construction of this test, and every retention
below is an UPPER bound: the true stability is worse than what is reported here. Stated
rather than modelled, because a speckle model for a texture statistic needs the spatial
correlation length, which we do not have.

Writes results/tables/p3_speckle_farm.csv, p3_speckle_sweep.csv.

Run:  py -3.12 -u src/p3_speckle.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CROPS, TABLES, log
from p3_build import LEVEL_DATES, derive_weights, health_index, health_parts, modifier

sys.stdout.reconfigure(encoding="utf8")
warnings.filterwarnings("ignore", category=RuntimeWarning)

RNG = np.random.default_rng(20260831)
NREP = 500
LOOKS = [2.0, 1.0, 0.5]          # independent looks per pixel; swept, not assumed
DECILE = 0.10


def perturb(f, F, rng):
    """A speckle realisation of the same scene: multiplicative Gamma noise on each
    farm-date mean, shape = npix * F. Applied to the linear mean and the dB mean together
    so the two stay consistent whichever convention the extraction used."""
    g = f.copy()
    for d in LEVEL_DATES:
        n = f[f"npix_{d}"].to_numpy(dtype="float64")
        k = np.maximum(n, 1.0) * F
        # The zero-noise case has to be written, not taken as a limit: numpy's gamma
        # returns NaN at shape=inf rather than collapsing onto 1.0, which made C1 report
        # chance-level retention on its first run. [caught by C1, P3-22]
        u = np.ones_like(n) if not np.isfinite(F) else rng.gamma(shape=k, scale=1.0 / k)
        g[f"g0_lin_{d}"] = f[f"g0_lin_{d}"].to_numpy() * u
        g[f"g0_db_{d}"] = f[f"g0_db_{d}"].to_numpy() + 10.0 * np.log10(u)
    return g


def index_of(f, crop):
    """The shipped index, built by the shipped function on whatever features it is given."""
    parts = health_parts(f, use_29oct=False)
    return health_index(parts, derive_weights(parts), crop)


def bottom(idx, k):
    return set(np.argsort(idx)[:k])


def main():
    f = pd.read_csv(TABLES / "p1_farm_features.csv")
    sub = pd.read_csv(TABLES.parent / "submission.csv")
    d = sub.merge(f, on="farm_id", how="left")
    assert len(d) == len(sub)
    crop = d.crop_type
    area = d.area_ha.to_numpy(dtype="float64")
    n = len(d)
    k = max(1, int(round(DECILE * n)))
    npix = d.npix_20250814.to_numpy(dtype="float64")
    print(f"{n} farms | attention list = bottom {k} | median npix {np.median(npix):.0f} "
          f"=> median speckle SE {100/np.sqrt(max(np.median(npix),1)):.0f}%")

    ref = index_of(f, crop)
    r0 = spearmanr(ref, d.health_index.to_numpy()).statistic
    print(f"harness reproduces the shipped index: rho = {r0:.4f}")
    ref_list = bottom(ref, k)

    # ---------------- C1: zero noise must be a no-op ----------------
    g0 = perturb(f, np.inf, RNG)          # shape -> inf collapses the Gamma onto 1.0
    i0 = index_of(g0, crop)
    ret0 = len(bottom(i0, k) & ref_list) / k
    rho0 = spearmanr(i0, ref).statistic
    print(f"C1 zero noise (known answer 1.000 / 1.0000): retention {ret0:.3f}, rho {rho0:.4f}")
    if not (abs(ret0 - 1.0) < 1e-12 and abs(rho0 - 1.0) < 1e-12):
        raise SystemExit("C1 FAILED: the perturbation harness is not a no-op at zero noise")

    chance = k / n
    print(f"C3 chance floor: a random {k}-farm list retains {100*chance:.1f}%\n")

    # margin: how far the farm sits below the decile boundary, in index points. The
    # quantity C2 turned out to be measuring.
    boundary = np.sort(ref)[k - 1]
    margin = boundary - ref

    sweep = []
    hits = {F: np.zeros(n) for F in LOOKS}
    sds = {}
    for F in LOOKS:
        rets, rhos, V = [], [], []
        for _ in range(NREP):
            g = perturb(f, F, RNG)
            v = index_of(g, crop)
            b = bottom(v, k)
            rets.append(len(b & ref_list) / k)
            rhos.append(spearmanr(v, ref).statistic)
            idxs = np.fromiter(b, dtype=int, count=len(b))
            hits[F][idxs] += 1.0
            V.append(v)
            # C4: the modifier renormalises, so the village total cannot move
            m = modifier(v, crop, area)
            for c in CROPS:
                msk = (crop == c).to_numpy()
                if msk.sum() >= 3:
                    assert abs(np.average(m[msk], weights=area[msk]) - 1.0) < 1e-9
        hits[F] /= NREP
        rets, rhos = np.array(rets), np.array(rhos)
        sds[F] = np.std(np.array(V), axis=0)
        listed = np.fromiter(ref_list, dtype=int, count=len(ref_list))
        c2 = spearmanr(hits[F][listed], npix[listed]).statistic       # as first written
        c2b = spearmanr(sds[F], npix).statistic                       # correctly specified
        c2m = spearmanr(hits[F][listed], margin[listed]).statistic     # what C2 measured
        print(f"F={F:<4} retention {rets.mean():.3f} [{np.percentile(rets,5):.3f}, "
              f"{np.percentile(rets,95):.3f}]   rank rho {rhos.mean():.3f}")
        print(f"       C2  rho(retention, npix)   = {c2:+.3f}  <- mis-specified, reported failed")
        print(f"       C2m rho(retention, margin) = {c2m:+.3f}  <- what C2 was really measuring")
        print(f"       C2b rho(index SD, npix)    = {c2b:+.3f}  <- the control that counts")
        sweep.append({"looks_per_pixel": F, "retention": rets.mean(),
                      "retention_lo": np.percentile(rets, 5),
                      "retention_hi": np.percentile(rets, 95),
                      "rank_rho": rhos.mean(), "c2_retention_vs_npix": c2,
                      "c2m_retention_vs_margin": c2m, "c2b_sd_vs_npix": c2b,
                      "chance_floor": chance})

    F0 = 1.0
    nom = [s for s in sweep if s["looks_per_pixel"] == F0][0]
    if not (nom["c2b_sd_vs_npix"] < -0.30):
        raise SystemExit(f"C2b FAILED: index spread does not track pixel count "
                         f"({nom['c2b_sd_vs_npix']:+.3f}); the noise model is not reaching "
                         "the index and nothing here can be reported")
    if not (nom["c2m_retention_vs_margin"] > 0.30):
        raise SystemExit("C2's failure is NOT explained by margin; the mis-specification "
                         "claim is unsupported and this test must be discarded")

    out = pd.DataFrame({"village_id": d.village_id_x, "farm_id": d.farm_id,
                        "crop_type": d.crop_type, "health_index": d.health_index,
                        "npix_peak": npix.astype(int),
                        "on_attention_list": [i in ref_list for i in range(n)]})
    for F in LOOKS:
        out[f"retain_F{F}"] = np.round(hits[F], 3)
    out["index_sd_F1"] = np.round(sds[F0], 2)
    out.to_csv(TABLES / "p3_speckle_farm.csv", index=False)

    print("\nC2b in full -- index spread by parcel size:")
    for lo, hi in [(0, 20), (20, 50), (50, 150), (150, 10 ** 9)]:
        m = (npix >= lo) & (npix < hi)
        print(f"   npix {lo:>4}-{'inf' if hi > 10**8 else hi:<5} n={m.sum():4d}   "
              f"mean index SD {sds[F0][m].mean():5.2f} points")
    pd.DataFrame(sweep).to_csv(TABLES / "p3_speckle_sweep.csv", index=False)

    li = out[out.on_attention_list]
    q = li.retain_F1_0 if "retain_F1_0" in li else li[f"retain_F{F0}"]
    solid = int((q >= 0.8).sum())
    weak = int((q < 0.5).sum())
    print(f"\nOf the {k} farms we name: {solid} come back in >=80% of re-flights, "
          f"{weak} in under half.")
    small = li[li.npix_peak < 30]
    print(f"The {len(small)} listed farms under 30 px retain {small[f'retain_F{F0}'].mean():.2f}; "
          f"the {len(li)-len(small)} larger ones retain "
          f"{li[li.npix_peak >= 30][f'retain_F{F0}'].mean():.2f}.")

    log("p3_speckle", n=n, k=k, c1_pass=True,
        retention_nominal=float(np.round([s["retention"] for s in sweep
                                          if s["looks_per_pixel"] == F0][0], 4)),
        chance_floor=round(chance, 4), solid=solid, weak=weak)


if __name__ == "__main__":
    main()
