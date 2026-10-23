"""T13 / D7: spatial hold-out, and an honest answer about what there is to hold out.

Round 2 adopted spatial hold-out after a competitor demonstrated it, whereupon it promptly
overturned one of our own published claims. That history is the argument for running it by
default rather than when convenient.

★ BUT THE FIRST QUESTION IS WHAT IS ACTUALLY FITTED, and for this pipeline the answer is
almost nothing:

  the health weights   derived from REDUNDANCY alone, w_k ~ 1 / sum_j |rho(part_k, part_j)|.
                       A function of the feature matrix only. It never sees a witness, a
                       label or a yield, so there is no target to overfit to.
  the modifier         a monotone transform of the health index, normalised. No free
                       parameter fitted to anything.
  the anchors          published statistics. Not fitted.
  YIELD_SPREAD = 0.30  a stated prior from the literature, not a swept optimum.
  the GLCM residual    the ONE regression in the pipeline: entropy on log(npix), fitted per
                       date to remove a known estimator artefact.

A pipeline with nothing fitted to a target cannot overfit that target -- but that is a claim,
and claims get tested. So:

  V1  WEIGHT STABILITY.   Split the village spatially at the median easting, re-derive the
                          weights independently on each half, and compare. If the weights
                          are a property of the physics they should barely move; if they
                          swing, they are absorbing local structure and the "not fitted"
                          claim is weaker than stated.
  V2  RANK TRANSFER.      Build the health index from the WEST half's weights and score the
                          EAST half with it, and vice versa. Correlate against the index
                          built from the whole village. A model that only works where it
                          was derived shows up here.
  V3  WITNESS TRANSFER.   Re-run T1's headline correlation -- gamma0 against same-day S2
                          NDVI, within crop -- separately on each spatial half. If the
                          relationship only exists in one half it is a local artefact, not
                          a physical one.
  V4  GLCM LEAKAGE.       Fit the npix regression on one half, apply to the other, and check
                          the residual is still uncorrelated with plot size out of sample.

Writes results/tables/p3_validate.csv.

Run:  py -3.12 src/p3_validate.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CROPS, DATES, FARMS, RESULTS, TABLES, log
from p3_build import derive_weights, health_index, health_parts

import geopandas as gpd

UTM = 32643
R = []


def rec(test, item, ok, detail):
    R.append({"test": test, "item": item, "pass": bool(ok), "detail": detail})
    print(f"  {'PASS' if ok else 'FAIL'}  {test} {item:<22} {detail}")


def main():
    d = pd.read_csv(RESULTS / "d4_debug.csv")
    w2 = pd.read_csv(TABLES / "witness_s2.csv")
    d = d.merge(w2, on="farm_id")
    fa = gpd.read_file(FARMS).to_crs(UTM)
    x = fa.centroid.x.to_numpy()
    west = x <= np.median(x)
    crop = d.crop_type
    log("p3_val.split", west=int(west.sum()), east=int((~west).sum()),
        boundary_easting=round(float(np.median(x)), 1))

    # ---------------- V1 weight stability ----------------
    print("\nV1  are the health weights a property of the physics or of the place?")
    parts_all = health_parts(d, use_29oct=False)
    W_all = derive_weights(parts_all)
    Ws = {}
    for name, m in (("west", west), ("east", ~west)):
        Ws[name] = derive_weights({k: v[m] for k, v in parts_all.items()})
    drift = {k: abs(Ws["west"][k] - Ws["east"][k]) for k in W_all}
    print("      " + "  ".join(f"{k}: {W_all[k]:.3f}" for k in W_all))
    print("      west " + "  ".join(f"{Ws['west'][k]:.3f}" for k in W_all))
    print("      east " + "  ".join(f"{Ws['east'][k]:.3f}" for k in W_all))
    rec("V1", "weight drift", max(drift.values()) < 0.05,
        f"largest half-to-half drift {max(drift.values()):.4f} "
        f"({max(drift, key=drift.get)}); all weights ~{1/len(W_all):.2f} nominal")

    # ---------------- V2 rank transfer ----------------
    print("\nV2  does a model derived on one half rank the other half the same way?")
    hi_all = health_index(parts_all, W_all, crop)
    for src, tgt, m in (("west", "east", ~west), ("east", "west", west)):
        hi_x = health_index(parts_all, Ws[src], crop)
        ok = np.isfinite(hi_all) & np.isfinite(hi_x) & m
        rho = spearmanr(hi_all[ok], hi_x[ok])[0]
        rec("V2", f"{src}-weights on {tgt}", rho > 0.95,
            f"rho {rho:.4f} against the whole-village index (n={int(ok.sum())})")

    # ---------------- V3 witness transfer ----------------
    print("\nV3  does T1's witness relationship exist in BOTH halves?")
    for date in ("20251013", "20251112"):
        nd = d[f"ndvi_{date}"].to_numpy()
        val = d[f"ndvi_valid_{date}"].to_numpy()
        g = d[f"g0_db_{date}"].to_numpy()
        for c in ("Cotton", "Groundnut"):        # the two largest classes by area
            rhos = {}
            for name, m in (("west", west), ("east", ~west)):
                s = m & (crop == c).to_numpy() & np.isfinite(g) & np.isfinite(nd) & (val > 0.5)
                rhos[name] = spearmanr(g[s], nd[s])[0] if s.sum() >= 20 else np.nan
            same_sign = (np.sign(rhos["west"]) == np.sign(rhos["east"])
                         and min(rhos.values()) > 0.15)
            rec("V3", f"{c} {date[4:]}", bool(same_sign),
                f"west {rhos['west']:+.3f}  east {rhos['east']:+.3f}")

    # ---------------- V4 GLCM leakage ----------------
    print("\nV4  the one regression in the pipeline -- does it leak across the split?")
    for date in DATES[:1] + DATES[-1:]:
        ent = d[f"glcm_ent_{date}"].to_numpy()
        npx = d[f"npix_{date}"].to_numpy().astype("float64")
        ok = np.isfinite(ent) & (npx > 0)
        fit = ok & west
        A = np.c_[np.ones(int(fit.sum())), np.log(npx[fit])]
        coef = np.linalg.lstsq(A, ent[fit], rcond=None)[0]
        tgt = ok & ~west
        resid = ent[tgt] - (coef[0] + coef[1] * np.log(npx[tgt]))
        rho = spearmanr(np.log(npx[tgt]), resid)[0]
        rec("V4", f"glcm {date[4:]}", abs(rho) < 0.25,
            f"out-of-sample residual vs log(npix) rho {rho:+.3f} "
            f"(in-sample it is 0 by construction)")

    out = pd.DataFrame(R)
    out.to_csv(TABLES / "p3_validate.csv", index=False)
    n_fail = int((~out["pass"]).sum())
    log("p3_val.done", n=len(out), passed=int(out["pass"].sum()), failed=n_fail,
        failing=[f"{r.test}:{r.item}" for r in out.itertuples() if not r._3],
        claim=("nothing in the shipped path is fitted to a target; the spatial hold-out "
               "tests that claim rather than assuming it"))
    print(f"\n{int(out['pass'].sum())} passed, {n_fail} failed")


if __name__ == "__main__":
    main()
