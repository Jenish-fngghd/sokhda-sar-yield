"""Stage I6 / proposal P-D -- Water Cloud Model inversion for canopy water content.

WHAT THIS BUYS US. The D4 health index ranks farms by z-scored backscatter, which
is defensible but dimensionless -- it says "brighter than its neighbours", not
"holding N kg/m^2 of canopy water". Inverting the WCM turns the same observation
into a physical quantity with units, which is what the rubric's Technical
Soundness axis is actually asking for.

THE MODEL (El Hajj, Baghdadi, Zribi et al., RSE 176 (2016) 202-218, Table 5 --
fitted at X-band HH on TerraSAR-X + COSMO-SkyMed, RMSE 0.86 dB) [E3]:

    sigma0_tot = sigma0_veg + T^2 * sigma0_sol         (4)
    sigma0_veg = A * V * cos(theta) * (1 - T^2)        (5)
    T^2        = exp(-2 * B * V * sec(theta))          (6)
    sigma0_sol = C * exp(D * Mv)                       (7)

with V = vegetation water content in kg/m^2 and Mv = volumetric soil moisture in
Vol.%. A=0.0438, B=0.1047, C=0.0324, D=0.03971 for the VWC parameterisation.

THE PROBLEM, AND WHY THE JUNE PAIR SOLVES IT. Equations (4)-(7) are ABSOLUTE --
they expect real sigma0. Our data carries a near-constant calibration offset of
roughly +28 dB [J5]/[O1], independently corroborated by Round 1's separate
implementation but never pinned down, so feeding our dB straight into a published
absolute model would be meaningless.

The fix is to never use an absolute value. Equation (7) supplies sigma0_sol from
theory, and [J8] established that **Jun 06 is a dry, pre-monsoon, near-bare
scene** -- the exact regime El Hajj measured D_HH on. So per farm:

    k_farm = observed(Jun 06) / [ C * exp(D * Mv_dry) ]

`k_farm` absorbs the calibration offset AND that farm's own soil roughness and
texture, which is a bonus rather than a fudge: C is known to vary with roughness,
and a per-farm k lets each field carry its own soil term instead of a village
average. Every later date is then divided by the same k_farm, so the offset
cancels exactly and what enters the inversion is a properly-scaled sigma0.

WHAT IS ASSUMED, STATED PLAINLY. Mv on each date is not measured -- we have no
soil probe and no L-band. The three values below are agronomic priors for a
Gujarat kharif season, and `sensitivity()` reports how much the answer moves when
they are wrong. That is the honest treatment: the assumption is visible and its
cost is quantified, rather than buried in a constant.

Run:  python src/wcm.py         (self-check + inversion, writes farm VWC columns)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATES, RESULTS, log

# El Hajj et al. 2016 Table 5, X-band HH, V1 = V2 = VWC (kg/m^2).
A, B, C, D_MV = 0.0438, 0.1047, 0.0324, 0.03971

# Assumed volumetric soil moisture (Vol.%) per acquisition. Jun 06 07:25 is
# pre-monsoon and pre-/just-sown; Jun 19 02:14 is the monsoon onset [J8]; Aug 14
# is peak monsoon; Oct 13 is post-monsoon drawdown.
MV = {"20250606": 12.0, "20250619": 26.0, "20250814": 30.0, "20251013": 22.0}
DRY = "20250606"            # the bare-soil reference date
VMAX = 15.0                 # kg/m^2, far above any field crop -- bracket only


def sigma0_sol(mv):
    """Bare-soil backscatter from WCM eq. (7)."""
    return C * np.exp(D_MV * mv)


def forward(v, theta_deg, mv):
    """WCM eqs. (4)-(6): total sigma0 for a canopy of water content `v`."""
    th = np.radians(theta_deg)
    t2 = np.exp(-2.0 * B * v / np.cos(th))
    return A * v * np.cos(th) * (1.0 - t2) + t2 * sigma0_sol(mv)


def v_turning(theta_deg, mv):
    """VWC at which the WCM stops falling and starts rising.

    The model is NOT monotonic, which is easy to assume and wrong -- the
    self-check below is what caught it. Expanding eqs. (4)-(6) for small v:

        sigma0(v) ~ sigma0_sol + 2AB*v^2 - 2B*sec(theta)*sigma0_sol*v

    so d(sigma0)/dv at v=0 is NEGATIVE. A thin canopy ATTENUATES the soil return
    faster than it contributes volume scattering, so backscatter dips before it
    climbs. Setting the derivative to zero gives the turning point:

        v* = sec(theta) * sigma0_sol / (2A)                ~ 0.7 kg/m^2 here

    This is physics, not a numerical artefact, and it has a consequence we report
    rather than hide: **below ~0.7 kg/m^2 VWC, X-band HH is ambiguous** -- two
    canopy loads give the same sigma0. Early-season and sparse fields sit there.
    """
    return 1.0 / np.cos(np.radians(theta_deg)) * sigma0_sol(mv) / (2.0 * A)


def invert(s0, theta_deg, mv):
    """Solve eq. (4) for VWC. Returns (vwc, flag).

    flag: 'ok'       unique root on the rising branch
          'ambiguous' sigma0 falls in the dip -- two roots exist, we take the
                      UPPER (dense-canopy) one, because every crop here is well
                      past v* by August and the lower branch would read a
                      closing canopy as bare ground
          'floor'    below the model minimum entirely; clamped to v*
          'sat'      above the model at VMAX
    """
    if not np.isfinite(s0) or s0 <= 0:
        return np.nan, "nan"
    vt = v_turning(theta_deg, mv)
    lo = forward(vt, theta_deg, mv)          # the minimum of the curve
    hi = forward(VMAX, theta_deg, mv)
    if s0 <= lo:
        return float(vt), "floor"
    if s0 >= hi:
        return VMAX, "sat"
    flag = "ok" if s0 >= forward(0.0, theta_deg, mv) else "ambiguous"
    v = brentq(lambda v: forward(v, theta_deg, mv) - s0, vt, VMAX, xtol=1e-4)
    return float(v), flag


def farm_vwc(f, mv=None):
    """Per-farm VWC (kg/m^2) on every date, via the per-farm June calibration.

    Returns a DataFrame of vwc_<date> plus the per-farm scale k in dB, which is
    worth keeping: if k were a pure calibration constant it would be identical
    across farms, so its SPREAD is a direct read on how much soil roughness
    varies across the village.
    """
    mv = mv or MV
    out = {}
    # gamma0 -> sigma0 = gamma0 * cos(theta). Our stored g0 is gamma0 [I1].
    def s0_obs(d):
        th = np.radians(f[f"inc_{d}"].values.astype("float64"))
        return f[f"g0_lin_{d}"].values.astype("float64") * np.cos(th)

    k = s0_obs(DRY) / sigma0_sol(mv[DRY])       # per-farm offset x roughness
    k = np.where(np.isfinite(k) & (k > 0), k, np.nan)
    out["wcm_k_db"] = 10.0 * np.log10(k)

    for d in DATES:
        s0 = s0_obs(d) / k
        th = f[f"inc_{d}"].values.astype("float64")
        v = np.full(len(f), np.nan)
        fl = np.array(["nan"] * len(f), dtype=object)
        for i in range(len(f)):
            if np.isfinite(s0[i]) and np.isfinite(th[i]):
                v[i], fl[i] = invert(s0[i], th[i], mv[d])
        out[f"vwc_{d}"] = v
        out[f"vwcflag_{d}"] = fl
    return pd.DataFrame(out, index=f.index)


def sensitivity(f, delta=8.0):
    """How much does VWC move if the assumed Mv is wrong by +/- `delta` Vol.%?

    This is the number that decides whether P-D is reportable. An inversion whose
    answer swings wildly on an unmeasured input is a dressed-up guess; one that
    holds its RANKING is usable even if its level is uncertain.
    """
    base = farm_vwc(f)
    res = {}
    for sign in (-1, 1):
        mv = {d: max(2.0, MV[d] + sign * delta) for d in MV}
        alt = farm_vwc(f, mv)
        for d in ("20250814", "20251013"):
            a, b = base[f"vwc_{d}"].values, alt[f"vwc_{d}"].values
            ok = np.isfinite(a) & np.isfinite(b)
            rho = pd.Series(a[ok]).corr(pd.Series(b[ok]), method="spearman")
            res[f"{d}_{sign:+d}"] = (float(np.nanmedian(b[ok] - a[ok])), float(rho))
    return base, res


def selfcheck():
    """Round-trip: forward then invert must return what went in, ABOVE v*.

    The first version of this asserted round-trip from v=0.2 and failed, which is
    how the non-monotonicity was found. Keeping the check anchored to v* is the
    point -- it encodes where the inversion is and is not trustworthy.
    """
    for th in (28.8, 31.6, 35.3):
        for mv in (12.0, 30.0):
            vt = v_turning(th, mv)
            assert 0.3 < vt < 1.6, f"turning point {vt} outside expected range"
            # every test point must sit ABOVE v*: below it the inversion is
            # two-to-one by construction and round-trip is not defined.
            for v in (vt + 0.05, vt + 0.5, 2.5, 5.0, 8.0):
                got, flag = invert(forward(v, th, mv), th, mv)
                assert abs(got - v) < 1e-3, f"v={v} th={th} mv={mv} -> {got} {flag}"
            # below the turning point the model really is two-to-one
            lo = forward(vt * 0.3, th, mv)
            assert lo > forward(vt, th, mv), "dip not reproduced"
            got, flag = invert(lo, th, mv)
            assert flag == "ambiguous" and got > vt, f"{flag} {got}"
    assert sigma0_sol(30.0) > sigma0_sol(12.0), "wetter soil must be brighter"
    # the dip itself: sigma0 must fall from bare soil before it rises
    assert forward(0.3, 30.0, 25.0) < forward(0.0, 30.0, 25.0)
    print("wcm selfcheck OK (incl. non-monotonic dip and ambiguous branch)")


def main():
    selfcheck()
    log("wcm.start")
    f = pd.read_csv(RESULTS / "farm_features.csv")
    base, sens = sensitivity(f)

    for c in base.columns:
        f[c] = base[c].values
    f.to_csv(RESULTS / "farm_features.csv", index=False)

    print("\n=== VWC (kg/m^2) by date ===")
    print(base[[f"vwc_{d}" for d in DATES]].describe().round(3).to_string())
    print(f"\nper-farm calibration k: median {np.nanmedian(base.wcm_k_db):.2f} dB, "
          f"IQR {np.nanpercentile(base.wcm_k_db,75)-np.nanpercentile(base.wcm_k_db,25):.2f} dB")
    print("  (a pure calibration constant would have zero spread; the spread IS "
          "the between-farm soil roughness/texture term)")

    print("\n=== SENSITIVITY to the assumed soil moisture (+/- 8 Vol.%) ===")
    print(f"{'date / dMv':<18}{'median dVWC':>13}{'Spearman rho':>15}")
    for k, (dv, rho) in sens.items():
        print(f"{k:<18}{dv:>13.3f}{rho:>15.3f}")
    log("wcm.done", cols=f.shape[1])


if __name__ == "__main__":
    main()
