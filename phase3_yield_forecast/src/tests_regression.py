"""Failure-mode regression suite. One assertion per defect that got past somebody competent.

The point is not coverage. Every check here corresponds to a bug that SURVIVED internal
consistency checking in a working pipeline -- ours, or a competitor's -- and was invisible
until someone went looking. Round 2's suite is ported forward and three classes it never
covered are added:

  SILENT NO-OP   an option that is accepted and does nothing. GDAL takes METHOD='GCP_TPS'
                 and ignores it; the key that works is SRC_METHOD='GCP_TPS'. Output that
                 passes every consistency check while the intended operation never ran.
                 The general guard: if a parameter is supposed to change a result, assert
                 that the result actually changed.
  DEGENERATE     a collapsed forecast, a single-crop map or a constant column must FAIL a
                 check rather than pass a schema.
  OFF-BY-ONE     a 1-based/0-based join that silently shifts every farm by one row. Tested
                 with a POSITIVE CONTROL: the deliberately shifted join must score much
                 worse than the real one, or the test cannot detect the bug it is for.

Run:  py -3.12 src/tests_regression.py
"""
import os
import sys
import traceback
from pathlib import Path

os.environ.pop("PROJ_LIB", None)

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AUX, CACHE, DATES, R2, TABLES, beta0, db, scene

BALE_KG = 170.0     # Indian cotton statistics report production in bales of 170 kg lint
N_FARMS = 966


# --------------------------------------------------------------------------- calibration

def test_scale_factor_is_squared():
    """R1 and R2 shipped beta0 = SF x |z|^2. The square is the fix and it must stay fixed.

    A unit DN of 1+0j must calibrate to exactly SF^2. If someone 'simplifies' common.beta0
    back to the unsquared form, every downstream dB moves by ~28 dB and nothing else in the
    pipeline notices, because every internal comparison is between our own numbers.
    """
    for d in DATES:
        sf = scene(d)["scale_factor"]
        got = float(beta0(np.array([1 + 0j]), d)[0])
        assert abs(got - sf * sf) < 1e-24, f"{d}: beta0(1+0j) = {got}, expected SF^2 = {sf*sf}"
        assert abs(got - sf) > 1e-9, f"{d}: beta0 is unsquared -- the R2 defect is back"


def test_darkest_content_sits_above_the_declared_noise_floor():
    """P1-2: a scene's darkest pixels cannot be quieter than the noise the vendor declares.

    This is the check that adjudicated the calibration convention, and it is also the
    check that catches a calibration DOUBLE-correction: applying SF^2 twice, or applying
    an incidence correction that has already been applied, drives the floor through the
    declared NESZ and this fires. It is asserted in BETA0, the quantity nesz_peak is
    referenced to -- in gamma0 it would fire on four of six correct scenes.
    """
    for d in DATES:
        with __import__("rasterio").open(CACHE / f"beta0_base_{d}.tif") as s:
            a = s.read(1).astype("float64")
        a = a[np.isfinite(a) & (a > 0)]
        p = float(np.percentile(10 * np.log10(a), 0.1))
        margin = p - scene(d)["nesz_peak"]
        assert 0.0 < margin < 6.0, \
            f"{d}: darkest 0.1% sits {margin:+.2f} dB from nesz_peak (want 0..6)"


def test_incidence_matches_the_vendors_own_annotation():
    """The geoid class of error. Another team's geoid mistake survived their whole pipeline.

    Our incidence comes from the orbit state vectors; the vendor computed a centre angle
    from the same orbit. Agreement to a hundredth of a degree is what licenses using ours.
    A wrong ellipsoid, a height convention slip or a swapped ECEF axis all break this.
    """
    from p1_prep import incidence_at
    import rasterio
    from common import slc_path
    for d in DATES:
        with rasterio.open(slc_path(d)) as s:
            gcps, _ = s.gcps
        _, inc_gcp = incidence_at(d, np.zeros(1), np.zeros(1), gcps)
        delta = float(inc_gcp.mean()) - float(scene(d)["incidence"])
        assert abs(delta) < 0.02, f"{d}: incidence off by {delta:+.4f} deg"


# ------------------------------------------------------------------------------- joins

def test_farm_join_is_not_off_by_one():
    """A 1-based/0-based slip shifts every farm by one row and changes no schema.

    POSITIVE CONTROL, without which this test cannot fail for the right reason: the join is
    deliberately shifted by one and must score MUCH worse. If the shifted join scores as
    well as the true one, farm_id is not carrying identity and this test is worthless.
    """
    ours = pd.read_csv(TABLES / "p1_farm_features.csv").sort_values("farm_id")
    them = pd.read_csv(AUX / "inherited" / "orion_feat.csv").sort_values("farm_id")
    assert list(ours.farm_id) == list(range(1, N_FARMS + 1)), "farm_id is not 1..966"
    assert list(ours.farm_id) == list(them.farm_id), "the two tables disagree on farm_id"

    x = ours["g0_db_20251013"].to_numpy()
    y = them["g0_db_T4"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    true_r = float(np.corrcoef(x[ok], y[ok])[0, 1])

    xs, ys = x[:-1], y[1:]                       # the off-by-one bug, applied on purpose
    ok = np.isfinite(xs) & np.isfinite(ys)
    shifted_r = float(np.corrcoef(xs[ok], ys[ok])[0, 1])

    assert true_r > 0.8, f"the true join only reaches r={true_r:.3f}"
    assert true_r - shifted_r > 0.5, \
        f"a one-row shift barely hurts (r {true_r:.3f} -> {shifted_r:.3f}); farm_id is not identity"


def test_every_farm_survives_to_the_table():
    """Coverage is 15 points. A farm with no pixels must still be a ROW, with a flag.

    Round 2's rule, kept: the extraction degrades through a buffer ladder and no farm is
    ever dropped. 29 farms have no SAR data on any date -- they must be present and marked,
    not silently absent.
    """
    d = pd.read_csv(TABLES / "p1_farm_features.csv")
    assert len(d) == N_FARMS, f"{len(d)} rows, expected {N_FARMS}"
    assert d.farm_id.is_unique
    assert (d.buffer_level >= 0).all(), "a farm failed every rung of the buffer ladder"
    assert set(d.qc_flag) <= {"ok", "partial", "no_sar_data"}
    n_none = int((d.n_dates == 0).sum())
    assert n_none == 29, f"{n_none} farms with no data -- was 29; coverage geometry changed"
    assert (d.loc[d.n_dates == 0, "qc_flag"] == "no_sar_data").all()


# ------------------------------------------------------------------------- silent no-op

def test_a_parameter_that_should_change_the_result_does():
    """The silent-no-op class, in the exact place it would bite us.

    GDAL accepts METHOD='GCP_TPS' and silently ignores it; SRC_METHOD='GCP_TPS' is the key
    that works. A parameter that is accepted and does nothing yields output that passes
    every consistency check while the intended operation never ran.

    Our geocoder takes a `resampling` argument, and `average` resampling IS our multilook --
    if it were being ignored we would be shipping nearest-neighbour single-look data with
    no visible symptom. So: run the same warp under two resamplings and assert the outputs
    actually differ. The generalisation is the point, not this one call.
    """
    from rasterio.warp import Resampling
    from p1_prep import geocode, slant

    bet, _, _, _, gcps, gcrs = slant(DATES[0])
    avg, _ = geocode(bet, gcps, gcrs, 20.0, Resampling.average)
    near, _ = geocode(bet, gcps, gcrs, 20.0, Resampling.nearest)
    ok = np.isfinite(avg) & np.isfinite(near)
    assert ok.sum() > 100, "the trial warp produced almost no valid pixels"
    assert not np.allclose(avg[ok], near[ok]), \
        "average and nearest resampling give identical output -- the argument is a no-op"
    # and it must differ in the direction a multilook does: less variance, not more
    assert avg[ok].std() < near[ok].std(), \
        "average resampling did not reduce variance -- it is not multilooking"


def test_texture_is_computed_on_unfiltered_data():
    """A speckle filter destroys the second-order statistics the texture features measure.

    Detected without inspecting any code: on single-look-derived data the lag-1 spatial
    autocorrelation of a FINE-grid dB image is low. A smoothing filter drives it toward 1.
    If someone inserts a Lee or boxcar filter upstream, this fires.
    """
    import rasterio
    with rasterio.open(CACHE / f"gamma0_fine_{DATES[-1]}.tif") as s:
        a = db(s.read(1).astype("float64"))
    a = a[200:-200, 200:-200]
    x, y = a[:, :-1], a[:, 1:]
    ok = np.isfinite(x) & np.isfinite(y)
    r = float(np.corrcoef(x[ok], y[ok])[0, 1])
    assert r < 0.85, f"lag-1 correlation {r:.3f} -- the fine grid looks smoothed"


def test_no_degenerate_parcel_defines_a_grid():
    """R2 e12: parcels enclosing ~0 ha, whose centroids landed up to 835 km away.

    They are legitimate rows in the submission. They are NOT legitimate inputs to anything
    spatial -- they stretched the village extent to '139 x 1110 km' and made a whole
    variance curve meaningless. Orion independently documented ten of them.

    ★ Re-testing this in Round 3 split the defect in two, and the halves need different
    guards. The zero-area parcels are real and still in the shapefile -- ten of them, the
    smallest at 0.0000 m2. The 835 km displacement was NOT: it came from Round 2's own
    hand-rolled area-weighted shoelace centroid dividing by a near-zero area. A library
    centroid puts all ten inside the village. So one assertion guards the data (the
    degenerate parcels must not silently disappear) and the other guards OUR CODE (whatever
    centroid routine we use must not fling them across the subcontinent).
    """
    import geopandas as gpd
    from common import FARMS
    f = gpd.read_file(FARMS).to_crs(32643)
    n_tiny = int((f.area < 1.0).sum())
    assert n_tiny == 10, f"{n_tiny} sub-1 m2 parcels -- was 10; the shapefile changed"

    xy = np.c_[f.centroid.x, f.centroid.y]
    dist = np.hypot(xy[:, 0] - np.median(xy[:, 0]), xy[:, 1] - np.median(xy[:, 1]))
    assert dist.max() < 5000, \
        f"a centroid landed {dist.max()/1000:.0f} km out -- the R2 shoelace bug is back"


# ------------------------------------------------------------- units and degenerate output

def test_apy_units_are_homogeneous():
    """R2 e15: the cotton row is bales-of-lint, the other four are tonnes.

    production/area reproduces the stated yield at 0.91-1.08 for rice, maize, bajra and
    groundnut, and at 5.76 for cotton. Nothing noticed, because the only yield guard is an
    UPPER bound and this defect makes cotton 2.9x too SMALL. It reaches 43% of village area.
    This does not assert which convention is right. It asserts the inhomogeneity is still
    there and still known, so nobody reads 776 kg/ha as comparable to bajra's 2714.
    """
    p = R2 / "data_aux" / "vadodara_apy.csv"
    if not p.exists():
        raise FileNotFoundError(f"anchor table missing: {p}")
    a = pd.read_csv(p).set_index("crop")
    ratio = (1000 * a.production_t_2022_23 / a.area_ha_2022_23) / a.yield_kg_ha_2022_23
    assert ratio.drop("Cotton").between(0.85, 1.15).all(), \
        f"a non-cotton crop changed units: {ratio.drop('Cotton').to_dict()}"
    assert ratio["Cotton"] > 3.0, "cotton looks unit-consistent now -- re-run the e15 analysis"
    cot = a.loc["Cotton"]
    as_bales = cot.production_t_2022_23 * BALE_KG / cot.area_ha_2022_23
    assert abs(as_bales / cot.yield_kg_ha_2022_23 - 1.0) < 0.10, \
        f"the bales reading no longer reconciles: {as_bales:.0f} vs {cot.yield_kg_ha_2022_23}"


def test_features_are_not_degenerate():
    """A constant column, a collapsed distribution or an all-NaN feature must FAIL here.

    A schema check passes all three. This is the class that ships a forecast of one number
    repeated 966 times with a perfectly valid header.
    """
    d = pd.read_csv(TABLES / "p1_farm_features.csv")
    cols = [c for c in d.columns
            if c.startswith(("g0_db_", "cv_", "ktex_", "glcm_resid_", "ref_", "d_"))]
    assert cols, "no feature columns found -- the extraction changed shape"
    for c in cols:
        v = d[c].to_numpy(dtype="float64")
        fin = np.isfinite(v)
        assert fin.mean() > 0.5, f"{c}: {100*(1-fin.mean()):.0f}% NaN"
        assert np.nanstd(v[fin]) > 1e-9, f"{c} is constant -- a degenerate feature"
        assert len(np.unique(np.round(v[fin], 6))) > 10, f"{c} takes almost no distinct values"


def test_forecast_sanity():
    """The battery for the forecast itself, decided NOW while there is no result to defend.

    Skips until a forecast exists. Deciding these thresholds after seeing a number is how a
    sanity check becomes a rationalisation.
    """
    p = Path(__file__).resolve().parent.parent / "results" / "submission.csv"
    if not p.exists():
        return "SKIP -- no forecast yet; thresholds are fixed above and will apply on sight"
    s = pd.read_csv(p)
    y = s.filter(like="yield").iloc[:, 0]
    assert len(s) == N_FARMS, f"{len(s)} rows"
    assert y.max() < 25.0, f"max {y.max()} t/ha -- looks like kg/ha, a 1000x unit error"
    assert y.min() >= 0.0
    assert y.nunique() > 20, "the forecast has collapsed to a handful of values"
    if "crop_type" in s:
        assert s.crop_type.nunique() >= 3, "a single-crop map is not a crop map"
    return None


def test_r2_submission_is_still_frozen():
    """Round 2 shipped. Its submission must never change, and this is the tripwire."""
    import hashlib
    p = R2 / "results" / "submission.csv"
    if not p.exists():
        raise FileNotFoundError(f"the frozen R2 submission is missing: {p}")
    h = hashlib.md5(p.read_bytes()).hexdigest()
    assert h == "89b0e4e2aef63ace4989fc0a44590ee5", f"R2 submission changed: md5 {h}"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    npass = nskip = 0
    fails = []
    for t in tests:
        try:
            r = t()
            if isinstance(r, str) and r.startswith("SKIP"):
                nskip += 1
                print(f"  SKIP  {t.__name__}  ({r[7:]})")
            else:
                npass += 1
                print(f"  ok    {t.__name__}")
        except Exception as e:                                   # noqa: BLE001
            fails.append((t.__name__, e))
            print(f"  FAIL  {t.__name__}\n        {type(e).__name__}: {e}")
    print(f"\n{npass} passed, {nskip} skipped, {len(fails)} failed")
    if fails:
        traceback.print_exception(type(fails[0][1]), fails[0][1], fails[0][1].__traceback__)
        sys.exit(1)


if __name__ == "__main__":
    main()
