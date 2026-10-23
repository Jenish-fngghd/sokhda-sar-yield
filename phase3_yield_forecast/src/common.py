"""Shared paths, scene metadata, run ledger and guards for the Round 3 pipeline.

Everything that names a path, a date or an AOI lives here and nowhere else.
Round 2 measured its own portability debt at the end: 20 of 33 files hardcoded the
acquisition dates and 12 of 33 hardcoded the village name, which turned "run it on
six dates" into a refactor. So DATES is *scanned from the delivered data*, not
written down, and every scene constant is read from the vendor's _extended.json.
"""
import os

# PROJ_LIB left over from another GDAL install silently breaks pyproj's CRS lookups.
# Must happen before rasterio/pyproj are imported anywhere. [R2 guard, cost real time]
os.environ.pop("PROJ_LIB", None)

import json
import re
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("AISE_DATA") or
            ROOT / "anrf-aise-hack-2-0-round-3-sar-crop-yield-forecasting")
R2 = Path(os.environ.get("AISE_R2") or ROOT.parent / "phase2_crop_type_health")  # read-only

VILLAGE_NAME = os.environ.get("AISE_VILLAGE", "Sokhda")
FARMS = Path(os.environ.get("AISE_FARMS") or
             DATA / "Farm_boundaries_shp" / "Farm_boundaries_shp" / f"{VILLAGE_NAME}_Farms.shp")
VILLAGE = Path(os.environ.get("AISE_VILLAGE_SHP") or
               DATA / "Village_Shp" / "Village_Shp" / f"{VILLAGE_NAME}_Village.shp")

RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"          # not "aux": reserved Windows device name
AUX = ROOT / "data_aux"
INTERNAL = ROOT / "internal"
for _d in (RESULTS, CACHE, FIGURES, TABLES, AUX, INTERNAL):
    _d.mkdir(parents=True, exist_ok=True)
LEDGER = RESULTS / "log.jsonl"

CROPS = ["Rice", "Cotton", "Maize", "Bajra", "Groundnut"]  # exact submission spellings


def _scan_scenes():
    """{YYYYMMDD: scene_stem} for every delivered SLC, scanned not hardcoded.

    A scene counts only when the date appears in BOTH the folder name and the file
    basename. Both failure modes are live in this archive: the 20250619 folder holds
    a byte-identical copy of the 20250606 SLC, so folder-only matching returns June 6
    data as "June 19", and basename-only matching finds the June 6 name in two folders.
    Requiring agreement resolves each date to exactly one file. [R2 A3/E7]
    """
    out = {}
    for tif in sorted(DATA.glob("CAPELLA_*/CAPELLA_*SLC_HH_*.tif")):
        m = re.search(r"SLC_HH_(\d{8})", tif.name)
        if not m or m.group(1) not in tif.parent.name:
            continue
        d = m.group(1)
        if d in out:
            raise RuntimeError(f"{d}: two scenes match, {out[d]} and {tif}")
        out[d] = tif
    if not out:
        raise FileNotFoundError(f"no SLCs under {DATA}")
    return out


_SCENES = _scan_scenes()
DATES = sorted(_SCENES)


def slc_path(date: str) -> Path:
    return _SCENES[date]


def meta_path(date: str) -> Path:
    return slc_path(date).with_name(slc_path(date).stem + "_extended.json")


@lru_cache(maxsize=None)
def meta(date: str) -> dict:
    """The vendor's _extended.json. Source of truth for every scene constant."""
    return json.loads(meta_path(date).read_text(encoding="utf8"))


def scene(date: str) -> dict:
    """The handful of scene constants everything downstream needs, read from metadata.

    Nothing here is transcribed from a table -- a transcribed constant is a constant
    that silently stops matching the data.
    """
    j = meta(date)
    img, col = j["collect"]["image"], j["collect"]
    return {
        "date": date,
        "utc": col["start_timestamp"],
        "local": col["local_datetime"],
        "pointing": col["radar"]["pointing"],
        "direction": col["state"]["direction"],
        "scale_factor": img["scale_factor"],
        "nesz_peak": img["nesz_peak"],
        "enl": img["enl"],
        "rows": img["rows"],
        "columns": img["columns"],
        "incidence": img["image_geometry"].get("center_incidence_angle")
                     or img["center_pixel"].get("incidence_angle"),
        "center_frequency": col["radar"]["center_frequency"],
    }


def beta0(z, date):
    """Calibrated beta-nought from complex DN.

    beta0 = scale_factor**2 * |z|**2.  The SQUARE is the point: Round 2 shipped the
    unsquared form and it puts the darkest pixels 26-28 dB above the vendor's own
    declared noise floor, which is impossible. Capella's capella-reader reference
    (beta0_complex = SF * DN, so power carries SF**2) settles it; the nesz_peak test
    confirms it to 0.35 dB. [R2 post-r2 e1/e6]
    """
    import numpy as np
    sf = scene(date)["scale_factor"]
    z = np.asarray(z)
    return (sf * sf) * (z.real.astype("float64") ** 2 + z.imag.astype("float64") ** 2)


def db(x):
    """Linear power -> dB; non-positive maps to nan, never -inf.

    Single-look speckle produces exact zeros and -inf poisons every downstream mean.
    """
    import numpy as np
    x = np.asarray(x, dtype="float64")
    return np.where(x > 0, 10.0 * np.log10(np.where(x > 0, x, 1.0)), np.nan)


def log(stage: str, **fields) -> dict:
    """Append one record to the append-only run ledger and echo it."""
    rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage, **fields}
    with LEDGER.open("a", encoding="utf8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    print(f"[{rec['t']}] {stage}: " + " ".join(f"{k}={v}" for k, v in fields.items()))
    return rec


def _selfcheck():
    """Written before the processing code. In R2 this caught the duplicate-SLC trap
    on the first run, before a single pixel was read."""
    import numpy as np
    assert np.isnan(db(0.0)), "db(0) must be nan, not -inf"
    assert abs(db(10.0) - 10.0) < 1e-12
    assert FARMS.exists() and VILLAGE.exists(), "shapefiles missing"
    assert len(DATES) == 6, f"expected 6 scenes, scanned {len(DATES)}: {DATES}"

    # the duplicate trap: the June-19 folder also holds the June-6 SLC.
    assert slc_path("20250619").name.startswith("CAPELLA_C14_SM_SLC_HH_20250619")
    assert slc_path("20250606").parent.name.startswith("CAPELLA_C14_SM_SLC_HH_20250606")
    assert slc_path("20250619") != slc_path("20250606")
    dup = slc_path("20250619").parent / slc_path("20250606").name
    assert dup.exists(), "the duplicate is gone -- re-check the archive, not this test"

    # beta0 must square the scale factor: a 1+0j DN under SF=s must give s**2.
    s = scene("20250606")["scale_factor"]
    assert abs(float(beta0(np.array([1 + 0j]), "20250606")[0]) - s * s) < 1e-24

    for d in DATES:
        sc = scene(d)
        assert sc["enl"] == 1.0 and sc["nesz_peak"] < 0
        assert d in slc_path(d).name and d in slc_path(d).parent.name
    print(f"common.py self-check OK -- {len(DATES)} scenes resolved, duplicate trap "
          f"guarded, SF-squared calibration asserted")


if __name__ == "__main__":
    _selfcheck()
    for d in DATES:
        s = scene(d)
        print(f"  {d}  {s['local'][11:16]} IST  {s['pointing']:>5}-look  "
              f"inc={s['incidence']}  SF={s['scale_factor']:.8f}  NESZ={s['nesz_peak']:.2f} dB")
