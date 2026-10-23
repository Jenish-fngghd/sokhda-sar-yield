"""Shared paths, run ledger, and guards for the R2 pipeline.

Every stage imports from here so paths and logging are defined in exactly one place.
See internal/MASTER_PLAN.md for what each stage does and why.
"""
import os

# PROJ_LIB left over from another GDAL install silently breaks pyproj's CRS lookups.
# Popping it must happen before rasterio/pyproj are imported anywhere. [PLAN.md guard]
os.environ.pop("PROJ_LIB", None)

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("AISE_DATA") or
            ROOT / "anrf-aise-hack-2-0-round-2-sar-crop-health-yield-estimation")

# The village is named in the shapefile basenames, so a different village needs different
# paths. Overridable by environment so a new AOI does not require editing this file under
# time pressure -- the defaults reproduce Sokhda exactly when nothing is set.
# ponytail: env vars, not a config file. Three values do not need a schema.
VILLAGE_NAME = os.environ.get("AISE_VILLAGE", "Sokhda")
FARMS = Path(os.environ.get("AISE_FARMS") or
             DATA / "Farm_boundaries_shp" / "Farm_boundaries_shp" / f"{VILLAGE_NAME}_Farms.shp")
VILLAGE = Path(os.environ.get("AISE_VILLAGE_SHP") or
               DATA / "Village_Shp" / "Village_Shp" / f"{VILLAGE_NAME}_Village.shp")
RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"
FIGURES = RESULTS / "figures"
# Secondary tables -- witness data, context, EDA/diagnostic exports -- kept out of
# results/'s root so the four primary outputs (submission.csv, farm_features.csv,
# d4_debug.csv, log.jsonl) are what a reader sees first.
# NOT named "aux": that is a reserved Windows device name (like con/nul/prn) and a
# directory called aux/ breaks Windows file APIs unpredictably -- git itself refused
# to open files inside one ("No such file or directory" despite the file existing).
RESULTS_AUX = RESULTS / "tables"
AUX = ROOT / "data_aux"
LEDGER = RESULTS / "log.jsonl"   # redirected below if the root is read-only

# Create the output tree, but tolerate a read-only root. On Kaggle the notebook runs
# with this repo mounted under /kaggle/input, which is READ-ONLY -- an unguarded mkdir
# raises at import time and the notebook dies on its first cell. "Notebook runs without
# errors" is a pass/fail rubric item, so this guard is load-bearing, not defensive
# habit. When the root is read-only, writes are redirected to a writable working dir.
def _ensure_writable(dirs):
    try:
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        return False
    except OSError:
        return True


_READONLY = _ensure_writable((RESULTS, CACHE, FIGURES, RESULTS_AUX, AUX))
if _READONLY:
    _work = Path(os.environ.get("KDSS_WORKDIR", "/kaggle/working"))
    if not _work.exists():
        _work = Path.cwd() / "_work"
    OUTPUT = _work
    OUTPUT.mkdir(parents=True, exist_ok=True)
else:
    OUTPUT = RESULTS

# Acquisition dates in order. Jun06/Jun19 are pre-monsoon bare soil, Aug14 peak
# vegetative, Oct13 post-monsoon. [RESEARCH_LOG B12]
DATES = ["20250606", "20250619", "20250814", "20251013"]

# Scene metadata read from *_extended.json at [A3]. Kept here so any stage can
# reason about geometry without re-parsing, but prep_r2 re-reads the JSON as the
# source of truth rather than trusting these.
INCIDENCE_DEG = {"20250606": 35.244, "20250619": 28.768,
                 "20250814": 28.692, "20251013": 31.528}
REF_INCIDENCE_DEG = 31.528  # October anchor -- the date the deliverable is "as of". [B8]

CROPS = ["Rice", "Cotton", "Maize", "Bajra", "Groundnut"]  # exact submission spellings [E1]


def slc_path(date: str) -> Path:
    """SLC tif for a date. Requires the date in BOTH the folder name and the basename.

    Neither test alone is sufficient, and both failure modes are live in this dataset:
      - folder only : the Jun-19 folder also contains a byte-identical copy of the
                      Jun-06 SLC, so globbing that folder can return the Jun-06 scene
      - basename only: that same duplicate means the Jun-06 basename matches in two
                      different folders
    Requiring agreement resolves each date to exactly one file. [RESEARCH_LOG A3, E7]
    """
    hits = [p for p in sorted(DATA.glob(f"CAPELLA_*/CAPELLA_*SLC_HH_{date}*.tif"))
            if date in p.name and date in p.parent.name]
    if len(hits) != 1:
        raise FileNotFoundError(f"{date}: expected 1 SLC tif, found {len(hits)}: {hits}")
    return hits[0]


def _ledger_path():
    """Ledger location, moved to the writable dir when the repo is mounted read-only."""
    return LEDGER if not _READONLY else OUTPUT / "log.jsonl"


def log(stage: str, **fields) -> dict:
    """Append one record to the run ledger and echo it.

    Immutable by construction: the ledger is append-only, so a rerun adds a row
    rather than overwriting the previous one. This is what makes a run
    reconstructable after the fact -- R1's lesson. [PLAN.md]
    """
    rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage, **fields}
    with _ledger_path().open("a", encoding="utf8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    print(f"[{rec['t']}] {stage}: " + " ".join(f"{k}={v}" for k, v in fields.items()))
    return rec


def db(x):
    """Linear power -> dB, with non-positive values mapped to nan rather than -inf.

    Speckle in single-look data produces exact zeros; letting those become -inf
    poisons every downstream mean.
    """
    import numpy as np
    x = np.asarray(x, dtype="float64")
    return np.where(x > 0, 10.0 * np.log10(np.where(x > 0, x, 1.0)), np.nan)


def farm_centroids():
    """Farm centroids in UTM 43N (EPSG:32643), row i = FID i+1.

    pyshp + pyproj only, deliberately no geopandas: this is used by the submission
    path and by the validation harness, and a dependency that is missing in one
    environment silently disables spatial imputation in the other.

    Area-weighted shoelace centroid summed over parts, so the 13 multipart farms
    land in the right place; degenerate rings fall back to the vertex mean.
    """
    import numpy as np
    import shapefile
    from pyproj import Transformer

    sf = shapefile.Reader(str(FARMS))
    tf = Transformer.from_crs(4326, 32643, always_xy=True)
    recs = sf.records()
    out = np.full((len(recs), 2), np.nan)
    for shp, rec in zip(sf.shapes(), recs):
        pts = np.asarray(shp.points, "float64")
        x, y = tf.transform(pts[:, 0], pts[:, 1])
        parts = list(shp.parts) + [len(x)]
        cx = cy = wsum = 0.0
        for a, b in zip(parts[:-1], parts[1:]):
            px, py = x[a:b], y[a:b]
            if len(px) < 3:
                continue
            cr = px * np.roll(py, -1) - np.roll(px, -1) * py
            A = cr.sum() / 2.0
            if abs(A) < 1e-9:
                continue
            cx += ((px + np.roll(px, -1)) * cr).sum() / (6 * A) * abs(A)
            cy += ((py + np.roll(py, -1)) * cr).sum() / (6 * A) * abs(A)
            wsum += abs(A)
        i = int(rec["FID"]) - 1
        out[i] = (cx / wsum, cy / wsum) if wsum > 0 else (x.mean(), y.mean())
    assert np.isfinite(out).all(), "every farm needs a centroid"
    return out


def _selfcheck():
    import numpy as np
    assert np.isnan(db(0.0)), "db(0) must be nan, not -inf"
    assert abs(db(1.0) - 0.0) < 1e-12
    assert abs(db(10.0) - 10.0) < 1e-12
    assert FARMS.exists(), f"farm shapefile missing: {FARMS}"
    assert VILLAGE.exists(), f"village shapefile missing: {VILLAGE}"
    # The 2.1 GB of SLCs are deliberately excluded from the Kaggle dataset -- FAST mode
    # reads the cached rasters instead -- so their absence is a SUPPORTED configuration,
    # not a failure. Checking them unconditionally made this self-check the first thing
    # that dies in the very environment the dataset was built for.
    if not any(DATA.glob("CAPELLA_*/CAPELLA_*SLC_HH_*.tif")):
        print("common.py self-check OK -- shapefiles present; SLCs absent, so scene "
              "resolution is skipped (this is FAST mode, the Kaggle configuration)")
        return
    for d in DATES:
        p = slc_path(d)
        assert d in p.name, f"{d} resolved to wrong scene {p.name}"
    # the stray-file trap: Jun-19's folder holds a duplicate Jun-06 SLC as well as
    # its own. Both dates must still resolve to the file in their OWN folder.
    assert slc_path("20250619").parent.name.startswith("CAPELLA_C14_SM_SLC_HH_20250619")
    assert slc_path("20250606").parent.name.startswith("CAPELLA_C14_SM_SLC_HH_20250606")
    assert slc_path("20250619") != slc_path("20250606")
    print("common.py self-check OK -- 4 scenes resolved by basename, shapefiles present")


if __name__ == "__main__":
    _selfcheck()
