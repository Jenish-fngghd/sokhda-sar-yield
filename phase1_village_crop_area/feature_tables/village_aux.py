"""Per-village aux signatures = mean of each aux band over DW-cropland pixels in the village.
These become the per-crop soft-blend features g_cv for the moonshot. Gap-free where optical exists;
villages with no cropland pixels get NaN -> filled with 0 (neutral).

Output results/cache/village_aux.csv: ID + mean of {s1_vh_aug,s1_vh_oct,s1_vhvv_oct,ndvi,l_hv_oct,
x_hh_aug, rice_dip=s1_vh_aug-s1_vh_jun}.
"""
import os
os.environ.pop("PROJ_LIB", None)
import numpy as np, rasterio, shapefile, csv
from rasterio.features import geometry_mask
from pyproj import Transformer

D = os.path.join(os.environ.get("R1_WORK", "."), "data/stack")
SHP = os.path.join(os.environ.get("R1_DATA", "anrf-aise-hack-2026-round-1-sar-crop-mapping-challenge"), "villages_clean/villages_clean.shp")
OUT = os.path.join(os.environ.get("R1_WORK", "."), "results/cache")
FEATS = ["s1_vh_aug", "s1_vh_oct", "s1_vhvv_oct", "ndvi", "l_hv_oct", "x_hh_aug"]

def main():
    with rasterio.open(os.path.join(D, "features.tif")) as ds:
        names = list(ds.descriptions)
        band = {n: ds.read(i + 1) for i, n in enumerate(names)}
        tf, H, W = ds.transform, ds.height, ds.width
    with rasterio.open(os.path.join(D, "cropmask.tif")) as ds:
        cn = list(ds.descriptions)
        dw = ds.read(cn.index("dw_crops") + 1); bu = ds.read(cn.index("dw_built") + 1)
        wc = ds.read(cn.index("worldcover") + 1)
    cropland = (dw > 0.30) & (bu < 0.35) & (wc != 80) & (wc != 50)
    band["rice_dip"] = band["s1_vh_aug"] - band["s1_vh_jun"]   # flooding: Aug-Jun VH
    # X-band (Capella) temporal features — different frequency than S1 C-band (sensor diversity)
    band["x_hh_jun"] = np.nanmean([band["x_hh_jun06"], band["x_hh_jun19"]], axis=0)
    band["x_hh_augjun"] = band["x_hh_aug"] - band["x_hh_jun"]   # X-band greenup
    band["x_hh_octaug"] = band["x_hh_oct"] - band["x_hh_aug"]   # X-band senescence/standing
    band["x_hh_octjun"] = band["x_hh_oct"] - band["x_hh_jun"]   # X-band season change
    cols = FEATS + ["rice_dip", "x_hh_jun", "x_hh_oct", "x_hh_augjun", "x_hh_octaug", "x_hh_octjun"]

    # denser aux (lever B): monthly NDVI Jun..Oct + S1 Jul/Sep, and phenology deltas
    dp = os.path.join(D, "denser_aux.tif")
    if os.path.exists(dp):
        with rasterio.open(dp) as ds:
            for i, n in enumerate(ds.descriptions):
                band[n] = ds.read(i + 1)
        # phenology signatures (S2 months = jun,sep,oct; jul/aug are monsoon-clouded)
        band["ndvi_sepoct"] = band["ndvi_oct"] - band["ndvi_sep"]    # late senescence vs standing
        band["ndvi_late"] = band["ndvi_oct"] - band["ndvi_jun"]      # still-green late (cotton/gnd)
        band["ndvi_greenup"] = band["ndvi_sep"] - band["ndvi_jun"]   # monsoon greenup
        band["s1_vh_julsep"] = band["s1_vh_sep"] - band["s1_vh_jul"] # mid-late canopy change
        cols = cols + ["ndvi_jun","ndvi_sep","ndvi_oct","ndvi_sepoct","ndvi_late","ndvi_greenup",
                       "s1_vh_jul","s1_vh_sep","s1_vh_julsep"]

    # texture (lever P1): within-village heterogeneity proxy (GLCM)
    txp = os.path.join(D, "texture.tif")
    if os.path.exists(txp):
        with rasterio.open(txp) as ds:
            for i, n in enumerate(ds.descriptions):
                band[n] = ds.read(i + 1)
        cols = cols + ["s1vh_contrast", "s1vh_var", "ndvi_contrast"]

    sf = shapefile.Reader(SHP); tfm = Transformer.from_crs(4326, 32643, always_xy=True)
    rows = []
    for shp, rec in zip(sf.shapes(), sf.records()):
        q = np.array(shp.points); x, y = tfm.transform(q[:, 0], q[:, 1])
        m = ~geometry_mask([{"type": "Polygon", "coordinates": [list(zip(x, y))]}], (H, W), tf) & cropland
        vals = []
        for c in cols:
            v = band[c][m]; v = v[np.isfinite(v)]
            vals.append(float(v.mean()) if v.size else np.nan)
        rows.append([rec["ID"]] + vals)

    # fill NaN per-column with column mean (neutral)
    arr = np.array([r[1:] for r in rows], float)
    for j in range(arr.shape[1]):
        col = arr[:, j]; col[~np.isfinite(col)] = np.nanmean(col[np.isfinite(col)])
    p = os.path.join(OUT, "village_aux.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ID"] + cols)
        for i, r in enumerate(rows):
            w.writerow([r[0]] + [round(float(x), 4) for x in arr[i]])
    print("wrote", p, "| villages", len(rows), "feats", cols)

if __name__ == "__main__":
    main()
