"""Rebuild per-village cropland cache using a better cropland mask (Dynamic World / WorldCereal /
WorldCover) instead of the crude Capella dB threshold. Writes results/village_cropland_v2.csv.

New cropland pixel = Capella-valid AND cropland-positive AND not built/water. Coverage still measured
from Capella-valid (the aux masks are gap-free, so 'coverage' = SAR observability, unchanged concept).
"""
import os
os.environ.pop("PROJ_LIB", None)
import numpy as np, rasterio, shapefile, csv
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling
from pyproj import Transformer

D = os.path.join(os.environ.get("R1_WORK", "."), "data/stack")
SHP = os.path.join(os.environ.get("R1_DATA", "anrf-aise-hack-2026-round-1-sar-crop-mapping-challenge"), "villages_clean/villages_clean.shp")
OUT = os.path.join(os.environ.get("R1_WORK", "."), "results")
PX_HA = 0.04
# thresholds
DW_CROP_MIN = 0.30                 # Dynamic World crop probability to count as cropland
DW_BUILT_MAX = 0.35                # exclude built-up
WCOV_WATER, WCOV_BUILT = 80, 50    # WorldCover class codes

def onref(path, band, ref_shape, ref_tf, ref_crs, resamp=Resampling.bilinear):
    with rasterio.open(path) as ds:
        out = np.full(ref_shape, np.nan, np.float32)
        reproject(ds.read(band), out, src_crs=ds.crs, src_transform=ds.transform,
                  dst_crs=ref_crs, dst_transform=ref_tf, resampling=resamp)
        return out

def main():
    feat = os.path.join(D, "features.tif")
    with rasterio.open(feat) as ds:
        tf, H, W, crs = ds.transform, ds.height, ds.width, ds.crs
        names = list(ds.descriptions)
        xhh = np.nanmean([ds.read(names.index(b) + 1) for b in
                          ["x_hh_jun06", "x_hh_jun19", "x_hh_aug", "x_hh_oct"]], axis=0)
        s1_oct = ds.read(names.index("s1_vv_oct") + 1)
    valid = np.isfinite(xhh)

    cm = os.path.join(D, "cropmask.tif")
    with rasterio.open(cm) as ds:
        cn = list(ds.descriptions)
        dw_crops = ds.read(cn.index("dw_crops") + 1)
        dw_built = ds.read(cn.index("dw_built") + 1)
        wcov = ds.read(cn.index("worldcover") + 1)
    # Dynamic World is GAP-FREE (optical, all villages) -> NO Capella-valid gate. Measures cropland
    # even for the 18 SAR-uncovered villages, dissolving the coverage-fill guess entirely.
    cropland = (dw_crops > DW_CROP_MIN) & (dw_built < DW_BUILT_MAX) \
               & (wcov != WCOV_WATER) & (wcov != WCOV_BUILT)
    print(f"DW cropland px {int(cropland.sum())} (gap-free, all villages)")

    sf = shapefile.Reader(SHP); tfm = Transformer.from_crs(4326, 32643, always_xy=True)
    rows = []
    for shp, rec in zip(sf.shapes(), sf.records()):
        q = np.array(shp.points); x, y = tfm.transform(q[:, 0], q[:, 1])
        geom = {"type": "Polygon", "coordinates": [list(zip(x, y))]}
        try:
            inside = ~geometry_mask([geom], (H, W), tf, invert=False)
        except Exception:
            inside = np.zeros((H, W), bool)
        poly = inside.sum()
        cov = (inside & valid).sum() / poly if poly else 0.0
        crop_ha = (inside & cropland).sum() * PX_HA
        vill_ha = poly * PX_HA
        rows.append((rec["ID"], round(cov, 2), round(crop_ha, 2), round(vill_ha, 2)))

    p = os.path.join(OUT, "village_cropland_v3.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ID", "cov", "cropland_ha", "village_ha"])
        for r in sorted(rows): w.writerow(r)
    print("wrote", p)

if __name__ == "__main__":
    main()
