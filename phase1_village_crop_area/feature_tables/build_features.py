"""Stage 2: fuse all free layers onto ONE 20 m grid (the Sentinel-1 grid).
Capella (10 m) resampled down; S1/ALOS-2/S2 native. Output data/stack/features.tif (multiband).

20 m keeps RAM低 (grid ~1348x1459). Capella 10 m detail only mattered for the mask, which we
recompute here at 20 m — fine for village-level area.
"""
import os
os.environ.pop("PROJ_LIB", None)
import numpy as np, rasterio
from rasterio.warp import reproject, Resampling

D = os.path.join(os.environ.get("R1_WORK", "."), "data/stack")
REF = os.path.join(D, "s1_vvvh_stack.tif")          # 20 m reference grid
AUX = ["s1_vvvh_stack.tif", "alos2_hhhv_stack.tif", "s2_oct_composite.tif"]
CAP = "capella_db_stack.tif"
CAP_NAMES = ["x_hh_jun06", "x_hh_jun19", "x_hh_aug", "x_hh_oct"]

def main():
    with rasterio.open(REF) as r:
        dst_crs, dst_tf, W, H = r.crs, r.transform, r.width, r.height
    bands, names = [], []

    # Capella -> resample to ref grid
    with rasterio.open(os.path.join(D, CAP)) as ds:
        for i in range(ds.count):
            out = np.full((H, W), np.nan, np.float32)
            reproject(ds.read(i + 1), out, src_crs=ds.crs, src_transform=ds.transform,
                      dst_crs=dst_crs, dst_transform=dst_tf, src_nodata=np.nan, dst_nodata=np.nan,
                      resampling=Resampling.average)
            bands.append(out); names.append(CAP_NAMES[i])

    # aux already on ref grid -> read straight
    for fn in AUX:
        with rasterio.open(os.path.join(D, fn)) as ds:
            for i in range(ds.count):
                bands.append(ds.read(i + 1).astype(np.float32))
                names.append(ds.descriptions[i] or f"{fn}_{i+1}")

    stack = np.stack(bands)
    prof = dict(driver="GTiff", height=H, width=W, count=len(bands), dtype="float32",
                crs=dst_crs, transform=dst_tf, nodata=np.nan, compress="deflate")
    outp = os.path.join(D, "features.tif")
    with rasterio.open(outp, "w", **prof) as ds:
        for i, (b, nm) in enumerate(zip(bands, names)):
            ds.write(b, i + 1); ds.set_band_description(i + 1, nm)
    print("wrote", outp, stack.shape)
    print("bands:", names)

if __name__ == "__main__":
    main()
