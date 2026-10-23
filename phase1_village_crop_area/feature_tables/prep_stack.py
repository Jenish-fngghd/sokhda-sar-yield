"""Step 1: Capella SLC (4 dates, HH) -> calibrated, geocoded, multi-looked dB stack in UTM 43N.

Per scene:  complex SLC -> beta0 intensity (|z*scale|^2) -> multilook (speckle down + shrink)
            -> sigma0 = beta0 * sin(incidence) -> geocode via 225 GCPs -> dB.
Output:     data/stack/capella_db_stack.tif  (4 bands float32 dB, EPSG:32643, 10 m)
            + quicklook PNG.

ponytail: uses per-scene CENTER incidence angle (not per-pixel local) for beta0->sigma0.
          Fine for flat Vadodara; upgrade to DEM local-incidence terrain flattening only if
          the leaderboard says radiometry is the bottleneck.
"""
import os, glob, json
os.environ.pop("PROJ_LIB", None)
import numpy as np, rasterio
from rasterio.control import GroundControlPoint
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import shapefile
from pyproj import Transformer
from skimage.measure import block_reduce

BASE = os.environ.get("R1_DATA", "anrf-aise-hack-2026-round-1-sar-crop-mapping-challenge")
OUT = os.path.join(os.environ.get("R1_WORK", "."), "data/stack")
os.makedirs(OUT, exist_ok=True)
DATES = ["20250606", "20250619", "20250814", "20251013"]
AZ_LOOK, RG_LOOK = 14, 9   # -> ~10 m pixels (azimuth 0.74 m, range 1.07 m)
PX = 10.0                  # output grid pixel size (m), UTM 43N

def out_grid():
    """Common output grid = village bbox (UTM 43N) buffered 2 km, at PX."""
    sf = shapefile.Reader(BASE + "/villages_clean/villages_clean.shp")
    tf = Transformer.from_crs(4326, 32643, always_xy=True)
    xs, ys = [], []
    for shp in sf.shapes():
        p = np.array(shp.points); x, y = tf.transform(p[:, 0], p[:, 1]); xs += list(x); ys += list(y)
    L, R = min(xs) - 2000, max(xs) + 2000
    B, T = min(ys) - 2000, max(ys) + 2000
    W, H = int((R - L) / PX), int((T - B) / PX)
    return from_bounds(L, B, R, T, W, H), W, H

def scene_sigma0(slc_path, ext_path):
    ext = json.load(open(ext_path))["collect"]["image"]
    scale = ext["scale_factor"]
    inc = np.deg2rad(ext["center_pixel"]["incidence_angle"])
    with rasterio.open(slc_path) as ds:
        z = ds.read(1)                       # complex_int16 -> numpy complex
        gcps, gcp_crs = ds.gcps
    beta0 = (z.real.astype(np.float32) ** 2 + z.imag.astype(np.float32) ** 2) * (scale ** 2)
    beta0 = block_reduce(beta0, (AZ_LOOK, RG_LOOK), np.mean)   # multilook
    sigma0 = beta0 * np.sin(inc)             # beta0 -> sigma0
    # rescale GCP row/col to the multilooked grid
    sg = [GroundControlPoint(row=g.row / AZ_LOOK, col=g.col / RG_LOOK, x=g.x, y=g.y, z=g.z)
          for g in gcps]
    return sigma0.astype(np.float32), sg, gcp_crs

def main():
    transform, W, H = out_grid()
    stack = np.full((len(DATES), H, W), np.nan, np.float32)
    for i, d in enumerate(DATES):
        # SLC tif whose basename carries this date (Jun19 folder holds a stray 0606 tif -> match basename)
        cands = [p for p in glob.glob(BASE + "/*/*.tif")
                 if d in os.path.basename(p) and "SLC" in os.path.basename(p) and "preview" not in p]
        slc = cands[0]
        ext = glob.glob(os.path.dirname(slc) + f"/*{d}*extended.json")[0]
        sigma0, sg, gcp_crs = scene_sigma0(slc, ext)
        dst = np.full((H, W), np.nan, np.float32)
        reproject(sigma0, dst, src_crs=gcp_crs, gcps=sg,
                  dst_crs="EPSG:32643", dst_transform=transform,
                  src_nodata=0, dst_nodata=np.nan, resampling=Resampling.bilinear)
        with np.errstate(divide="ignore", invalid="ignore"):
            stack[i] = 10.0 * np.log10(dst)          # dB
        v = stack[i][np.isfinite(stack[i])]
        print(f"{d}: dB mean {v.mean():.1f} std {v.std():.1f} valid {v.size} ({100*v.size/(H*W):.0f}%)")

    prof = dict(driver="GTiff", height=H, width=W, count=len(DATES), dtype="float32",
                crs="EPSG:32643", transform=transform, nodata=np.nan, compress="deflate")
    outp = os.path.join(OUT, "capella_db_stack.tif")
    with rasterio.open(outp, "w", **prof) as ds:
        for i in range(len(DATES)):
            ds.write(stack[i], i + 1); ds.set_band_description(i + 1, DATES[i])
    print("wrote", outp, stack.shape)

    # quicklook
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 13))
    for a, i in zip(ax.ravel(), range(4)):
        s = stack[i]; lo, hi = np.nanpercentile(s, [2, 98])
        a.imshow(np.clip((s - lo) / (hi - lo), 0, 1), cmap="gray"); a.set_title(DATES[i]); a.axis("off")
    fig.suptitle("Capella sigma0 dB — geocoded UTM 43N 10 m (step-1 stack)")
    fig.tight_layout(); p = os.path.join(OUT, "stack_quicklook.png"); fig.savefig(p, dpi=100)
    print("wrote", p)

if __name__ == "__main__":
    main()
