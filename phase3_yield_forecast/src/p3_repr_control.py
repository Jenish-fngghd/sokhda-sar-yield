"""T21b: the control that decides whether T21's offset is about Sokhda or about definitions.

T21 measures Sokhda over SURVEYED PARCEL POLYGONS and its neighbours over WORLDCOVER-CLASSIFIED
CROPLAND. Those are not the same object. A land-cover classifier's "cropland" includes margins,
tracks, bunds and fallow, all of which sit darker than a cultivated field -- so comparing
parcels against classified cropland can manufacture a positive offset with no help from Sokhda
being unusual at all.

This re-measures Sokhda THE SAME WAY AS ITS NEIGHBOURS: WorldCover cropland pixels inside
Sokhda's footprint, parcel polygons ignored. Both sides then come from one definition.

    offset SURVIVES  ->  the asymmetry is not the cause; Sokhda really does differ
    offset COLLAPSES ->  T21 measured our own mask convention and is void

Run:  py -3.12 -u src/p3_repr_control.py
"""
import os, sys
os.environ.pop("PROJ_LIB", None)
from pathlib import Path
import numpy as np, rasterio, geopandas as gpd
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FARMS, log
import p3_representative as T
T.MAX_SCENES = 1   # the control compares three quantities WITHIN the same scenes, so fewer suffice

def main():
    fa = gpd.read_file(FARMS)
    fb = fa.to_crs(4326).total_bounds
    box = [fb[0]-T.PAD, fb[1]-T.PAD, fb[2]+T.PAD, fb[3]+T.PAD]
    poly = {"type":"Polygon","coordinates":[[[box[0],box[1]],[box[2],box[1]],
            [box[2],box[3]],[box[0],box[3]],[box[0],box[1]]]]}
    cen = {"type":"Point","coordinates":[float((fb[0]+fb[2])/2), float((fb[1]+fb[3])/2)]}
    ref = T.search("sentinel-1-rtc", cen, "2025-08-01T00:00:00Z/2025-08-31T00:00:00Z", 1)[0]
    with rasterio.open(T.sign(ref["assets"]["vh"]["href"])) as ds:
        win = from_bounds(*transform_bounds("EPSG:4326", ds.crs, *box), ds.transform)
        H, W = int(win.height)//T.DECIM, int(win.width)//T.DECIM
        gcrs, gtr = ds.crs, ds.window_transform(win) * rasterio.Affine.scale(T.DECIM)
    crop = np.zeros((H, W), "uint8")
    for f in [x for x in T.search("esa-worldcover", poly) if "2021" in x["id"]]:
        with rasterio.open(T.sign(f["assets"]["map"]["href"])) as ds:
            w2 = from_bounds(*transform_bounds("EPSG:4326", ds.crs, *box), ds.transform)
            src = ds.read(1, window=w2, boundless=True, fill_value=0)
            dst = np.zeros((H, W), "uint8")
            reproject(source=src, destination=dst, src_transform=ds.window_transform(w2),
                      src_crs=ds.crs, dst_transform=gtr, dst_crs=gcrs,
                      resampling=Resampling.nearest)
        crop = np.maximum(crop, (dst == T.CROPLAND).astype("uint8"))
    sok = rasterize(((g,1) for g in fa.to_crs(gcrs).geometry), out_shape=(H,W),
                    transform=gtr, fill=0, dtype="uint8").astype(bool)
    # Sokhda's footprint = convex hull of the parcels, so "cropland inside Sokhda" is defined
    # by WorldCover exactly as the neighbour blocks are.
    hull = rasterize(((fa.to_crs(gcrs).union_all().convex_hull, 1),), out_shape=(H,W),
                     transform=gtr, fill=0, dtype="uint8").astype(bool)
    sok_wc = hull & crop.astype(bool)
    print(f"Sokhda parcels {int(sok.sum())} px | WorldCover-cropland in hull {int(sok_wc.sum())} px"
          f" | overlap {int((sok & sok_wc).sum())} px")

    for year in (2025,):
        fs = [f for f in T.search("sentinel-1-rtc", cen,
              f"{year}-06-01T00:00:00Z/{year}-09-30T00:00:00Z")
              if (f["properties"].get("sat:orbit_state"),
                  f["properties"].get("sat:relative_orbit")) == T.TRACK]
        fs = sorted(fs, key=lambda f: f["properties"]["datetime"])
        fs = fs[::max(1, len(fs)//T.MAX_SCENES)][:T.MAX_SCENES]
        acc = np.zeros((H,W)); cnt = np.zeros((H,W))
        for f in fs:
            with rasterio.open(T.sign(f["assets"]["vh"]["href"])) as ds:
                w2 = from_bounds(*transform_bounds("EPSG:4326", ds.crs, *box), ds.transform)
                src = ds.read(1, window=w2, boundless=True, fill_value=0).astype("float32")
                dst = np.full((H,W), np.nan, "float32")
                reproject(source=src, destination=dst, src_transform=ds.window_transform(w2),
                          src_crs=ds.crs, dst_transform=gtr, dst_crs=gcrs,
                          resampling=Resampling.average, src_nodata=0, dst_nodata=np.nan)
            g = np.isfinite(dst) & (dst > 0); acc[g] += dst[g]; cnt[g] += 1
        m = np.where(cnt > 0, acc/np.maximum(cnt,1), np.nan)
        a = 10*np.log10(float(np.nanmean(m[sok])))
        b = 10*np.log10(float(np.nanmean(m[sok_wc])))
        outside = crop.astype(bool) & ~hull & np.isfinite(m)
        c = 10*np.log10(float(np.nanmean(m[outside])))
        print(f"  {year}  parcels {a:+6.2f}  WC-cropland-in-hull {b:+6.2f}  "
              f"neighbours(WC) {c:+6.2f}   offset_parcels {a-c:+5.2f}  offset_like4like {b-c:+5.2f}")
        log("p3_reprctl.year", year=year, parcels_db=round(a,3), sokhda_wc_db=round(b,3),
            neighbour_wc_db=round(c,3), offset_parcels_db=round(a-c,3),
            offset_like_for_like_db=round(b-c,3),
            definition_effect_db=round(a-b,3))

if __name__ == "__main__":
    main()
