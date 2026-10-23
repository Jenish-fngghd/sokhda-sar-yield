"""Lever B: denser aux for finer phenology.
  - Sentinel-2 monthly NDVI, cloud-masked (Jun..Oct) -> 5 bands ndvi_jun..ndvi_oct
  - Sentinel-1 gap-fill months Jul, Sep (VV,VH) -> s1_vv_jul,s1_vh_jul,s1_vv_sep,s1_vh_sep
Output data/stack/denser_aux.tif (20m, EPSG:32643). Low RAM (float32, tiled).
"""
import os
EE_PROJECT = os.environ.get("EE_PROJECT", "REPLACE_WITH_YOUR_GEE_CLOUD_PROJECT")
OUT = os.path.join(os.environ.get("R1_WORK", "."), "data/stack")
import ee, geemap
ee.Initialize(project=EE_PROJECT)
AOI = ee.Geometry.Rectangle([72.99, 22.25, 73.27, 22.49])
CRS, SCALE = "EPSG:32643", 20
MONTHS = {"jun": ("2025-06-01", "2025-06-30"), "jul": ("2025-07-01", "2025-07-31"),
          "aug": ("2025-08-01", "2025-08-31"), "sep": ("2025-09-01", "2025-09-30"),
          "oct": ("2025-10-01", "2025-10-31")}
S2_MONTHS = ["jun", "sep", "oct"]           # monsoon jul/aug: no usable optical

def s2_ndvi_month(s, e, lab):
    def mask(img):
        scl = img.select("SCL")
        keep = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return img.normalizedDifference(["B8", "B4"]).rename("ndvi").updateMask(keep)
    c = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(AOI).filterDate(s, e).map(mask)
    # guard empty collection -> constant masked band, keeps structure
    return ee.Image(ee.Algorithms.If(c.size().gt(0), c.median(),
                    ee.Image.constant(0).updateMask(ee.Image.constant(0)))).rename(f"ndvi_{lab}")

def s1_month(s, e, lab):
    c = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(AOI)
         .filter(ee.Filter.eq("instrumentMode", "IW"))
         .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
         .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")).filterDate(s, e))
    return ee.Image.cat([c.select("VV").median().rename(f"s1_vv_{lab}"),
                         c.select("VH").median().rename(f"s1_vh_{lab}")])

bands = [s2_ndvi_month(*MONTHS[m], m) for m in S2_MONTHS]
bands += [s1_month(*MONTHS["jul"], "jul"), s1_month(*MONTHS["sep"], "sep")]
img = ee.Image.cat(bands).clip(AOI).toFloat()
p = os.path.join(OUT, "denser_aux.tif")
geemap.download_ee_image(img, filename=p, scale=SCALE, crs=CRS, region=AOI, dtype="float32", max_tile_size=8)
print("wrote", p)
