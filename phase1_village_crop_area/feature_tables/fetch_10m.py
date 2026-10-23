"""Lever 1: refetch the key layers at 10m (vs 20m) for sharper cropland + texture on small Indian fields.
Only the layers that matter for the current champion: DW cropmask, S1 VH (aug/oct/jul/sep), S2 NDVI
(jun/sep/oct), X-band is local (Capella, resample separately). Writes *_10m.tif.
RAM: 10m = 4x pixels of 20m -> use max_tile_size=4, float32.
"""
import os
EE_PROJECT = os.environ.get("EE_PROJECT", "REPLACE_WITH_YOUR_GEE_CLOUD_PROJECT")
OUT = os.path.join(os.environ.get("R1_WORK", "."), "data/stack")
import ee, geemap
ee.Initialize(project=EE_PROJECT)
AOI = ee.Geometry.Rectangle([72.99, 22.25, 73.27, 22.49])
CRS, SCALE = "EPSG:32643", 10

def dl(img, name):
    p = os.path.join(OUT, name + "_10m.tif")
    geemap.download_ee_image(img.clip(AOI).toFloat(), filename=p, scale=SCALE, crs=CRS,
                             region=AOI, dtype="float32", max_tile_size=4)
    print("wrote", p)

# cropmask (DW crops/built + WorldCover)
dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(AOI).filterDate("2025-06-01","2025-10-31")
cm = ee.Image.cat([dw.select("crops").median().rename("dw_crops"),
                   dw.select("built").median().rename("dw_built"),
                   ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("worldcover")])
dl(cm, "cropmask")

# S1 VH months
def s1vh(s,e,lab):
    c=(ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(AOI).filter(ee.Filter.eq("instrumentMode","IW"))
       .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VH")).filterDate(s,e))
    return c.select("VH").median().rename(f"s1_vh_{lab}")
S1M={"jun":("2025-06-01","2025-06-30"),"jul":("2025-07-01","2025-07-31"),"aug":("2025-08-01","2025-08-31"),
     "sep":("2025-09-01","2025-09-30"),"oct":("2025-10-01","2025-10-31")}
dl(ee.Image.cat([s1vh(*S1M[m],m) for m in S1M]), "s1vh")

# S2 NDVI months + VV for cross-pol ratio
def ndvi(s,e,lab):
    def mask(img):
        scl=img.select("SCL"); k=scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return img.normalizedDifference(["B8","B4"]).rename("ndvi").updateMask(k)
    c=ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(AOI).filterDate(s,e).map(mask)
    return ee.Image(ee.Algorithms.If(c.size().gt(0),c.median(),ee.Image.constant(0).updateMask(ee.Image.constant(0)))).rename(f"ndvi_{lab}")
def s1ratio(s,e):
    c=(ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(AOI).filter(ee.Filter.eq("instrumentMode","IW"))
       .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VV"))
       .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VH")).filterDate(s,e))
    return c.select("VH").median().subtract(c.select("VV").median()).rename("s1_vhvv_oct")
dl(ee.Image.cat([ndvi(*S1M["jun"],"jun"),ndvi(*S1M["sep"],"sep"),ndvi(*S1M["oct"],"oct"),
                 s1ratio(*S1M["oct"])]), "s2ndvi")

# texture (GLCM) at 10m
s1a=(ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(AOI).filter(ee.Filter.eq("instrumentMode","IW"))
     .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VH")).filterDate("2025-08-01","2025-08-31").select("VH").median())
s1t=s1a.multiply(100).add(3000).toInt().glcmTexture(size=3)
s2o=ndvi(*S1M["oct"],"oct").rename("nd").multiply(100).add(100).toInt().glcmTexture(size=3)
dl(ee.Image.cat([s1t.select("VH_contrast").rename("s1vh_contrast"),
                 s1t.select("VH_var").rename("s1vh_var"),
                 s2o.select("nd_contrast").rename("ndvi_contrast")]), "texture")
print("DONE 10m")
