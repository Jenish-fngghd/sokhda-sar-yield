"""Lever 3: fresh texture metrics/scales. GLCM entropy (disorder, not contrast) + 7x7 big window.
On NDVI-Oct + S1-VH-Aug (the two that worked). Output texture2.tif (20m, EPSG:32643)."""
import os
EE_PROJECT = os.environ.get("EE_PROJECT", "REPLACE_WITH_YOUR_GEE_CLOUD_PROJECT")
OUT = os.path.join(os.environ.get("R1_WORK", "."), "data/stack")
import ee, geemap
ee.Initialize(project=EE_PROJECT)
AOI = ee.Geometry.Rectangle([72.99, 22.25, 73.27, 22.49])
CRS, SCALE = "EPSG:32643", 20

def ndvi_oct():
    def m(img):
        scl=img.select("SCL");k=scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return img.normalizedDifference(["B8","B4"]).rename("nd").updateMask(k)
    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(AOI).filterDate("2025-10-01","2025-10-31").map(m).median()
s1a=(ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(AOI).filter(ee.Filter.eq("instrumentMode","IW"))
     .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VH")).filterDate("2025-08-01","2025-08-31").select("VH").median())

nd_i = ndvi_oct().multiply(100).add(100).toInt()
s1_i = s1a.multiply(100).add(3000).toInt()
# entropy (3x3) + big window contrast (7x7)
nd3 = nd_i.glcmTexture(size=3); nd7 = nd_i.glcmTexture(size=7)
s13 = s1_i.glcmTexture(size=3); s17 = s1_i.glcmTexture(size=7)
img = ee.Image.cat([
    nd3.select("nd_ent").rename("ndvi_ent"),
    nd7.select("nd_contrast").rename("ndvi_contrast_w7"),
    s13.select("VH_ent").rename("s1vh_ent"),
    s17.select("VH_contrast").rename("s1vh_contrast_w7"),
]).clip(AOI).toFloat()
geemap.download_ee_image(img, filename=os.path.join(OUT,"texture2.tif"), scale=SCALE, crs=CRS, region=AOI, dtype="float32", max_tile_size=8)
print("wrote texture2.tif")
