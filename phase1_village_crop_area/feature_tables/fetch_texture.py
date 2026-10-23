"""Priority-1 lever: within-village TEXTURE (GLCM) — a village-level proxy for within-village crop-mix
heterogeneity (the residual the scalar model can't otherwise reach).
  - S1 VH August GLCM contrast + variance (radar texture of canopy structure)
  - S2 October NDVI GLCM contrast (optical texture)
Output data/stack/texture.tif (20m, EPSG:32643).
"""
import os
EE_PROJECT = os.environ.get("EE_PROJECT", "REPLACE_WITH_YOUR_GEE_CLOUD_PROJECT")
OUT = os.path.join(os.environ.get("R1_WORK", "."), "data/stack")
import ee, geemap
ee.Initialize(project=EE_PROJECT)
AOI = ee.Geometry.Rectangle([72.99, 22.25, 73.27, 22.49])
CRS, SCALE = "EPSG:32643", 20

# S1 VH Aug (glcmTexture needs integer input -> scale+offset to positive int)
s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(AOI)
      .filter(ee.Filter.eq("instrumentMode", "IW"))
      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
      .filterDate("2025-08-01", "2025-08-31").select("VH").median())
s1t = s1.multiply(100).add(3000).toInt().glcmTexture(size=3)
s1_contrast = s1t.select("VH_contrast").rename("s1vh_contrast")
s1_var = s1t.select("VH_var").rename("s1vh_var")

# S2 NDVI Oct
def mask(img):
    scl = img.select("SCL"); keep = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return img.normalizedDifference(["B8", "B4"]).rename("ndvi").updateMask(keep)
s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(AOI).filterDate("2025-10-01","2025-10-31").map(mask).median()
s2t = s2.multiply(100).add(100).toInt().glcmTexture(size=3)
s2_contrast = s2t.select("ndvi_contrast").rename("ndvi_contrast")

img = ee.Image.cat([s1_contrast, s1_var, s2_contrast]).clip(AOI).toFloat()
p = os.path.join(OUT, "texture.tif")
geemap.download_ee_image(img, filename=p, scale=SCALE, crs=CRS, region=AOI, dtype="float32", max_tile_size=8)
print("wrote", p)
