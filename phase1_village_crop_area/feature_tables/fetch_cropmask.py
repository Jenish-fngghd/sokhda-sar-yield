"""Fetch free cropland-mask layers over AOI (20m, same grid) to replace crude dB-threshold cropland:
  - Dynamic World: median 'crops' + 'built' probability over Kharif 2025 (Jun-Oct)
  - ESA WorldCereal 2021: temporary-crop cropland (annual)
  - ESA WorldCover 2021: class map (40=cropland, 50=built, 80=water)
Outputs data/stack/cropmask.tif (bands: dw_crops, dw_built, worldcereal, worldcover).
"""
import os
EE_PROJECT = os.environ.get("EE_PROJECT", "REPLACE_WITH_YOUR_GEE_CLOUD_PROJECT")
OUT = os.path.join(os.environ.get("R1_WORK", "."), "data/stack")
import ee, geemap
ee.Initialize(project=EE_PROJECT)
AOI = ee.Geometry.Rectangle([72.99, 22.25, 73.27, 22.49])
CRS, SCALE = "EPSG:32643", 20

def export(img, name):
    p = os.path.join(OUT, name + ".tif")
    geemap.download_ee_image(img.clip(AOI).toFloat(), filename=p, scale=SCALE, crs=CRS,
                             region=AOI, dtype="float32", max_tile_size=8)
    print("wrote", p)

dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(AOI)
      .filterDate("2025-06-01", "2025-10-31"))
dw_crops = dw.select("crops").median().rename("dw_crops")
dw_built = dw.select("built").median().rename("dw_built")
# WorldCereal 2021 temporary crops (classification=100 -> cropland). Mosaic over AOI.
try:
    wc = (ee.ImageCollection("ESA/WorldCereal/2021/MODELS/v100")
          .filter(ee.Filter.eq("product", "temporarycrops")).select("classification")
          .mosaic().rename("worldcereal"))
except Exception:
    wc = ee.Image(0).rename("worldcereal")
wcov = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("worldcover")

export(ee.Image.cat([dw_crops, dw_built, wc, wcov]), "cropmask")
