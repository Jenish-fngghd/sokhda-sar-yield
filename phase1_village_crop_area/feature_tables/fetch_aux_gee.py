"""Step 2: fetch FREE auxiliary layers over the Vadodara AOI via Google Earth Engine,
on the same EPSG:32643 / 10 m grid as the Capella stack.

  - Sentinel-1 C-band VV/VH  (CORE  — crop-type signal HH can't give)
  - ALOS-2 PALSAR-2 L-band HH/HV (ADD — biomass axis: cotton/maize vs bajra/groundnut)
  - Sentinel-2 October composite (CONFIRM — clouds block Jul-Aug; Oct green = cotton/groundnut)

One-time setup (run in a terminal, NOT here — it's interactive):
    pip install earthengine-api geemap
    earthengine authenticate          # opens a browser once
    # set your Cloud project id below (EE now requires one)

Then:  py -3.12 src/fetch_aux_gee.py
Outputs land in data/stack/ as GeoTIFFs. Toggle layers with the flags at the bottom.
"""
import os
EE_PROJECT = os.environ.get("EE_PROJECT", "REPLACE_WITH_YOUR_GEE_CLOUD_PROJECT")
OUT = os.path.join(os.environ.get("R1_WORK", "."), "data/stack")
os.makedirs(OUT, exist_ok=True)

import ee, geemap
ee.Initialize(project=EE_PROJECT)

# AOI = village bbox (+ small buffer), lon/lat. Export CRS/scale match the Capella stack.
AOI = ee.Geometry.Rectangle([72.99, 22.25, 73.27, 22.49])
CRS, SCALE = "EPSG:32643", 20   # 20 m keeps RAM low; resampled to 10 m Capella grid in build_features
# Capella acquisition dates -> build a matched composite window around each
WINDOWS = {  # label: (start, end)
    "jun": ("2025-06-01", "2025-06-30"),
    "aug": ("2025-08-01", "2025-08-31"),
    "oct": ("2025-10-01", "2025-10-31"),
}

def export(img, name):
    path = os.path.join(OUT, name + ".tif")
    # tiled download (geedim), float32 + small tiles to fit low RAM
    geemap.download_ee_image(img.clip(AOI).toFloat(), filename=path, scale=SCALE, crs=CRS,
                             region=AOI, dtype="float32", max_tile_size=8)
    print("wrote", path)

def sentinel1():
    """Per-window VV, VH, VH/VV median composites (dB)."""
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(AOI).filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    bands = []
    for lab, (s, e) in WINDOWS.items():
        c = col.filterDate(s, e)
        vv = c.select("VV").median().rename(f"s1_vv_{lab}")
        vh = c.select("VH").median().rename(f"s1_vh_{lab}")
        ratio = vh.subtract(vv).rename(f"s1_vhvv_{lab}")   # dB difference = ratio
        bands += [vv, vh, ratio]
    export(ee.Image.cat(bands), "s1_vvvh_stack")

def alos2():
    """L-band HH/HV per window (values are in dB per JAXA L2.2 gamma0 convention)."""
    col = ee.ImageCollection("JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR").filterBounds(AOI)
    bands = []
    for lab, (s, e) in WINDOWS.items():
        c = col.filterDate(s, e)
        if c.size().getInfo() == 0:
            print(f"ALOS-2: no scenes in {lab} window"); continue
        hh = c.select("HH").median().rename(f"l_hh_{lab}")
        hv = c.select("HV").median().rename(f"l_hv_{lab}")
        bands += [hh, hv]
    if bands:
        export(ee.Image.cat(bands), "alos2_hhhv_stack")

def sentinel2_oct():
    """October cloud-masked 6-band composite + NDVI (green-standing = cotton/groundnut)."""
    def mask(img):
        scl = img.select("SCL")
        keep = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))  # shadow/cloud/cirrus
        return img.updateMask(keep)
    c = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
         .filterBounds(AOI).filterDate("2025-10-01", "2025-10-31")
         .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)).map(mask))
    b = c.median().select(["B2", "B3", "B4", "B8", "B11", "B12"],
                          ["blue", "green", "red", "nir", "swir1", "swir2"])
    ndvi = b.normalizedDifference(["nir", "red"]).rename("ndvi")
    export(b.addBands(ndvi), "s2_oct_composite")

if __name__ == "__main__":
    sentinel1()      # CORE
    alos2()          # ADD  (comment out if not wanted)
    sentinel2_oct()  # CONFIRM
