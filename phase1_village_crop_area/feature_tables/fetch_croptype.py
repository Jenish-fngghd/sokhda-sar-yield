"""Fetch REAL crop-type priors per village (untried levers beyond scalar features):
  - WorldCereal 2021 maize fraction (direct crop-type product, 10m)  -> maize direction
  - WorldCereal temporarycrops (active-crop mask) fraction
  - Sentinel-1 rice-flood index: min VH over Jun-Sep (transplant/flood = very low VH) then greenup.
    Rice's flood signature is SARs most reliable crop signal -> strong rice direction.
Output results/cache/village_croptype.csv: ID, wc_maize, wc_tempcrop, wc_wcereal, wc_scereal, rice_flood.
"""
import os, csv
os.environ.pop("PROJ_LIB", None)
import ee, numpy as np, shapefile
from pyproj import CRS

ee.Initialize(project=os.environ.get("EE_PROJECT", "REPLACE_WITH_YOUR_GEE_CLOUD_PROJECT"))
SHP = os.path.join(os.environ.get("R1_DATA", "anrf-aise-hack-2026-round-1-sar-crop-mapping-challenge"), "villages_clean/villages_clean.shp")
OUT = os.path.join(os.environ.get("R1_WORK", "."), "results/cache")

def village_fc():
    sf = shapefile.Reader(SHP)
    feats = []
    for shp, rec in zip(sf.shapes(), sf.records()):
        pts = [[float(x), float(y)] for x, y in shp.points]
        feats.append(ee.Feature(ee.Geometry.Polygon([pts]), {"ID": int(rec["ID"])}))
    return ee.FeatureCollection(feats)

def wc_product(prod):
    ic = ee.ImageCollection("ESA/WorldCereal/2021/MODELS/v100").filter(ee.Filter.eq("product", prod))
    return ic.select("classification").mosaic().unmask(0).divide(100.0)  # 0/1 fraction

def main():
    fc = village_fc()
    maize = wc_product("maize").rename("wc_maize")
    temp = wc_product("temporarycrops").rename("wc_tempcrop")
    wcer = wc_product("wintercereals").rename("wc_wcereal")
    scer = wc_product("springcereals").rename("wc_scereal")

    # Sentinel-1 rice-flood: low VH in flooding window vs Oct standing crop
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(fc).filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")).select("VH"))
    def mn(a, b): return s1.filterDate(f"2025-{a}", f"2025-{b}").min()
    vh_flood = mn("06-15", "08-15")   # transplant/flood window min VH (deep low = water)
    vh_oct = mn("10-01", "10-31").rename("vhoct")
    rice_flood = vh_oct.subtract(vh_flood).rename("rice_flood")  # big rise = flooded->grown = rice

    img = maize.addBands([temp, wcer, scer, rice_flood])
    stats = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=20).getInfo()
    rows = []
    for f in stats["features"]:
        p = f["properties"]
        rows.append([p["ID"], p.get("wc_maize"), p.get("wc_tempcrop"), p.get("wc_wcereal"),
                     p.get("wc_scereal"), p.get("rice_flood")])
    rows.sort()
    cols = ["wc_maize", "wc_tempcrop", "wc_wcereal", "wc_scereal", "rice_flood"]
    with open(os.path.join(OUT, "village_croptype.csv"), "w", newline="") as fp:
        w = csv.writer(fp); w.writerow(["ID"] + cols)
        for r in rows: w.writerow([r[0]] + [round(x, 5) if x is not None else "" for x in r[1:]])
    print("wrote village_croptype.csv,", len(rows), "villages")
    a = np.array([[x if x is not None else np.nan for x in r[1:]] for r in rows], float)
    for j, c in enumerate(cols):
        print(f"  {c:12s} mean {np.nanmean(a[:,j]):.4f} range [{np.nanmin(a[:,j]):.4f},{np.nanmax(a[:,j]):.4f}]")

if __name__ == "__main__":
    main()
