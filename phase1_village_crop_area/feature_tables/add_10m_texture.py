"""Add 10m-resolution texture as new village features (corr 0.53 w/ 20m texture = fresh).
Appends ndvi_contrast_10m, s1vh_contrast_10m, s1vh_var_10m to village_aux.csv."""
import os
os.environ.pop("PROJ_LIB", None)
import numpy as np, rasterio, shapefile, csv
from rasterio.features import geometry_mask
from pyproj import Transformer

D = os.path.join(os.environ.get("R1_WORK", "."), "data/stack")
SHP = os.path.join(os.environ.get("R1_DATA", "anrf-aise-hack-2026-round-1-sar-crop-mapping-challenge"), "villages_clean/villages_clean.shp")
AUXP = os.path.join(os.environ.get("R1_WORK", "."), "results/cache/village_aux.csv")

with rasterio.open(os.path.join(D,"cropmask_10m.tif")) as ds:
    cn=list(ds.descriptions);dw=ds.read(cn.index("dw_crops")+1);bu=ds.read(cn.index("dw_built")+1);wc=ds.read(cn.index("worldcover")+1)
    tf,H,W=ds.transform,ds.height,ds.width
crop=(dw>0.30)&(bu<0.35)&(wc!=80)&(wc!=50)
with rasterio.open(os.path.join(D,"texture_10m.tif")) as ds:
    tn=list(ds.descriptions); T={n:ds.read(i+1) for i,n in enumerate(tn)}

sf=shapefile.Reader(SHP); tfm=Transformer.from_crs(4326,32643,always_xy=True)
newcols=["ndvi_contrast_10m","s1vh_contrast_10m","s1vh_var_10m"]
srcname={"ndvi_contrast_10m":"ndvi_contrast","s1vh_contrast_10m":"s1vh_contrast","s1vh_var_10m":"s1vh_var"}
vals={}
for shp,rec in zip(sf.shapes(),sf.records()):
    q=np.array(shp.points);x,y=tfm.transform(q[:,0],q[:,1])
    m=~geometry_mask([{"type":"Polygon","coordinates":[list(zip(x,y))]}],(H,W),tf)&crop
    vals[rec["ID"]]=[ (lambda v: float(v[np.isfinite(v)].mean()) if np.isfinite(v).any() else np.nan)(T[srcname[c]][m]) for c in newcols ]
# fill nan w/ col mean
arr=np.array([vals[i] for i in sorted(vals)],float)
for j in range(arr.shape[1]): arr[~np.isfinite(arr[:,j]),j]=np.nanmean(arr[np.isfinite(arr[:,j]),j])

rows=list(csv.DictReader(open(AUXP))); hdr=rows[0].keys()
ids=sorted(vals)
with open(AUXP,"w",newline="") as f:
    w=csv.writer(f); w.writerow(list(hdr)+newcols)
    for r,i in zip(rows,ids):
        w.writerow([r[k] for k in hdr]+[round(float(x),4) for x in arr[ids.index(int(r["ID"]))]])
print("appended",newcols,"to village_aux.csv")
