"""Append texture2 features (entropy, 7x7 window) to village_aux + test orthogonality vs used texture."""
import os
os.environ.pop("PROJ_LIB", None)
import numpy as np, rasterio, shapefile, csv
from rasterio.features import geometry_mask
from pyproj import Transformer

D=os.path.join(os.environ.get("R1_WORK", "."), "data/stack")
SHP=os.path.join(os.environ.get("R1_DATA", "anrf-aise-hack-2026-round-1-sar-crop-mapping-challenge"), "villages_clean/villages_clean.shp")
AUXP=os.path.join(os.environ.get("R1_WORK", "."), "results/cache/village_aux.csv")

with rasterio.open(os.path.join(D,"cropmask.tif")) as ds:
    cn=list(ds.descriptions);dw=ds.read(cn.index("dw_crops")+1);bu=ds.read(cn.index("dw_built")+1);wc=ds.read(cn.index("worldcover")+1)
    tf,H,W=ds.transform,ds.height,ds.width
crop=(dw>0.30)&(bu<0.35)&(wc!=80)&(wc!=50)
with rasterio.open(os.path.join(D,"texture2.tif")) as ds:
    T={n:ds.read(i+1) for i,n in enumerate(ds.descriptions)}
new=list(T.keys())
sf=shapefile.Reader(SHP);tfm=Transformer.from_crs(4326,32643,always_xy=True)
vals={}
for shp,rec in zip(sf.shapes(),sf.records()):
    q=np.array(shp.points);x,y=tfm.transform(q[:,0],q[:,1])
    m=~geometry_mask([{"type":"Polygon","coordinates":[list(zip(x,y))]}],(H,W),tf)&crop
    vals[rec["ID"]]=[(lambda v:float(v[np.isfinite(v)].mean()) if np.isfinite(v).any() else np.nan)(T[c][m]) for c in new]
ids=sorted(vals); arr=np.array([vals[i] for i in ids],float)
for j in range(arr.shape[1]): arr[~np.isfinite(arr[:,j]),j]=np.nanmean(arr[np.isfinite(arr[:,j]),j])

rows=list(csv.DictReader(open(AUXP))); hdr=list(rows[0].keys())
with open(AUXP,"w",newline="") as f:
    w=csv.writer(f);w.writerow(hdr+new)
    for r in rows: w.writerow([r[k] for k in hdr]+[round(float(x),4) for x in arr[ids.index(int(r["ID"]))]])
# orthogonality vs used texture (ndvi_contrast, s1vh_contrast, s1vh_var)
aux={int(r["ID"]):r for r in csv.DictReader(open(AUXP))}
def z(f):v=np.array([float(aux[i][f]) for i in ids]);return (v-v.mean())/(v.std()+1e-9)
used=["ndvi_contrast","s1vh_contrast","s1vh_var"]
print("new texture2 feat: max_corr_vs_used_texture")
for f in new:
    mx=max(abs(np.corrcoef(z(u),z(f))[0,1]) for u in used)
    print(f"  {f:20s} {mx:.2f}{'  <== FRESH' if mx<0.7 else ''}")
