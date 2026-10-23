"""Phase 0: copy, join and verify the per-farm assets inherited from Round 2.

The Round 3 shapefile is byte-identical to Round 2's, so farm_id 1-966 means the same
polygon and every table below joins exactly. "Fresh rebuild" applies to the pipeline,
not to data already fetched -- two of these came through a Google auth route that is
now interactive-only and cannot be re-run unattended.

They are COPIED into this round's tree, never read across at analysis time: reading
across from a frozen folder is how a frozen folder stops being frozen.

Anything that fails the join is dropped and reported, not patched.
Writes data_aux/inherited/*.csv and results/tables/p0_inherit_audit.csv.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AUX, R2, TABLES, log

DEST = AUX / "inherited"
POST = R2 / "post-r2"

# (label, path relative to post-r2/, farm-id column, why it is worth inheriting)
ASSETS = [
    ("embed",      "results/e14_embeddings/farm_embed.csv",              None,
     "64-band AlphaEarth annual embedding; re-fetch needs interactive Google auth"),
    ("s1_agree",   "results/e17_dense_s1/two_sensor_agreement.csv",      "farm_id",
     "per-farm s1_ok/emb_ok: the 42.5%-uncorroborated finding as a column"),
    ("label_dist", "results/e18_label_distribution/label_distribution.csv", "farm_id",
     "per-farm five-class posterior + entropy"),
    ("uncert",     "results/e11_uncertainty/health_uncertainty.csv",     "farm_id",
     "calibrated per-farm sampling SE"),
    ("gee",        "results/e13_gee/farm_gee.csv",                       "farm_id",
     "Dynamic World + WorldCereal; re-fetch needs interactive Google auth"),
    ("consensus",  "results/e4_consensus_all/consensus_crop.csv",        "farm_id",
     "all six teams' crop label per farm"),
    ("calib_ref",  "results/e1_calibration/farm_features_calibrated.csv", None,
     "SF-squared four-date features: a REPRODUCTION TARGET, never an input"),
    ("anchors",    "results/e10_kharif_anchors/vadodara_season_crops.csv", None,
     "461 Vadodara season-split district records; source API is paged and flaky"),
    # third-party: same scepticism as any competitor submission
    ("orion_feat", "writeups_submissions/project_orion_team_apes/farm_features.csv", "farm_id",
     "THIRD PARTY. Their own per-farm g0_db_T1..T4: the only external check on our chain"),
    ("orion_ndvi", "writeups_submissions/project_orion_team_apes/farm_ndvi.csv", "farm_id",
     "THIRD PARTY. Per-farm NDVI incl. an 18 Oct scene we never fetched"),
    ("orion_vill", "writeups_submissions/project_orion_team_apes/village_summary.csv", None,
     "THIRD PARTY. A worked example of the 15-point aggregation deliverable"),
    ("orion_zone", "writeups_submissions/project_orion_team_apes/zone_summary.csv", None,
     "THIRD PARTY. 46-zone 500 m grid: the sub-village product we never built"),
]


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, rel, idcol, why in ASSETS:
        src = POST / rel
        r = {"asset": label, "src": rel, "why": why}
        if not src.exists():
            r.update(status="MISSING", rows=0, cols=0, note="file not found")
            rows.append(r)
            print(f"  {label:<12} MISSING  {rel}")
            continue
        dst = DEST / f"{label}.csv"
        shutil.copy2(src, dst)
        d = pd.read_csv(dst)
        r["rows"], r["cols"] = len(d), d.shape[1]

        if idcol is None:
            r.update(status="COPIED", note="no farm-id join expected")
        elif idcol not in d.columns:
            r.update(status="NO_KEY", note=f"expected column {idcol} absent")
        else:
            ids = pd.to_numeric(d[idcol], errors="coerce").dropna().astype(int)
            exact = (len(d) == 966 and ids.min() == 1 and ids.max() == 966
                     and ids.nunique() == 966)
            r.update(status="JOIN_OK" if exact else "JOIN_PARTIAL",
                     note=f"ids {ids.min()}..{ids.max()}, unique {ids.nunique()}")
        rows.append(r)
        print(f"  {label:<12} {r['status']:<13} {r['rows']:>4} rows x {r['cols']:>3} cols"
              f"   {r.get('note','')}")

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "p0_inherit_audit.csv", index=False)
    ok = (out.status.isin(["JOIN_OK", "COPIED"])).sum()
    log("p0_inherit", assets=len(rows), usable=int(ok), dest=str(DEST))


if __name__ == "__main__":
    main()
