"""Score every explanation against the exact truth (PLAN.md section 7) and test H1 to H8.

Writes results/metrics.csv (per run x explainer), results/summary.csv (mean and sd over seeds),
results/leak_by_feature.csv, results/hypotheses.json.

Run:  .venv/Scripts/python.exe analyze.py
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from run_benchmark import ATTR, RES, date_of, design_cols, family, load_setup, truth_matrix

meta, X, tr = load_setup()
runs = pd.read_csv(RES / "runs.csv")
CAUSAL = set(meta["causal"])


def feature_info(T, D):
    cols = design_cols(meta, D, T)
    causal = np.array([c in CAUSAL or c == "crop" for c in cols])
    Xs = X[cols].rank().to_numpy()
    C = np.abs(np.corrcoef(Xs, rowvar=False))
    maxrho = np.where(causal, 1.0, C[:, causal].max(1))
    stratum = np.where(causal, "causal", np.where(maxrho >= 0.98, "copy",
                                                  np.where(maxrho >= 0.5, "proxy", "unrelated")))
    return cols, causal, maxrho, stratum


def groups(cols, key):
    labs = sorted(set(map(key, cols)))
    return labs, [[i for i, c in enumerate(cols) if key(c) == g] for g in labs]


def grouped(A, idx):
    """Signed group sums per farm: (N, n_groups)."""
    return np.column_stack([A[:, ix].sum(1) for ix in idx])


def rho(a, b):
    r = spearmanr(a, b)[0]
    return float(r) if np.isfinite(r) else np.nan


def score(A, Tm, cols, causal, stratum, y, local):
    k = int(causal.sum())
    out = {}
    if local:
        gi, ti = np.abs(A).mean(0), np.abs(Tm).mean(0)
    else:
        gi, ti = np.clip(A, 0, None), np.abs(Tm).mean(0)
    tot = gi.sum()
    out["null_mass"] = gi[~causal].sum() / tot
    for s in ("copy", "proxy", "unrelated"):
        out[f"mass_{s}"] = gi[stratum == s].sum() / tot
    out["prec_at_k"] = causal[np.argsort(-gi)[:k]].mean()
    out["rho_feature"] = rho(gi, ti)
    for name, key in (("date", date_of), ("family", family)):
        labs, idx = groups(cols, key)
        if local:
            ga, gt = np.abs(grouped(A, idx)).mean(0), np.abs(grouped(Tm, idx)).mean(0)
        else:
            ga, gt = np.array([gi[ix].sum() for ix in idx]), np.array([ti[ix].sum() for ix in idx])
        out[f"rho_{name}"] = rho(ga, gt)
        if name == "date":
            share = dict(zip(labs, ga / ga.sum()))
            tshare = dict(zip(labs, gt / gt.sum()))
            out["oct29_share"] = share.get("20251029", 0.0)
            out["aug14_share"] = share.get("20250814", 0.0)
            out["aug14_share_true"] = tshare.get("20250814", 0.0)
    if local:
        labs, idx = groups(cols, lambda c: f"{date_of(c)}|{family(c)}")
        GA, GT = grouped(A, idx), grouped(Tm, idx)
        cos = (GA * GT).sum(1) / (np.linalg.norm(GA, axis=1) * np.linalg.norm(GT, axis=1) + 1e-300)
        out["cos_group_med"], out["cos_group_q25"], out["cos_group_q75"] = np.percentile(cos, [50, 25, 75])
        cosf = (A * Tm).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(Tm, axis=1) + 1e-300)
        out["cos_feature_med"] = np.median(cosf)
        flag = y <= np.percentile(y, 10)
        out["why_flagged_agree"] = (GA[flag].argmin(1) == GT[flag].argmin(1)).mean()
    return out


rows, leaks = [], []
for r in runs.itertuples():
    z = np.load(ATTR / f"{r.tag}.npz")
    cols, causal, maxrho, stratum = feature_info(r.target, r.design)
    Tm = truth_matrix(meta, tr, r.design, r.target)
    y = tr[f"y_{r.target}"]
    for key in ("int", "path", "agn", "perm", "gain"):
        if key not in z.files:
            continue
        local = key in ("int", "path", "agn")
        m = score(z[key].astype(float), Tm, cols, causal, stratum, y, local)
        rows.append(dict(tag=r.tag, target=r.target, design=r.design, noise=r.noise, model=r.model,
                         seed=r.seed, permuted=r.permuted, explainer=key, r2_true=r.r2_true,
                         r2_train_target=r.r2_train_target, **m))
        if local and r.design == "D2" and not r.permuted:
            gi = np.abs(z[key]).mean(0)
            for j, c in enumerate(cols):
                leaks.append(dict(target=r.target, noise=r.noise, model=r.model, seed=r.seed,
                                  explainer=key, feature=c, date=date_of(c), family=family(c),
                                  stratum=stratum[j], maxrho=maxrho[j], share=gi[j] / gi.sum(),
                                  true_share=np.abs(Tm).mean(0)[j] / np.abs(Tm).mean(0).sum()))

M = pd.DataFrame(rows)
M.to_csv(RES / "metrics.csv", index=False)
L = pd.DataFrame(leaks)
L.groupby(["target", "noise", "model", "explainer", "feature", "date", "family", "stratum"]) \
 .agg(maxrho=("maxrho", "first"), share=("share", "mean"), share_sd=("share", "std"),
      true_share=("true_share", "first")).reset_index().to_csv(RES / "leak_by_feature.csv", index=False)

num = [c for c in M.columns if c not in ("tag", "target", "design", "model", "explainer", "permuted", "seed", "noise")]
S = M.groupby(["target", "design", "noise", "model", "explainer", "permuted"])[num].agg(["mean", "std"])
S.columns = [f"{a}_{b}" for a, b in S.columns]
S = S.reset_index()
S.to_csv(RES / "summary.csv", index=False)

# ------------------------------------------------------------------ hypotheses
ok = M[~M.permuted]
q = lambda **kw: ok.loc[np.logical_and.reduce([ok[k] == v for k, v in kw.items()])]
H = {}
lg = q(model="lgbm", noise=0.0, explainer="int")
H["H1"] = {"r2_true_D1": lg[lg.design == "D1"].r2_true.mean(), "r2_true_D2": lg[lg.design == "D2"].r2_true.mean()}
H["H1"]["supported"] = bool(min(v for v in H["H1"].values()) >= 0.90)
H["H2"] = {"rho_feature_D1_lgbm_int_T1": q(model="lgbm", noise=0.0, explainer="int", design="D1", target="T1").rho_feature.mean(),
           "rho_feature_D1_all_tree_int": q(noise=0.0, explainer="int", design="D1").groupby(["target", "model"]).rho_feature.mean().to_dict()}
H["H2"]["supported"] = bool(H["H2"]["rho_feature_D1_lgbm_int_T1"] >= 0.90)
d2 = q(noise=0.0, explainer="int", design="D2")
H["H3"] = {"null_mass": d2.groupby(["target", "model"]).null_mass.mean().to_dict(),
           "copy_share_of_null": (d2.mass_copy / d2.null_mass).groupby([d2.target, d2.model]).mean().to_dict()}
H["H3"]["supported"] = bool(d2.null_mass.mean() >= 0.20 and (d2.mass_copy / d2.null_mass).mean() > 0.5)
H["H4"] = {"rho_date": d2.groupby(["target", "model"]).rho_date.mean().to_dict(),
           "rho_family": d2.groupby(["target", "model"]).rho_family.mean().to_dict(),
           "oct29_share": d2.groupby(["target", "model"]).oct29_share.mean().to_dict()}
H["H4"]["supported"] = bool(d2.rho_date.mean() >= 0.90 and d2.rho_family.mean() >= 0.90 and d2.oct29_share.mean() < 0.05)
pi = q(noise=0.0, design="D2")
pv = pi.pivot_table(index=["target", "model", "seed"], columns="explainer", values="null_mass").dropna(subset=["int", "path"])
H["H5"] = {"null_mass_int": pv["int"].mean(), "null_mass_path": pv["path"].mean(),
           "path_gt_int_fraction_of_runs": float((pv["path"] > pv["int"]).mean())}
H["H5"]["supported"] = bool(pv["path"].mean() > pv["int"].mean())
loc = ok[ok.explainer.isin(["int", "agn"])]
agg = loc.groupby(["target", "design", "noise", "model"])[["r2_true", "cos_group_med", "rho_feature"]].mean()
H["H6"] = {"spearman_r2_vs_cos_group": rho(agg.r2_true, agg.cos_group_med),
           "spearman_r2_vs_rho_feature": rho(agg.r2_true, agg.rho_feature), "n_configs": len(agg)}
H["H6"]["supported"] = bool(abs(H["H6"]["spearman_r2_vs_rho_feature"]) < 0.5)
H["H7"] = "see roar.csv / figure; judged in RESULTS.md"
nz = loc.groupby(["noise"])[["cos_group_med", "rho_date", "rho_feature"]].mean()
H["H8"] = {"by_noise": nz.to_dict(),
           "local_drop": float(nz.cos_group_med.loc[0.0] - nz.cos_group_med.loc[0.5]),
           "global_date_drop": float(nz.rho_date.loc[0.0] - nz.rho_date.loc[0.5])}
H["H8"]["supported"] = bool(H["H8"]["local_drop"] > H["H8"]["global_date_drop"])
neg = M[M.permuted]
H["C4_negative_control"] = {"r2_train_target": neg.r2_train_target.mean(), "r2_true": neg.r2_true.mean(),
                            "pass": bool(neg.r2_train_target.mean() <= 0.05)}
def keys(o):
    if isinstance(o, dict):
        return {(" ".join(map(str, k)) if isinstance(k, tuple) else str(k)): keys(v) for k, v in o.items()}
    return o


H = keys(H)
(RES / "hypotheses.json").write_text(json.dumps(H, indent=1, default=float))
print(json.dumps(H, indent=1, default=float))
