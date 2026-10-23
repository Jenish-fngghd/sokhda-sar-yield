"""Known-answer SHAP benchmark. Stages (run in order):

  setup     analysis set, designs, background, exact truth, controls C1 C2 C3 C5
  tune      hyperparameters per configuration, 5-fold CV on seed 0
  grid      every (target, design, noise, model, seed) run + negative control C4
  roar      feature-level and date-level remove-and-retrain

Run:  .venv/Scripts/python.exe run_benchmark.py <stage>
See PLAN.md for the design. Nothing in R3 is written.
"""
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
ATTR = RES / "attr"
for p in (RES, ATTR):
    p.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
NOISE = [0.0, 0.5]
NBG = 100
import os
NJOBS = int(os.environ.get("NJOBS", "6"))   # 14 ran the machine out of memory with RF + TreeSHAP
LEVEL = ["g0_db", "g0_lin", "g0_db_med", "b0_db", "nesz_margin_db"]
TEX = ["cv", "ktex", "glcm_ent", "glcm_resid"]
GEO = ["inc"]
QUAL = ["cov", "npix", "nesz_frac0", "nesz_frac3", "nesz_frac6"]
TYPES = LEVEL + TEX + GEO + QUAL
DATES6 = ["20250606", "20250619", "20250814", "20251013", "20251029", "20251112"]


def split_name(c):
    if c == "crop":
        return "crop", "static"
    if c == "area_ha":
        return "area_ha", "static"
    t, dt = c.rsplit("_", 1)
    return t, dt


def family(c):
    t, _ = split_name(c)
    if t == "crop":
        return "crop"
    if t in LEVEL:
        return "level"
    if t in TEX:
        return "texture"
    if t in GEO:
        return "geometry"
    return "quality"


def date_of(c):
    return split_name(c)[1]


# ------------------------------------------------------------------ setup

def setup():
    from truth import CAUSAL, CROPS, Index, exact_shapley, load
    import shap
    d = load()
    I = Index(d)
    ok = np.isfinite(d[CAUSAL].to_numpy()).all(1)
    d = d[ok].reset_index(drop=True)
    crop = d.crop_type.map({c: i for i, c in enumerate(CROPS)}).to_numpy().astype(float)

    d2 = [f"{t}_{dt}" for dt in DATES6 for t in TYPES if f"{t}_{dt}" in d.columns] + ["area_ha"]
    designs = {"D1": list(CAUSAL), "D2": d2}
    Xall = d[sorted(set(d2) | set(CAUSAL))].copy()
    nulls_na = int(Xall[[c for c in d2 if c not in CAUSAL]].isna().sum().sum())
    Xall = Xall.fillna(Xall.median())                      # nulls only; causal are complete
    Xall["crop"] = crop

    rng = np.random.default_rng(0)
    bg_idx = np.sort(rng.choice(len(d), NBG, replace=False))

    Xc = d[CAUSAL].to_numpy()
    Xc2 = np.column_stack([Xc, crop])
    ytrue = {"T1": I.score(Xc), "T2": I.index(Xc2)}
    controls = {}

    # C5: f reproduces shipped values
    controls["C5_S_maxerr"] = float(np.abs(ytrue["T1"] - I.S_shipped[ok]).max())
    controls["C5_HI_maxerr_rounded"] = float(np.abs(np.round(ytrue["T2"], 2) - I.hi_shipped[ok]).max())
    assert controls["C5_S_maxerr"] < 1e-9 and controls["C5_HI_maxerr_rounded"] < 0.011

    truth = {}
    for T, f, X in (("T1", I.score, Xc), ("T2", I.index, Xc2)):
        t0 = time.time()
        phi, v0, v1 = exact_shapley(f, X, X[bg_idx])
        controls[f"C1_{T}_efficiency_maxerr"] = float(np.abs(phi.sum(1) - (v1 - v0)).max())
        assert controls[f"C1_{T}_efficiency_maxerr"] < 1e-9
        # C2: independent implementation (shap's Exact explainer) on 100 farms
        sv = shap.explainers.Exact(f, shap.maskers.Independent(X[bg_idx], max_samples=NBG))(X[:100]).values
        diff = np.abs(sv - phi[:100]).max()
        controls[f"C2_{T}_maxabs"] = float(diff)
        controls[f"C2_{T}_rel"] = float(diff / np.abs(phi).max())
        assert controls[f"C2_{T}_rel"] < 1e-7
        truth[T] = phi
        print(f"truth {T}: {time.time() - t0:.0f}s, C2 abs {diff:.2e}")

    # C3: positive control. SHAP on the TRUE function in the D2 feature space
    cols = designs["D2"]
    X2 = Xall[cols].to_numpy()
    pos = [cols.index(c) for c in CAUSAL]
    fT1 = lambda Z: I.score(Z[..., pos])
    ex = shap.explainers.Permutation(fT1, shap.maskers.Independent(X2[bg_idx], max_samples=NBG))
    sv = ex(X2[:150], max_evals=4 * (2 * len(cols) + 1)).values
    nullmask = np.array([c not in CAUSAL for c in cols])
    controls["C3_null_mass"] = float(np.abs(sv[:, nullmask]).sum() / np.abs(sv).sum())
    emb = np.zeros_like(sv)
    emb[:, pos] = truth["T1"][:150]
    dates = sorted(set(map(date_of, cols)))
    g = lambda A: np.array([np.abs(A[:, [i for i, c in enumerate(cols) if date_of(c) == dt]].sum(1)).mean() for dt in dates])
    from scipy.stats import spearmanr
    controls["C3_date_rho"] = float(spearmanr(g(sv), g(emb))[0])
    controls["C3_causal_maxabs_err"] = float(np.abs(sv[:, pos] - truth["T1"][:150]).max())
    assert controls["C3_null_mass"] < 1e-6
    print("controls", json.dumps(controls, indent=1))

    meta = {"N": int(len(d)), "N_dropped_incomplete_causal": int((~ok).sum()),
            "null_values_imputed": nulls_na, "designs": designs, "causal": CAUSAL,
            "bg_idx": bg_idx.tolist(), "farm_id": d.farm_id.tolist(),
            "crop_code": crop.tolist(), "weights": {k: float(v) for k, v in I.W.items()},
            "controls": controls}
    (RES / "meta.json").write_text(json.dumps(meta, indent=1))
    Xall.to_csv(RES / "X.csv", index=False)
    np.savez_compressed(RES / "truth.npz", T1=truth["T1"], T2=truth["T2"],
                        y_T1=ytrue["T1"], y_T2=ytrue["T2"])


def load_setup():
    meta = json.loads((RES / "meta.json").read_text())
    X = pd.read_csv(RES / "X.csv")
    tr = np.load(RES / "truth.npz")
    return meta, X, tr


def design_cols(meta, design, target):
    cols = list(meta["designs"][design])
    return cols + ["crop"] if target == "T2" else cols


def truth_matrix(meta, tr, design, target):
    cols = design_cols(meta, design, target)
    players = meta["causal"] + (["crop"] if target == "T2" else [])
    out = np.zeros((meta["N"], len(cols)))
    for j, c in enumerate(players):
        out[:, cols.index(c)] = tr[target][:, j]
    return out


# ------------------------------------------------------------------ models

GRIDS = {"lgbm": [dict(num_leaves=a, min_child_samples=b) for a in (3, 4, 7, 15, 31) for b in (5, 20)],
         "rf": [dict(min_samples_leaf=a, max_features=b) for a in (1, 5) for b in (0.33, 1.0)],
         "mlp": [dict(alpha=a) for a in (1e-4, 1e-2)]}
MODELS = {"T1": ["lgbm", "rf", "mlp"], "T2": ["lgbm", "rf"]}


def fit(model, params, X, y, seed):
    if model == "lgbm":
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        base = dict(learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                    random_state=seed, n_jobs=1, verbose=-1, **params)
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.15, random_state=seed)
        m = lgb.LGBMRegressor(n_estimators=4000, **base)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], callbacks=[lgb.early_stopping(100, verbose=False)])
        # refit with the found size so every tree the explainer sees is a tree the model uses
        m = lgb.LGBMRegressor(n_estimators=max(m.best_iteration_, 10), **base)
        return m.fit(X, y)
    if model == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=1, **params).fit(X, y)
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    reg = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64), early_stopping=True,
                                                       max_iter=3000, random_state=seed, **params))
    return TransformedTargetRegressor(regressor=reg, transformer=StandardScaler()).fit(X, y)


def target_vector(tr, target, noise, seed, permute=False):
    y = tr[f"y_{target}"].copy()
    rng = np.random.default_rng(1000 + seed)
    if noise:
        y = y + rng.normal(0, noise * y.std(), len(y))
    if permute:
        y = rng.permutation(y)
    return y


def cv_r2(model, params, X, y, seed, ytrue=None):
    from sklearn.model_selection import KFold
    oof = np.zeros(len(y))
    for tri, tei in KFold(5, shuffle=True, random_state=seed).split(X):
        oof[tei] = fit(model, params, X[tri], y[tri], seed).predict(X[tei])
    ref = y if ytrue is None else ytrue
    return 1 - ((ref - oof) ** 2).sum() / ((ref - ref.mean()) ** 2).sum(), oof


def tune_one(T, D, s, m, gi, p):
    meta, X, tr = load_setup()
    Xd = X[design_cols(meta, D, T)].to_numpy()
    y = target_vector(tr, T, s, 0)
    r2, _ = cv_r2(m, p, Xd, y, 0)
    return dict(target=T, design=D, noise=s, model=m, grid=gi, params=json.dumps(p), r2=r2)


def tune():
    meta, X, tr = load_setup()
    jobs = []
    for T, D, s, m in itertools.product(("T1", "T2"), ("D1", "D2"), NOISE, ("lgbm", "rf", "mlp")):
        if m not in MODELS[T]:
            continue
        for gi, p in enumerate(GRIDS[m]):
            jobs.append((T, D, s, m, gi, p))

    rows = Parallel(n_jobs=NJOBS, verbose=5)(delayed(tune_one)(*j) for j in jobs)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "tuning_all.csv", index=False)
    best = df.loc[df.groupby(["target", "design", "noise", "model"]).r2.idxmax()]
    best.to_csv(RES / "tuning_best.csv", index=False)
    print(best.to_string())


# ------------------------------------------------------------------ grid

def run_one(T, D, s, m, seed, permute=False):
    import shap
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import KFold
    meta, X, tr = load_setup()
    tag = f"{T}_{D}_s{s}_{m}_seed{seed}" + ("_PERMUTED" if permute else "")
    if (ATTR / f"{tag}.npz").exists():          # resume: a finished run is never recomputed
        z = np.load(ATTR / f"{tag}.npz")
        yt, oof, y = tr[f"y_{T}"], z["oof"], z["y"]
        r2 = lambda ref: 1 - ((ref - oof) ** 2).sum() / ((ref - ref.mean()) ** 2).sum()
        return dict(tag=tag, target=T, design=D, noise=s, model=m, seed=seed, permuted=permute,
                    r2_train_target=r2(y), r2_true=r2(yt), secs=np.nan)
    best = pd.read_csv(RES / "tuning_best.csv")
    p = json.loads(best[(best.target == T) & (best.design == D) & (best.noise == s)
                        & (best.model == m)].params.iloc[0])
    cols = design_cols(meta, D, T)
    Xd = X[cols].to_numpy()
    bg = Xd[meta["bg_idx"]]
    y = target_vector(tr, T, s, seed, permute)
    n, k = Xd.shape
    out = {"oof": np.zeros(n), "y": y, "perm": np.zeros(k)}
    keys = ["int", "path"] if m != "mlp" else ["agn"]
    for kk in keys:
        out[kk] = np.zeros((n, k))
    if m == "lgbm":
        out["gain"] = np.zeros(k)
    t0 = time.time()
    for tri, tei in KFold(5, shuffle=True, random_state=seed).split(Xd):
        mod = fit(m, p, Xd[tri], y[tri], seed)
        out["oof"][tei] = mod.predict(Xd[tei])
        if m == "mlp":
            if D == "D1":
                ex = shap.explainers.Exact(mod.predict, shap.maskers.Independent(bg, max_samples=NBG))
                out["agn"][tei] = ex(Xd[tei], silent=True).values
            else:
                ex = shap.explainers.Permutation(mod.predict, shap.maskers.Independent(bg, max_samples=NBG),
                                                 seed=seed)
                out["agn"][tei] = ex(Xd[tei], max_evals=4 * (2 * k + 1), silent=True).values
        else:
            out["int"][tei] = shap.TreeExplainer(mod, data=bg, feature_perturbation="interventional",
                                                 model_output="raw").shap_values(Xd[tei], check_additivity=False)
            out["path"][tei] = shap.TreeExplainer(mod, feature_perturbation="tree_path_dependent").shap_values(
                Xd[tei], check_additivity=False)
        pi = permutation_importance(mod, Xd[tei], y[tei], n_repeats=10, random_state=seed, scoring="r2", n_jobs=1)
        out["perm"] += pi.importances_mean / 5
        if m == "lgbm":
            g = mod.booster_.feature_importance("gain")
            out["gain"] += g / g.sum() / 5
    np.savez_compressed(ATTR / f"{tag}.npz", **{k_: v.astype("float32") if v.ndim == 2 else v
                                                 for k_, v in out.items()})
    yt = tr[f"y_{T}"]
    r2 = lambda ref: 1 - ((ref - out["oof"]) ** 2).sum() / ((ref - ref.mean()) ** 2).sum()
    return dict(tag=tag, target=T, design=D, noise=s, model=m, seed=seed, permuted=permute,
                r2_train_target=r2(y), r2_true=r2(yt), secs=round(time.time() - t0, 1))


def grid(shard="all"):
    """shard: 'rf' (random forests), 'other' (LightGBM, MLP, negative control) or 'all'."""
    jobs = [(T, D, s, m, sd) for T in ("T1", "T2") for D in ("D1", "D2") for s in NOISE
            for m in MODELS[T] for sd in SEEDS]
    jobs += [("T1", "D2", 0.0, "lgbm", sd, True) for sd in SEEDS]  # C4 negative control
    if shard == "rf":
        jobs = [j for j in jobs if j[3] == "rf"]
    elif shard == "other":
        jobs = [j for j in jobs if j[3] != "rf"]
    jobs.sort(key=lambda j: (j[3] != "rf", j[3] != "mlp"))     # slowest first
    rows = Parallel(n_jobs=NJOBS, verbose=10)(delayed(run_one)(*j) for j in jobs)
    pd.DataFrame(rows).to_csv(RES / f"runs_{shard}.csv", index=False)


def versions():
    import platform, lightgbm, shap, sklearn
    v = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
         "sklearn": sklearn.__version__, "shap": shap.__version__, "lightgbm": lightgbm.__version__,
         "cpus": os.cpu_count()}
    (RES / "versions.json").write_text(json.dumps(v, indent=1))
    print(v)


# ------------------------------------------------------------------ roar

ROAR_K = [0, 1, 2, 3, 5, 8, 12, 20, 30]


def roar_one(seed, level, order_name, order):
    meta, X, tr = load_setup()
    best = pd.read_csv(RES / "tuning_best.csv")
    p = json.loads(best[(best.target == "T1") & (best.design == "D2") & (best.noise == 0.0)
                        & (best.model == "lgbm")].params.iloc[0])
    cols = design_cols(meta, "D2", "T1")
    y = tr["y_T1"]
    rows = []
    ks = ROAR_K if level == "feature" else list(range(len(order) + 1))
    for k in ks:
        drop = set(order[:k]) if level == "feature" else {c for c in cols if date_of(c) in order[:k]}
        keep = [c for c in cols if c not in drop]
        r2 = cv_r2("lgbm", p, X[keep].to_numpy(), y, seed)[0] if keep else 0.0
        rows.append(dict(seed=seed, level=level, order=order_name, k=k, r2=r2))
    return rows


def roar():
    meta, X, tr = load_setup()
    cols = design_cols(meta, "D2", "T1")
    T = truth_matrix(meta, tr, "D2", "T1")
    dates = sorted(set(map(date_of, cols)))
    gimp = lambda A: {dt: np.abs(A[:, [i for i, c in enumerate(cols) if date_of(c) == dt]].sum(1)).mean()
                      for dt in dates}
    # the SHAP ranking ROAR follows comes from the grid's own runs; make them if absent
    Parallel(n_jobs=NJOBS)(delayed(run_one)("T1", "D2", 0.0, "lgbm", sd) for sd in (0, 1, 2))
    jobs = []
    for sd in (0, 1, 2):
        A = np.load(ATTR / f"T1_D2_s0.0_lgbm_seed{sd}.npz")["int"]
        rng = np.random.default_rng(500 + sd)
        tie = rng.random(len(cols)) * 1e-12                # truth ties (all nulls = 0) broken at random
        f_truth = [cols[i] for i in np.argsort(-(np.abs(T).mean(0) + tie))]
        f_shap = [cols[i] for i in np.argsort(-np.abs(A).mean(0))]
        jobs += [(sd, "feature", "truth", f_truth), (sd, "feature", "shap", f_shap)]
        gt, gs = gimp(T), gimp(A)
        jobs += [(sd, "date", "truth", sorted(dates, key=lambda x: -gt[x])),
                 (sd, "date", "shap", sorted(dates, key=lambda x: -gs[x]))]
        for r in range(5):
            jobs.append((sd, "feature", f"random{r}", list(rng.permutation(cols))))
            jobs.append((sd, "date", f"random{r}", list(rng.permutation(dates))))
    rows = Parallel(n_jobs=NJOBS, verbose=5)(delayed(roar_one)(*j) for j in jobs)
    pd.DataFrame([r for rr in rows for r in rr]).to_csv(RES / "roar.csv", index=False)


if __name__ == "__main__":
    stage = {"setup": setup, "tune": tune, "grid": grid, "roar": roar, "versions": versions}[sys.argv[1]]
    stage(*sys.argv[2:])
