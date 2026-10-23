"""The known answer: the shipped health index as a closed-form function with frozen constants,
and its exact interventional Shapley values by full subset enumeration.

Every data-dependent constant (robust-z medians/MADs, growth median, weights, per-crop
medians/MADs) is taken from the shipped 966-farm computation, so f is a FIXED function and its
Shapley values are a well-defined truth. Control C5 (in run_benchmark) proves f reproduces the
shipped score and index.
"""
import sys
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
R3 = HERE.parents[1] / "phase3_yield_forecast"
sys.path.insert(0, str(R3 / "src"))
import p3_build as pb  # shipped code; used for constants and for control C5 only
from common import CROPS

TABLES = R3 / "results" / "tables"
DATES5 = ["20250606", "20250619", "20250814", "20251013", "20251112"]   # LEVEL_DATES
CAUSAL = (["g0_db_20250814", "g0_db_20250619", "g0_db_20251013", "g0_db_20251112"]
          + [f"g0_lin_{d}" for d in DATES5] + ["cv_20250814"])
DOY = np.array([pd.Timestamp(d).dayofyear for d in DATES5], dtype="float64")
LATE_DOY = np.array([pd.Timestamp(d).dayofyear for d in ("20251013", "20251112")], dtype="float64")


def robust(x):
    """(median, MAD*1.4826) exactly as pb.z computes them."""
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return med, (mad if mad > 1e-9 else 1.0)


def load():
    f = pd.read_csv(TABLES / "p1_farm_features.csv")
    lab = pd.read_csv(R3 / "data_aux" / "inherited" / "label_dist.csv")[["farm_id", "crop_type"]]
    return f.merge(lab, on="farm_id")


class Index:
    """f(x) for T1 (composite score S) and T2 (within-crop health index)."""

    def __init__(self, d):
        # raw part values on all 966 farms, exactly as the shipped health_parts builds them
        growth = (d["g0_db_20250814"] - d["g0_db_20250619"]).to_numpy()
        self.gmed = np.nanmedian(growth)
        lin = d[[f"g0_lin_{x}" for x in DATES5]].to_numpy()
        integral = np.trapezoid(np.where(np.isfinite(lin), lin, np.nan), DOY, axis=1)
        L = d[["g0_db_20251013", "g0_db_20251112"]].to_numpy()
        xc = LATE_DOY - LATE_DOY.mean()
        sen = np.nansum((L - np.nanmean(L, axis=1, keepdims=True)) * xc, axis=1) / (xc ** 2).sum()
        raw = {"level": d["g0_db_20250814"].to_numpy(), "growth": growth - self.gmed,
               "uniform": d["cv_20250814"].to_numpy(), "persist": integral, "senesce": sen}
        self.c = {k: robust(v) for k, v in raw.items()}

        parts = pb.health_parts(d, use_29oct=False)
        self.W = pb.derive_weights(parts)
        self.S_shipped = sum(self.W[k] * parts[k] for k in self.W)
        self.hi_shipped = pb.health_index(parts, self.W, d.crop_type)
        crop = d.crop_type.to_numpy()
        self.cc = np.array([robust(self.S_shipped[crop == c]) for c in CROPS])  # (5, 2)

    def _z(self, x, k):
        med, mad = self.c[k]
        return np.clip((x - med) / mad, -4, 4)

    def score(self, X):
        """T1. X columns in CAUSAL order; any leading shape."""
        g = lambda name: X[..., CAUSAL.index(name)]
        lin = np.stack([g(f"g0_lin_{x}") for x in DATES5], axis=-1)
        integral = np.trapezoid(lin, DOY, axis=-1)
        sen = (g("g0_db_20251112") - g("g0_db_20251013")) / (LATE_DOY[1] - LATE_DOY[0])
        p = {"level": self._z(g("g0_db_20250814"), "level"),
             "growth": self._z(g("g0_db_20250814") - g("g0_db_20250619") - self.gmed, "growth"),
             "uniform": -self._z(g("cv_20250814"), "uniform"),
             "persist": self._z(integral, "persist"),
             "senesce": self._z(sen, "senesce")}
        return sum(self.W[k] * p[k] for k in self.W)

    def index(self, X):
        """T2. X = CAUSAL columns + crop code (0..4) as the last column."""
        S = self.score(X[..., :-1])
        c = X[..., -1].astype(int)
        z = np.clip((S - self.cc[c, 0]) / self.cc[c, 1], -4, 4)
        return np.clip(100.0 * norm.cdf(z / 2.0), 0, 100)


def exact_shapley(f, X, bg):
    """Exact interventional Shapley values of f at rows X against background bg.
    v(S) = mean_b f(x_S, b_notS); phi_i = sum_S |S|!(n-|S|-1)!/n! [v(S+i) - v(S)]."""
    N, n = X.shape
    masks = np.arange(2 ** n)
    bits = ((masks[:, None] >> np.arange(n)) & 1).astype(bool)          # (2^n, n)
    v = np.empty((2 ** n, N))
    for m in masks:
        Xm = np.where(bits[m], X[:, None, :], bg[None, :, :])           # (N, B, n)
        v[m] = f(Xm).mean(axis=1)
    size = bits.sum(1)
    wt = np.array([factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)])
    phi = np.zeros((N, n))
    for i in range(n):
        without = masks[~bits[:, i]]
        phi[:, i] = (wt[size[without]][:, None] * (v[without | (1 << i)] - v[without])).sum(0)
    return phi, v[0], v[-1]          # v[0] = E_bg f, v[-1] = f(x)
