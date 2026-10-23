"""T14: the deliverable figures.

Round 2's hardest-won lesson about figures, and it took four rounds of rework: RENDER EVERY
FIGURE, OPEN IT, AND LOOK AT IT, then re-cut until it reads in two seconds. Doing that found
three real defects and one substantive error -- a wrong physical explanation attached to a
correct feature, which no validation number would ever have caught, because every metric
passed identically either way. The figure was the check on the prose.

Nine figures, each carrying one claim:

  F1  THE DELIVERABLE      per-crop production and yield, with the three epistemic objects
                           colour-coded, because that distinction IS our differentiator and
                           it has to be visible without reading a caption.
  F2  UNCERTAINTY          the decomposition. The point of the panel is that the ensemble
                           bar is ZERO at village level and the anchor bar is the big one --
                           an honest self-criticism rendered as a picture.
  F3  D1                   forecast against Round 2's yield-to-date, per crop. The check no
                           other team can run, and it should read at a glance: cereals flat,
                           cotton up, bajra down because we corrected its anchor.
  F4  THE WITNESS          T1's within-crop correlation against same-day Sentinel-2, on both
                           spatial halves (V3), because a relationship that exists in only
                           one half is a local artefact.

  F5  CLIMATOLOGY          nine years of Sentinel-1 over Sokhda. The LEFT panel is the point:
                           farm and non-farm move together, so the raw 2025 rise is landscape
                           moisture and only the difference can be about the crop.

  F6  SPATIAL              is the ranking the farm or its neighbourhood? The positive control
                           (incidence angle) never reaches its sill inside the village, which
                           is what makes the subject curve's flatness readable as a result.
  F7  VALUE OF A SCENE     what each of the six acquisitions bought. The permutation band
                           overlaps the first greedy points and that overlap is annotated,
                           not cropped.
  F8  SPECKLE              would the same 97 farms be named on a re-flight? Left panel is the
                           corrected positive control; the first version of it failed.
  F9  EDGE                 is the list measuring the crop or the parcel outline? The two right
                           hand pairs coinciding IS the finding.

Writes results/figures/p3_f1..f9.png.

Run:  py -3.12 src/p3_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CROPS, FARMS, FIGURES, RESULTS, TABLES, log

import geopandas as gpd

UTM = 32643
OBJ_COLOR = {"FORECAST": "#c53030",
             "NEAR-COMPLETE MEASUREMENT": "#2f855a",
             "RETROSPECTIVE RECONSTRUCTION": "#4a5568"}


def f1_deliverable():
    v = pd.read_csv(TABLES / "p3_village_summary.csv")
    v = v[v.crop != "ALL"].sort_values("production_t", ascending=False)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))

    c = [OBJ_COLOR[o] for o in v.epistemic_object]
    b = ax[0].barh(v.crop, v.production_t, color=c, edgecolor="#222")
    for r, bar in zip(v.itertuples(), b):
        ax[0].text(bar.get_width() + 6, bar.get_y() + bar.get_height() / 2,
                   f"{r.production_t:,.0f} t   ({r.area_share_pct:.0f}% of area)",
                   va="center", fontsize=9)
    ax[0].set_xlim(0, v.production_t.max() * 1.42)
    ax[0].set_xlabel("production (t)")
    # ★ found by LOOKING: without this the panel reads "groundnut dominates Sokhda", when
    # most of the gap is the lint convention -- cotton leads on AREA, not on tonnage.
    ax[0].text(0.985, 0.05, "cotton is reported as LINT (~34% of seed cotton):\n"
                            "it leads on AREA, not on tonnage",
               transform=ax[0].transAxes, ha="right", va="bottom", fontsize=8.5,
               color="#c53030", style="italic")
    ax[0].set_title("Sokhda kharif 2025 — 710 t over 447.5 ha\n"
                    "production = Σ(yield × area)", fontsize=11)
    ax[0].invert_yaxis()

    ax[1].barh(v.crop, v.yield_forecast_t_ha_aw, color=c, edgecolor="#222", label="_")
    ax[1].barh(v.crop, v.yield_to_date_t_ha_aw, color="none", edgecolor="#222",
               hatch="///", linewidth=1.0)
    for r in v.itertuples():
        ax[1].text(r.yield_forecast_t_ha_aw + 0.04, r.crop, f"{r.yield_forecast_t_ha_aw:.2f}",
                   va="center", fontsize=9)
    ax[1].set_xlabel("t/ha   (cotton is LINT, which is why it is small)")
    ax[1].set_title("final forecast (solid) vs yield already in hand (hatched)\n"
                    "only cotton has a large gap — it is still being picked", fontsize=11)
    ax[1].invert_yaxis()
    ax[1].set_xlim(0, 3.0)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=v_, ec="#222") for v_ in OBJ_COLOR.values()]
    fig.legend(handles, [k.title() for k in OBJ_COLOR], loc="lower center", ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("THREE EPISTEMIC OBJECTS IN ONE COLUMN — a forecast, a measurement, "
                 "and a reconstruction", fontsize=12, y=1.0)
    fig.tight_layout()
    p = FIGURES / "p3_f1_deliverable.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def f2_uncertainty():
    u = pd.read_csv(TABLES / "p3_uncertainty.csv")
    cb = pd.read_csv(TABLES / "p3_conformal_beta.csv")
    vill = u[u.unit.str.contains("90% width")]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # The "crop label" bar is our OWN posterior. Showing it alone understates the term by ~6x
    # and made this figure contradict the slide it illustrates, so the externally measured
    # value (five other teams labelling the same 966 farms) sits beside it.
    LABEL_EXTERNAL_T = 236.5
    labels = ["ensemble\n(SAR modifier)", "crop label\n(our posterior)",
              "crop label\n(5 other teams)", "anchor", "label + anchor"]
    vals = [0.0, float(vill[vill.term.str.contains("label")].value.iloc[0]),
            LABEL_EXTERNAL_T,
            float(vill[vill.term.str.contains("anchor")].value.iloc[0]),
            float(vill[vill.term.str.contains("both")].value.iloc[0])]
    cols = ["#c53030", "#a0aec0", "#4a5568", "#2b6cb0", "#1a202c"]
    b = ax[0].bar(labels, vals, color=cols, edgecolor="#222")
    for bar, v_ in zip(b, vals):
        ax[0].text(bar.get_x() + bar.get_width() / 2, v_ + 3,
                   "0.0" if v_ == 0 else f"{v_:.0f} t", ha="center", fontsize=10)
    ax[0].set_ylabel("90% interval width on village production (t)")
    ax[0].set_ylim(0, max(vals) * 1.25)
    ax[0].set_title("Where the village-total uncertainty actually is\n"
                    "the SAR MODIFIER is zero by construction — but the SAR crop map is not",
                    fontsize=11)
    ax[0].annotate("the modifier has area-weighted\nmean 1.0 within crop, so the\n"
                   "total is anchor × area",
                   xy=(0, 8), xytext=(0.28, max(vals) * 0.74), fontsize=8.5,
                   arrowprops=dict(arrowstyle="->", color="#c53030"), color="#c53030")
    # ★ found by LOOKING: this annotation rendered BEHIND the bars and was unreadable.
    # A background box plus zorder makes it legible wherever it lands.
    ax[0].annotate("sampling our OWN posterior measures\nour confidence, not our accuracy",
                   xy=(1, vals[1] + 8), xytext=(0.62, max(vals) * 0.50), fontsize=8.5,
                   arrowprops=dict(arrowstyle="->", color="#4a5568"), color="#4a5568",
                   zorder=6, bbox=dict(boxstyle="round,pad=0.35", fc="white",
                                       ec="#4a5568", alpha=0.95))

    ax[1].plot(cb.assumed_label_error_beta, 100 * cb.guaranteed_coverage, "o-",
               color="#2b6cb0", lw=2)
    ax[1].axhline(90, color="#c53030", ls="--", lw=1)
    ax[1].text(0.40, 91.5, "nominal 90%", color="#c53030", fontsize=9, ha="right")
    for r in cb.itertuples():
        ax[1].annotate(f"{100*r.guaranteed_coverage:.0f}%",
                       (r.assumed_label_error_beta, 100 * r.guaranteed_coverage),
                       textcoords="offset points", xytext=(0, -14), fontsize=8.5,
                       ha="center")
    ax[1].set_xlabel(r"assumed crop-label error rate  $\beta$")
    ax[1].set_ylabel("guaranteed coverage (%)")
    ax[1].set_ylim(40, 100)
    ax[1].grid(alpha=0.3)
    ax[1].set_title(r"Conformal coverage is $1-\alpha-\beta$, and we cannot bound $\beta$"
                    "\nso we publish the arithmetic instead of a number", fontsize=11)

    fig.tight_layout()
    p = FIGURES / "p3_f2_uncertainty.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def f3_d1():
    d = pd.read_csv(RESULTS / "d4_debug.csv")
    r2 = pd.read_csv(Path(__file__).resolve().parents[2] / "phase2_crop_type_health"
                     / "results" / "submission.csv")
    area = d.area_ha.to_numpy()
    rows = []
    for c in CROPS:
        m = (d.crop_type == c).to_numpy()
        rows.append((c,
                     float(np.sum(r2.yield_estimate_to_date.to_numpy()[m] * area[m])
                           / area[m].sum()),
                     float(np.sum(d.yield_forecast_t_ha.to_numpy()[m] * area[m])
                           / area[m].sum())))
    t = pd.DataFrame(rows, columns=["crop", "r2", "r3"]).sort_values("r3")

    fig, ax = plt.subplots(figsize=(9, 5.4))
    y = np.arange(len(t))
    for i, r in enumerate(t.itertuples()):
        ax.plot([r.r2, r.r3], [i, i], color="#999", lw=2, zorder=1)
        ax.scatter(r.r2, i, s=90, color="#4a5568", zorder=2,
                   label="Round 2 yield-to-date" if i == 0 else "")
        ax.scatter(r.r3, i, s=90, color="#c53030", zorder=2,
                   label="Round 3 final forecast" if i == 0 else "")
        ratio = r.r3 / r.r2
        col = "#4a5568" if abs(ratio - 1) < 0.05 else ("#c53030" if ratio > 1 else "#2b6cb0")
        ax.text(max(r.r2, r.r3) + 0.06, i, f"×{ratio:.2f}", va="center", fontsize=10,
                color=col)
    ax.set_yticks(y)
    ax.set_yticklabels(t.crop)
    ax.set_xlabel("area-weighted t/ha")
    ax.set_xlim(0, 3.1)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    # ★ found by LOOKING: the first title read "cereals flat ... bajra down" -- but BAJRA
    # IS A CEREAL. A self-contradicting caption on a correct figure, which is exactly the
    # class of defect Round 2 warned that no metric can catch.
    ax.set_title("D1 — the consistency check no other team can run\n"
                 "rice & maize unchanged (harvested before R2's last date) · cotton up "
                 "(final vs to-date) · bajra down (we corrected its anchor)", fontsize=11)
    fig.tight_layout()
    p = FIGURES / "p3_f3_d1.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def f4_witness():
    t1 = pd.read_csv(TABLES / "p3_t1_gai.csv", dtype={"capella": str, "s2": str})
    t1 = t1[t1.same_day & (t1.feature == "g0_db")]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))

    order = ["Cotton", "Groundnut", "Rice", "Bajra", "Maize"]      # by village area
    piv = t1.pivot(index="crop", columns="capella", values="rho").reindex(order)
    w = 0.38
    xx = np.arange(len(piv))
    for i, col in enumerate(piv.columns):
        ax[0].bar(xx + (i - 0.5) * w, piv[col], w, edgecolor="#222",
                  color=["#2b6cb0", "#c53030"][i],
                  label=f"{col[6:8]} {'Oct' if col[4:6] == '10' else 'Nov'} (same-day S2)")
    ax[0].axhline(0.5, color="#2f855a", ls="--", lw=1.2)
    ax[0].text(len(piv) - 0.45, 0.515, "pre-registered threshold", color="#2f855a",
               fontsize=8.5, ha="right")
    # found by LOOKING: the line implies rho decided the SAFY gate. It did not.
    ax[0].text(0.02, 0.965, "SAFY was killed by AVAILABILITY, not by these bars:\n"
                            "only 2 of 6 dates have a same-day optical partner\n"
                            "(monsoon cloud, 11 May - 13 Oct), and neither is in\n"
                            "the June-September growth window SAFY models",
               transform=ax[0].transAxes, va="top", fontsize=8, color="#444",
               bbox=dict(boxstyle="round,pad=0.4", fc="#f7fafc", ec="#cbd5e0"))
    ax[0].set_xticks(xx)
    ax[0].set_xticklabels(piv.index, fontsize=9)
    ax[0].set_ylabel(r"within-crop Spearman $\rho$")
    ax[0].set_ylim(0, 0.85)
    ax[0].legend(fontsize=9, loc="upper right")
    ax[0].grid(alpha=0.3, axis="y")
    ax[0].set_title(r"T1 — $\gamma^0$ vs SAME-DAY Sentinel-2 NDVI, within crop"
                    "\nbajra collapses by 12 Nov because it is off the field", fontsize=11)

    v = pd.read_csv(TABLES / "p3_validate.csv")
    v3 = v[v.test == "V3"].copy()
    vals = v3.detail.str.extract(r"west ([+-][\d.]+)\s+east ([+-][\d.]+)").astype(float)
    xx = np.arange(len(v3))
    ax[1].bar(xx - 0.19, vals[0], 0.38, color="#805ad5", edgecolor="#222", label="west half")
    ax[1].bar(xx + 0.19, vals[1], 0.38, color="#d69e2e", edgecolor="#222", label="east half")
    ax[1].set_xticks(xx)
    # found by LOOKING: raw "1013" / "1112" read as counts, not dates
    ax[1].set_xticklabels([i.replace(" 1013", "\n13 Oct").replace(" 1112", "\n12 Nov")
                           for i in v3.item], fontsize=9)
    ax[1].set_ylabel(r"within-crop Spearman $\rho$")
    ax[1].set_ylim(0, 0.8)
    ax[1].legend(fontsize=9)
    ax[1].grid(alpha=0.3, axis="y")
    ax[1].set_title("V3 — the same relationship, in both spatial halves\n"
                    "east runs consistently higher, but every pair agrees in sign and size",
                    fontsize=11)

    fig.tight_layout()
    p = FIGURES / "p3_f4_witness.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def f5_climatology():
    """The one late approach that survived. The CONTROL is the point of the left panel:
    the raw farm signal and the non-farm reference move together, so only the difference
    can be about the crop."""
    c = pd.read_csv(TABLES / 'p3_climatology.csv')
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))

    ax[0].plot(c.year, c.vh_farm_db, 'o-', color='#2f855a', lw=2, label='farm pixels (signal)')
    ax[0].plot(c.year, c.vh_nonfarm_db, 's--', color='#a0aec0', lw=2,
               label='non-farm pixels (CONTROL)')
    ax[0].axvline(2025, color='#c53030', ls=':', lw=1.2)
    ax[0].set_ylabel('growing-season VH (dB)')
    ax[0].legend(fontsize=8.5, frameon=False)
    ax[0].set_title('They move TOGETHER — so the raw 2025 rise is the landscape,\n'
                    'not the crop. Only the difference can be about the crop.', fontsize=11)
    ax[0].grid(alpha=0.3)

    h = c[c.year < 2025]
    mu, sd = h.vh_diff_db.mean(), h.vh_diff_db.std(ddof=1)
    ax[1].axhspan(mu - sd, mu + sd, color='#bee3f8', alpha=0.7, label='2017–24 climatology ±1σ')
    ax[1].axhline(mu, color='#2b6cb0', lw=1.2)
    ax[1].plot(c.year, c.vh_diff_db, 'o-', color='#1a202c', lw=2)
    v25 = float(c[c.year == 2025].vh_diff_db.iloc[0])
    ax[1].plot([2025], [v25], 'o', ms=11, mfc='none', mec='#c53030', mew=2.2)
    ax[1].annotate(f'2025  z = {(v25-mu)/sd:+.2f}\nan ORDINARY growing season',
                   xy=(2025, v25), xytext=(2020.4, mu - 2.6 * sd), fontsize=9,
                   color='#c53030', arrowprops=dict(arrowstyle='->', color='#c53030'))
    ax[1].set_ylabel('VH  farm − non-farm  (dB)')
    ax[1].set_title('The differenced quantity: 2025 sits inside the nine-year norm.\n'
                    'adj_2025_26 = 1.00 becomes a measurement, not an assumption.', fontsize=11)
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8.5, frameon=False, loc='upper right')

    fig.suptitle('NINE YEARS OF SENTINEL-1 OVER SOKHDA, ONE TRACK — the growing season was '
                 'ordinary; October was not', fontsize=12, y=1.02)
    fig.tight_layout()
    p_ = FIGURES / 'p3_f5_climatology.png'
    fig.savefig(p_, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return p_


def f6_spatial():
    """A null result that only means something because the POSITIVE CONTROL is on the
    same axes. Incidence angle is a known landscape-scale gradient measured over the
    same 966 centroids in the same bins: it rises to its sill at ~1400 m. The ranking
    does not. That is what licenses reading the flat curve as 'field-scale'."""
    v = pd.read_csv(TABLES / 'p3_spatial_variogram.csv')
    s = pd.read_csv(TABLES / 'p3_spatial_stats.csv').set_index('stat')
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))

    style = {'hi_resid': ('#2f855a', 'o-', 'health-index ranking (subject)'),
             'incidence': ('#c05621', 's--', 'incidence angle (POSITIVE CONTROL)'),
             'jun_baseline': ('#a0aec0', '^:', 'June bare soil (control)')}
    for name, (col, mk, lab) in style.items():
        q = v[v.series == name]
        ax[0].plot(q.lag_m, q.gamma, mk, color=col, lw=2, ms=5, label=lab)
    ax[0].axvline(68, color='#4a5568', lw=1)
    ax[0].annotate('mean field\n68 m', xy=(68, 0.12), fontsize=8.5, color='#4a5568')
    ax[0].axhline(1.0, color='#cbd5e0', lw=1, zorder=0)
    ax[0].set_xlabel('separation between farm centroids (m)')
    ax[0].set_ylabel('semivariance (standardised)')
    ax[0].set_title('The ranking is flat from the first bin — all nugget, no structure.\n'
                    'Incidence, same farms same bins, has not reached its sill at 1.4 km.',
                    fontsize=11)
    ax[0].legend(fontsize=8.5, frameon=False, loc='lower right')
    ax[0].grid(alpha=0.3)

    land = float(s.loc['variance_split', 'landscape'])
    farm = float(s.loc['variance_split', 'farm_specific'])
    ax[1].barh(['where the farm is\n(landscape, not actionable)',
                'the farm itself\n(neighbour contrast, actionable)'],
               [land, farm], color=['#a0aec0', '#2f855a'])
    for i, val in enumerate([land, farm]):
        ax[1].text(val + 0.02, i, f'{val:.0%}', va='center', fontsize=11, weight='bold')
    ax[1].set_xlim(0, 1.15)
    ax[1].set_xlabel('share of within-crop ranking variance')
    ax[1].set_title("Moran's I = +0.073 (p = 0.001, z = +8.1): real, and small.\n"
                    'Significant is not the same as large.', fontsize=11)
    ax[1].grid(alpha=0.3, axis='x')

    fig.suptitle('IS THE RANKING ABOUT THE FARM, OR ABOUT WHERE THE FARM IS? — '
                 'field-scale, so the advice is actionable', fontsize=12, y=1.02)
    fig.tight_layout()
    p_ = FIGURES / 'p3_f6_spatial.png'
    fig.savefig(p_, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return p_


def f7_voi():
    """What each of the six acquisitions bought. The left panel's jump at the fourth date
    is mechanical, not fitted: senescence needs TWO late dates, so it does not exist until
    13 October joins 12 November."""
    g = pd.read_csv(TABLES / 'p3_voi_greedy.csv')
    l = pd.read_csv(TABLES / 'p3_voi_lodo.csv')
    s = pd.read_csv(TABLES / 'p3_voi_stats.csv').set_index('test')
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))

    ax[0].plot(g.n_dates, g.rho, 'o-', color='#2f855a', lw=2.5, ms=8)
    for _, r in g.iterrows():
        ax[0].annotate(pd.Timestamp(str(r.added)).strftime('%d %b'),
                       (r.n_dates, r.rho), textcoords='offset points', xytext=(0, -18),
                       ha='center', fontsize=8.5, color='#4a5568')
    q = s.loc['Q3_calendar_permutation']
    lo, hi, mid = float(q.lo), float(q.hi), float(q.rho)
    ax[0].axhspan(lo, hi, color='#fed7d7', alpha=0.5, zorder=0)
    ax[0].axhline(mid, color='#9b2c2c', ls='--', lw=1.6, zorder=1)
    ax[0].annotate(f'shuffled calendar, mean rho = {mid:.2f}\n'
                   '(same six scenes, time axis destroyed)',
                   xy=(1.05, mid), xytext=(1.05, mid - 0.22), fontsize=8.5, color='#9b2c2c',
                   arrowprops=dict(arrowstyle='->', color='#9b2c2c'))
    ax[0].text(1.05, -0.06, 'shaded: 5-95% of shuffles. Its top edge beats a single\n'
                            'real scene — some shuffles keep the peak date in place.',
               fontsize=8, color='#9b2c2c', ha='left', va='bottom')
    ax[0].annotate('senescence needs TWO late dates —\nit does not exist until here',
                   xy=(4, 0.9993), xytext=(1.9, 0.58), fontsize=9, color='#2f855a',
                   arrowprops=dict(arrowstyle='->', color='#2f855a'))
    ax[0].set_xticks(range(1, 6))
    ax[0].set_ylim(-0.12, 1.12)
    ax[0].set_xlabel('number of acquisitions (greedily chosen)')
    ax[0].set_ylabel('Spearman vs the shipped ranking')
    ax[0].set_title('One scene gets 0.857 of the ordering; the fourth closes the gap.\n'
                    'Destroying the time axis costs more than dropping four scenes.',
                    fontsize=11)
    ax[0].grid(alpha=0.3)

    lab = [pd.Timestamp(str(d)).strftime('%d %b') for d in l.date]
    col = ['#c05621' if m > 50 else '#2f855a' if m < 5 else '#a0aec0' for m in l.moved_of_97]
    ax[1].bar(lab, l.moved_of_97, color=col)
    for i, (m, r) in enumerate(zip(l.moved_of_97, l.rho)):
        ax[1].text(i, m + 1.5, f'{m}\nρ={r:.3f}', ha='center', fontsize=9)
    ax[1].set_ylim(0, 92)
    ax[1].set_ylabel('farms leaving the 97-farm attention list')
    ax[1].set_title('Drop one date, rebuild. 14 August IS the index;\n'
                    '6 June moves a single farm.', fontsize=11)
    ax[1].grid(alpha=0.3, axis='y')

    fig.suptitle('WHAT DID EACH OF THE SIX ACQUISITIONS BUY? — one peak-canopy scene, '
                 'plus a late pair that measures senescence', fontsize=12, y=1.02)
    fig.tight_layout()
    p_ = FIGURES / 'p3_f7_voi.png'
    fig.savefig(p_, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return p_


def f8_speckle():
    """Left panel is the CONTROL, and it is the reason the right panel is readable at all:
    speckle noise reaches the index in proportion to how few pixels a parcel has. The first
    version of that control was mis-specified and failed; this is the corrected one."""
    fm = pd.read_csv(TABLES / 'p3_speckle_farm.csv')
    sw = pd.read_csv(TABLES / 'p3_speckle_sweep.csv')
    nom = sw[sw.looks_per_pixel == 1.0].iloc[0]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))

    ax[0].scatter(fm.npix_peak.clip(lower=1), fm.index_sd_F1, s=9, alpha=0.35,
                  color='#4a5568', edgecolors='none')
    for lo, hi in [(0, 20), (20, 50), (50, 150), (150, 10 ** 9)]:
        m = (fm.npix_peak >= lo) & (fm.npix_peak < hi)
        if m.sum():
            ax[0].plot([max(lo, 1), min(hi, fm.npix_peak.max())],
                       [fm.index_sd_F1[m].mean()] * 2, color='#c05621', lw=3)
    ax[0].set_xscale('log')
    ax[0].set_xlabel('pixels in the parcel, peak-canopy date (log)')
    ax[0].set_ylabel('index spread across re-flights (points)')
    ax[0].set_title('CONTROL: speckle reaches the index, and small parcels\n'
                    f'pay for it — rho = {nom.c2b_sd_vs_npix:+.2f}, 12.2 pts vs 4.1',
                    fontsize=11)
    ax[0].grid(alpha=0.3, which='both')

    li = fm[fm.on_attention_list].sort_values('retain_F1.0', ascending=False)
    r = li['retain_F1.0'].to_numpy()
    col = ['#2f855a' if v >= 0.8 else '#c05621' if v < 0.5 else '#a0aec0' for v in r]
    ax[1].bar(range(len(r)), r, color=col, width=1.0)
    ax[1].axhline(float(nom.chance_floor), color='#9b2c2c', ls='--', lw=1.5)
    ax[1].annotate('chance floor 10% — a list drawn at random',
                   xy=(2, float(nom.chance_floor) + 0.03), fontsize=8.5, color='#9b2c2c')
    ax[1].text(19, 0.55, f'{int((r >= 0.8).sum())} solid\nnamed in >=80%\nof re-flights',
               fontsize=10, color='white', ha='center', va='center', weight='bold')
    ax[1].text(82, 0.72, f'{int((r < 0.5).sum())} named in\nunder half',
               fontsize=10, color='#c05621', ha='center', va='center', weight='bold')
    ax[1].set_xlabel('the 97 farms we name, sorted by stability')
    ax[1].set_ylabel('fraction of re-flights that still name the farm')
    ax[1].set_ylim(0, 1.02)
    ax[1].set_title(f'Overall {nom.retention:.0%} of the attention list survives a re-flight.\n'
                    'The list is real; a third of the names are not.', fontsize=11)
    ax[1].grid(alpha=0.3, axis='y')

    fig.suptitle('WOULD THE SAME 97 FARMS BE NAMED IF WE RE-FLEW THE MISSION? — '
                 'speckle alone, same crop, same day', fontsize=12, y=1.02)
    fig.tight_layout()
    p_ = FIGURES / 'p3_f8_speckle.png'
    fig.savefig(p_, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return p_


def f9_edge():
    """Left panel is the alarming fact; right panel is the control that defuses it. Half of a
    typical parcel is boundary pixel — and removing every one of them moves the ranking no
    more than removing the same COUNT of pixels at random. The two right-hand pairs sitting on
    top of each other IS the finding, so they must be readable as identical at a glance."""
    fm = pd.read_csv(TABLES / 'p3_edge_farm.csv')
    st = pd.read_csv(TABLES / 'p3_edge_stats.csv').iloc[0]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))

    ax[0].scatter(fm.npix_full.clip(lower=1), fm.ring_frac, s=9, alpha=0.35,
                  color='#4a5568', edgecolors='none')
    for lo, hi in [(0, 20), (20, 50), (50, 150), (150, 10 ** 9)]:
        m = (fm.npix_full >= lo) & (fm.npix_full < hi)
        if m.sum():
            ax[0].plot([max(lo, 1), min(hi, fm.npix_full.max())],
                       [fm.ring_frac[m].median()] * 2, color='#c05621', lw=3)
    ax[0].axhline(float(st.median_ring_frac), color='#9b2c2c', ls='--', lw=1.5)
    ax[0].annotate(f'median parcel is {st.median_ring_frac:.0%} boundary pixel',
                   xy=(1.3, float(st.median_ring_frac) + 0.03), fontsize=9, color='#9b2c2c')
    ax[0].annotate(f'{int(st.farms_no_interior)} farms have NO interior pixel at all',
                   xy=(1.3, 0.965), fontsize=9, color='#2b6cb0')
    ax[0].set_xscale('log')
    ax[0].set_xlabel('pixels in the parcel, peak-canopy date (log)')
    ax[0].set_ylabel('fraction of the parcel that is boundary')
    ax[0].set_ylim(0, 1.02)
    ax[0].set_title('Every boundary pixel mixes the farm with the bund,\n'
                    'the track and the neighbour — and there are a lot of them', fontsize=11)
    ax[0].grid(alpha=0.3, which='both')

    labels = ['rank correlation\nwith the shipped index', 'attention list\nretained']
    core = [float(st.rho_core_vs_shipped), float(st.retention_core)]
    rand = [float(st.rho_random_vs_shipped), float(st.retention_random)]
    x = np.arange(2)
    ax[1].bar(x - 0.18, core, 0.34, color='#2b6cb0', label='drop the BOUNDARY pixels')
    ax[1].bar(x + 0.18, rand, 0.34, color='#a0aec0',
              label='CONTROL: drop the same NUMBER at random')
    for xi, (a, b) in enumerate(zip(core, rand)):
        ax[1].text(xi - 0.18, a + 0.02, f'{a:.3f}', ha='center', fontsize=9.5, color='#2b6cb0')
        ax[1].text(xi + 0.18, b + 0.02, f'{b:.3f}', ha='center', fontsize=9.5, color='#4a5568')
    ax[1].axhline(float(st.chance_floor), color='#9b2c2c', ls='--', lw=1.2)
    ax[1].annotate('chance floor 10%', xy=(1.28, float(st.chance_floor) + 0.02),
                   fontsize=8.5, color='#9b2c2c')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, fontsize=10)
    ax[1].set_ylim(0, 1.05)
    # legend sat on top of the bars and hid the chance floor; pushed below the axis [F9]
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9,
                 frameon=False)
    ax[1].set_title('The pairs coincide: the ranking is not an artefact of\n'
                    'parcel boundaries, it is only sample size', fontsize=11)
    ax[1].grid(alpha=0.3, axis='y')

    fig.suptitle('IS THE ATTENTION LIST MEASURING THE CROP, OR THE PARCEL BOUNDARY? — '
                 f'rho(index, boundary fraction) = {st.rho_index_vs_ring_frac:+.3f}',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    p_ = FIGURES / 'p3_f9_edge.png'
    fig.savefig(p_, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return p_


def main():
    for fn in (f1_deliverable, f2_uncertainty, f3_d1, f4_witness, f5_climatology,
               f6_spatial, f7_voi, f8_speckle, f9_edge):
        p = fn()
        log("p3_fig", name=p.name, kb=round(p.stat().st_size / 1024, 1))
    print(f"\n{len(list(FIGURES.glob('p3_*.png')))} figures in {FIGURES}")
    print("NOW OPEN THEM AND LOOK. The figure is the check on the prose.")


if __name__ == "__main__":
    main()
