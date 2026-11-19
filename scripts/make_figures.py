"""All non-strip-map figures for the notebook and report."""
from __future__ import annotations

import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sidewalk import config as C

OUT = pathlib.Path("outputs"); FIG = OUT / "figures"; TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
INK, MUTED = "#1d1f21", "#6b7177"


def f_estimators():
    """The single most important chart: naive vs kerb-normal vs fan."""
    s = pd.read_parquet(TAB / "samples.parquet")
    v = s[s.valid]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    bins = np.linspace(0, 14, 71)
    for col, lab, c in [("w_perp", "centreline-perpendicular (naive)", "#b00020"),
                        ("w_fan", "±30° ray-fan minimum", "#edae49"),
                        ("w_kerb", "kerb-normal (ours)", "#2e6f95")]:
        d = v[col].dropna()
        ax[0].hist(d.clip(0, 14), bins=bins, histtype="step", lw=1.6, color=c,
                   label=f"{lab}\n  median {d.median():.2f} m, max {d.max():.1f} m")
    ax[0].set_xlabel("Footway width (m)"); ax[0].set_ylabel("Cross-sections")
    ax[0].legend(fontsize=7.5, frameon=False)
    ax[0].set_title("Three estimators on the same 6,800 cross-sections", loc="left", fontweight="bold")

    d = v.dropna(subset=["w_kerb", "w_perp"])
    ax[1].scatter(d.w_kerb, d.w_perp, s=3, alpha=.18, color="#2e6f95", lw=0)
    lim = 14
    ax[1].plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--")
    ax[1].set_xlim(0, lim); ax[1].set_ylim(0, lim)
    ax[1].set_xlabel("kerb-normal width (m)"); ax[1].set_ylabel("centreline-perpendicular (m)")
    over = 100 * (d.w_perp > d.w_kerb + 0.25).mean()
    ax[1].set_title(f"Naive over-reads on {over:.0f}% of cross-sections", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "10_estimators.png"); plt.close(fig)


def f_closure():
    xs = pd.read_csv(TAB / "cross_sections.csv")
    order = ["GOOD", "INVESTIGATE", "AMBIGUOUS", "N/A"]
    cols = {"GOOD": "#2e7d32", "INVESTIGATE": "#edae49", "AMBIGUOUS": "#b00020", "N/A": "#c0c4c9"}
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"width_ratios": [1.5, 1]})
    roads = sorted(xs.road.unique())
    for i, road in enumerate(roads):
        d = xs[(xs.road == road)].dropna(subset=["closure_residual"])
        if not len(d):
            continue
        y = np.random.normal(i, 0.11, len(d))
        ax[0].scatter(d.closure_residual.clip(-3, 12), y, s=5, alpha=.35, lw=0,
                      c=[cols.get(k, "#999") for k in d.closure_class])
    for t, ls in [(C.CLOSURE_GOOD, "-"), (C.CLOSURE_INVESTIGATE, "--")]:
        for sgn in (1, -1):
            ax[0].axvline(sgn * t, color=MUTED, lw=1, ls=ls)
    ax[0].set_yticks(range(len(roads))); ax[0].set_yticklabels(roads, fontsize=8)
    ax[0].set_xlabel("closure residual  =  frontage-to-frontage  −  (left + carriageway + right)   [m]")
    ax[0].set_title("Cross-section closure by road", loc="left", fontweight="bold")
    ax[0].text(C.CLOSURE_GOOD, len(roads) - .4, " QA thresholds (ours, not a UK standard)",
               fontsize=7.5, color=MUTED)

    share = (xs.groupby(["road", "closure_class"]).size().unstack(fill_value=0))
    share = share.reindex(columns=[c for c in order if c in share.columns]).fillna(0)
    share = share.div(share.sum(axis=1), axis=0) * 100
    left = np.zeros(len(share))
    for c in share.columns:
        ax[1].barh(share.index, share[c], left=left, color=cols[c], label=c, height=.7)
        left += share[c].values
    ax[1].set_yticklabels([]); ax[1].set_xlabel("% of cross-sections")
    ax[1].legend(fontsize=7.5, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(.5, -.28))
    ax[1].set_title("Closure class share", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "11_closure.png"); plt.close(fig)


def f_os_validation():
    lk = pd.read_csv(TAB / "validation_os_carriageway_by_link.csv")
    sc = lk[(lk.formofway == "Single Carriageway") & (lk.n_xs >= 5)]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax[0].scatter(sc.os_averagewidth_m, sc.ours_median_m, s=26, alpha=.75,
                  color="#2e6f95", edgecolor="white", lw=.5)
    lim = [0, max(sc.os_averagewidth_m.max(), sc.ours_median_m.max()) * 1.08]
    ax[0].plot(lim, lim, color=MUTED, ls="--", lw=1)
    ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("OS Highways published averagewidth (m)")
    ax[0].set_ylabel("our measured carriageway (m)")
    r = np.corrcoef(sc.os_averagewidth_m, sc.ours_median_m)[0, 1]
    ax[0].set_title(f"n={len(sc)} links   r={r:.3f}   MAE={sc.residual_m.abs().mean():.2f} m",
                    loc="left", fontweight="bold")

    ar = pd.read_csv(TAB / "validation_areal.csv")
    ax[1].hist(ar.coverage.clip(0, 1), bins=np.linspace(0, 1, 41), color="#66a182",
               edgecolor="white", lw=.4)
    ax[1].axvline(ar.coverage.median(), color="#b00020", lw=1.4)
    ax[1].set_xlabel("fraction of the measured ribbon lying on surveyed footway polygons")
    ax[1].set_ylabel("segments")
    ax[1].set_title(f"Areal reconciliation — median {ar.coverage.median():.3f}",
                    loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "12_validation.png"); plt.close(fig)


def f_sensitivity():
    s = pd.read_csv(TAB / "validation_sensitivity.csv")
    piv = s.pivot(index="setting", columns="road", values="median_width_delta_m")
    piv = piv.reindex(piv.abs().max(axis=1).sort_values().index)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    y = np.arange(len(piv))
    for i, road in enumerate(piv.columns):
        ax.scatter(piv[road], y + (i - 1) * 0.18, s=46, label=road,
                   edgecolor="white", lw=.6)
    ax.axvline(0, color=INK, lw=1)
    for t in (-0.25, 0.25):
        ax.axvline(t, color=MUTED, lw=1, ls=":")
    ax.set_yticks(y); ax.set_yticklabels(piv.index, fontsize=8.5)
    ax.set_xlabel("change in median footway width vs baseline (m)")
    ax.set_title("Sensitivity: the answer does not depend on the knobs", loc="left", fontweight="bold")
    ax.text(0.25, len(piv) - .5, "  0.25 m = our minimum meaningful change", fontsize=7.5, color=MUTED)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "13_sensitivity.png"); plt.close(fig)


def f_results():
    seg = pd.read_csv(TAB / "segments.csv")
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    roads = seg.groupby("road").apply(
        lambda g: pd.Series({b: g.loc[g.band == b, "length_m"].sum() for b in C.BAND_NAMES}))
    roads = roads.div(roads.sum(axis=1), axis=0) * 100
    roads = roads.loc[roads[C.BAND_NAMES[0]].add(roads[C.BAND_NAMES[1]]).sort_values().index]
    left = np.zeros(len(roads))
    for b in C.BAND_NAMES:
        ax[0].barh(roads.index, roads[b], left=left, color=C.BAND_COLOURS[b], label=b, height=.72)
        left += roads[b].values
    ax[0].set_xlabel("% of measured footway length"); ax[0].tick_params(labelsize=8)
    ax[0].legend(fontsize=7, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.5, -.32))
    ax[0].set_title("UK width band by road", loc="left", fontweight="bold")

    ax[1].hist(seg.sigma_m, bins=40, color="#9aa7b4", edgecolor="white", lw=.4)
    ax[1].axvline(seg.sigma_m.median(), color="#b00020", lw=1.4)
    ax[1].set_xlabel("segment 1σ uncertainty (m)"); ax[1].set_ylabel("segments")
    ax[1].set_title(f"Uncertainty — median {seg.sigma_m.median():.2f} m", loc="left", fontweight="bold")

    sc = ax[2].scatter(seg.width_median, seg.confidence, c=seg.artefact_risk,
                       cmap="YlOrRd", s=24, edgecolor="white", lw=.4, vmin=0, vmax=.8)
    ax[2].set_xlabel("segment width (m)"); ax[2].set_ylabel("confidence")
    plt.colorbar(sc, ax=ax[2], label="artefact risk", fraction=.046)
    ax[2].set_title("Wide does not mean wrong — but it correlates", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "14_results.png"); plt.close(fig)


def f_rejections():
    s = pd.read_parquet(TAB / "samples.parquet")
    r = s.reason.value_counts()
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    labels = {"ray_reentered_road": "ray re-entered the road (junction mouth)",
              "carriageway_leak": "corridor implausibly wide (leak)",
              "junction_mouth": "transect crosses another street",
              "neighbour_spike": "isolated spike vs neighbours",
              "axis_not_on_carriageway": "axis point not on a carriageway",
              "footway_beyond_plausible_strip": "pavement beyond a plausible kerbside strip"}
    r.index = [labels.get(i, i) for i in r.index]
    r = r.sort_values()
    ax.barh(r.index, r.values, color="#c1553f", height=.68)
    for i, v in enumerate(r.values):
        ax.text(v + 6, i, f"{v}  ({100*v/len(s):.1f}%)", va="center", fontsize=8, color=MUTED)
    ax.set_xlabel(f"cross-sections rejected (of {len(s):,} measured)")
    ax.set_title(f"Every rejection has a named reason — {100*s.valid.mean():.1f}% retained",
                 loc="left", fontweight="bold")
    ax.set_xlim(0, r.max() * 1.35)
    fig.tight_layout(); fig.savefig(FIG / "15_rejections.png"); plt.close(fig)


if __name__ == "__main__":
    for f in (f_estimators, f_closure, f_os_validation, f_sensitivity, f_results, f_rejections):
        try:
            f(); print("ok", f.__name__)
        except Exception as e:
            print("FAIL", f.__name__, type(e).__name__, e)
