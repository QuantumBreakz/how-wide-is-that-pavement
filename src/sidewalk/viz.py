"""Figures: the strip maps the brief asks for, plus the diagnostic plots."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from . import config as C

BAND_ORDER = C.BAND_NAMES
GREY = "#9aa0a6"


def strip_map(segs_left: pd.DataFrame, segs_right: pd.DataFrame, road: str,
              axis_length: float, carriageway: pd.DataFrame = None,
              ax=None, show_labels: bool = True, band_colours: dict = None,
              label_min_len: float = 24.0, road_band: float = 0.42):
    """The deliverable figure: left side above the road, right side below.

    x is position along the street in metres, matching the brief's example.
    Bar height is footway width to scale; colour is the UK band. The grey strip
    down the middle is the carriageway, annotated with its measured width.
    Stretches with no reliable measurement are left hatched rather than
    interpolated over -- a gap in the strip is a gap in the evidence.
    """
    cols = band_colours or C.BAND_COLOURS
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 4.8))

    def draw(segs, sign):
        for _, r in segs.iterrows():
            w = float(r.width_median)
            short = bool(r.get("short_run", False))
            y0 = road_band if sign > 0 else -road_band - w
            ax.add_patch(Rectangle((r.s_start, y0), max(r.length_m, 0.5), w,
                                   facecolor=cols.get(r.band, GREY),
                                   edgecolor="white", linewidth=0.6,
                                   hatch="///" if short else None,
                                   alpha=0.5 if short else 1.0, zorder=3))
            if show_labels and r.length_m >= label_min_len and w > 0.45:
                ax.text(r.s_start + r.length_m / 2, y0 + w / 2, f"{w:.1f}",
                        ha="center", va="center", fontsize=7.5,
                        color="white", fontweight="bold", zorder=4)

    # --- coverage: hatch the stretches we could not measure ------------------
    for segs, sign in ((segs_left, 1), (segs_right, -1)):
        covered = [(r.s_start, r.s_end) for _, r in segs.sort_values("s_start").iterrows()]
        cursor = 0.0
        gaps = []
        for a, b in covered:
            if a - cursor > 1.5:
                gaps.append((cursor, a))
            cursor = max(cursor, b)
        if axis_length - cursor > 1.5:
            gaps.append((cursor, axis_length))
        for a, b in gaps:
            y0 = road_band if sign > 0 else -road_band - 0.6
            ax.add_patch(Rectangle((a, y0), b - a, 0.6, facecolor="none",
                                   edgecolor="#c0c4c9", hatch="xxx", linewidth=0.0, zorder=2))

    # --- the carriageway itself ---------------------------------------------
    ax.add_patch(Rectangle((0, -road_band), axis_length, 2 * road_band,
                           facecolor="#d5d9de", edgecolor="none", zorder=1))
    if carriageway is not None and len(carriageway):
        cw = carriageway.dropna(subset=["carriageway_w"]).sort_values("s")
        if len(cw):
            med = float(np.nanmedian(cw.carriageway_w))
            lo, hi = np.nanpercentile(cw.carriageway_w, [10, 90])
            ax.text(axis_length * 0.5, 0,
                    f"CARRIAGEWAY   median {med:.1f} m   (P10-P90 {lo:.1f}-{hi:.1f} m)",
                    ha="center", va="center", fontsize=8, color="#3c4043",
                    fontweight="bold", zorder=6)

    draw(segs_left, +1)
    draw(segs_right, -1)

    hi_w = max(float(segs_left.width_median.max()) if len(segs_left) else 1.0,
               float(segs_right.width_median.max()) if len(segs_right) else 1.0, 1.0)
    top = road_band + hi_w * 1.12
    ax.set_ylim(-top, top)
    ax.set_xlim(-axis_length * 0.012, axis_length * 1.012)
    ax.set_xlabel("Position along road (m)")
    ax.set_ylabel("Footway width (m)")
    step = 1.0 if hi_w <= 6 else 2.0
    ticks = np.arange(0, np.floor(hi_w) + step, step)
    ax.set_yticks(np.r_[-(ticks + road_band)[::-1], ticks + road_band])
    ax.set_yticklabels([f"{v:.0f}" for v in ticks[::-1]] + [f"{v:.0f}" for v in ticks])
    ax.set_title(road, loc="left", fontsize=13, fontweight="bold")
    ax.text(axis_length * 1.005, road_band + hi_w * 0.55, "LEFT", fontsize=8,
            color="#5f6368", fontweight="bold", rotation=90, va="center")
    ax.text(axis_length * 1.005, -road_band - hi_w * 0.55, "RIGHT", fontsize=8,
            color="#5f6368", fontweight="bold", rotation=90, va="center")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    return ax


def band_legend(ax, loc="upper right", ncol=5):
    handles = [Line2D([0], [0], marker="s", linestyle="", markersize=9,
                      markerfacecolor=C.BAND_COLOURS[b], markeredgecolor="white", label=b)
               for b in BAND_ORDER]
    handles.append(Line2D([0], [0], marker="s", linestyle="", markersize=9,
                          markerfacecolor="white", markeredgecolor="#c0c4c9",
                          label="no reliable measurement"))
    ax.legend(handles=handles, loc=loc, ncol=ncol, fontsize=8, frameon=False,
              handletextpad=0.4, columnspacing=1.2)


def confidence_strip(segs_left, segs_right, road, axis_length, ax=None):
    """Companion strip showing where we trust the numbers above."""
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 1.5))
    cmap = matplotlib.colormaps["RdYlGn"]
    for segs, y0 in ((segs_left, 0.55), (segs_right, 0.05)):
        for _, r in segs.iterrows():
            c = cmap(float(np.clip(r.confidence, 0, 1)))
            ax.add_patch(Rectangle((r.s_start, y0), max(r.length_m, 0.6), 0.38,
                                   facecolor=c, edgecolor="white", lw=0.5))
    ax.set_xlim(-axis_length * 0.01, axis_length * 1.01)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.24, 0.74])
    ax.set_yticklabels(["right", "left"], fontsize=8)
    ax.set_xlabel("Position along road (m)")
    ax.set_title("Measurement confidence", loc="left", fontsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def cross_section_profile(xs: pd.DataFrame, road: str, ax=None):
    """Total street width decomposed into its three measured components."""
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 3.4))
    d = xs.sort_values("s")
    ok = d.left_valid & d.right_valid
    d = d[ok]
    ax.fill_between(d.s, 0, d.left_w, color=C.BAND_COLOURS["good"], alpha=.85, label="left footway")
    ax.fill_between(d.s, d.left_w, d.left_w + d.carriageway_w, color="#9aa7b4", alpha=.9, label="carriageway")
    ax.fill_between(d.s, d.left_w + d.carriageway_w, d.total_w, color=C.BAND_COLOURS["minimum acceptable"],
                    alpha=.85, label="right footway")
    # Only draw the frontage line where the closure test actually applies -- at a
    # junction mouth the nearest building is round a corner and the line is noise.
    f = d.copy()
    f.loc[~f.closure_class.isin(["GOOD", "INVESTIGATE"]), "frontage_w"] = np.nan
    ax.plot(f.s, f.frontage_w, color="#b00020", lw=1.1, ls="--",
            label="frontage to frontage (where closure applies)")
    ax.set_xlabel("Position along road (m)")
    ax.set_ylabel("Width (m)")
    ax.set_title(f"{road} - cross-section composition", loc="left", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=4, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return ax
