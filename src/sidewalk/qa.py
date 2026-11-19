"""Quality assurance: neighbour cross-checking and cross-section closure.

Two independent ideas.

1. *Neighbour cross-check.* A pavement does not change width by a metre and change
   back again two metres later. A one-sample spike is a geometry artefact (a
   vehicle crossover polygon, a sliver, a ray slipping through a gate). A real
   change persists. We therefore test every sample against its neighbours with a
   robust (median/MAD) rule and require persistence before accepting a change.

2. *Cross-section closure.* For a street enclosed by buildings the three measured
   components should account for the whole street:

       closure residual = frontage-to-frontage - (left footway + carriageway + right footway)

   A residual near zero means we have explained the street. A large residual means
   there is material in the cross-section we have not accounted for -- a forecourt,
   a service strip, railway land -- and the width should be treated with caution.

   The GOOD / INVESTIGATE / AMBIGUOUS thresholds below are OUR QA thresholds,
   chosen from the sensitivity analysis. They are not published UK standards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

CLOSURE_LABELS = ["GOOD", "INVESTIGATE", "AMBIGUOUS", "N/A"]


def neighbour_check(df: pd.DataFrame, window: float = C.NEIGHBOUR_WINDOW,
                    tol: float = C.NEIGHBOUR_TOL, k: float = C.NEIGHBOUR_MAD_K,
                    spacing: float = C.SAMPLE_SPACING) -> pd.DataFrame:
    """Flag isolated, non-persistent width spikes, per road and side."""
    df = df.copy()
    df["neighbour_outlier"] = False
    df["neighbour_dev"] = np.nan

    half = max(int(round(window / spacing)), 1)
    win = 2 * half + 1

    for _key, grp in df.groupby(["road", "side"], sort=False):
        g = grp.sort_values("s")
        w = g.foot_w.where(g.valid).astype(float)
        if w.notna().sum() < 5:
            continue
        med = w.rolling(win, center=True, min_periods=3).median()
        mad = (w - med).abs().rolling(win, center=True, min_periods=3).median()
        scale = np.maximum(mad.to_numpy() * 1.4826, 0.05)
        dev = (w - med).to_numpy()
        thresh = np.maximum(tol, k * scale)
        cand = np.abs(dev) > thresh

        # Persistence: a genuine change is shared by at least one immediate
        # neighbour in the same direction. A lone disagreeing sample is a spike.
        sgn = np.sign(dev)
        prev_same = np.r_[False, cand[:-1] & (sgn[:-1] == sgn[1:])]
        next_same = np.r_[cand[1:] & (sgn[1:] == sgn[:-1]), False]
        isolated = cand & ~(prev_same | next_same)

        df.loc[g.index, "neighbour_dev"] = dev
        df.loc[g.index, "neighbour_outlier"] = isolated

    df.loc[df.neighbour_outlier & df.valid, "reason"] = "neighbour_spike"
    df.loc[df.neighbour_outlier, "valid"] = False
    return df


def classify_closure(paired: pd.DataFrame,
                     good: float = C.CLOSURE_GOOD,
                     invest: float = C.CLOSURE_INVESTIGATE) -> pd.DataFrame:
    """Attach the closure class and, where it fails, what the residual is made of."""
    p = paired.copy()
    r = p.closure_residual.abs()
    lab = np.where(~np.isfinite(r), "N/A",
          np.where(r < good, "GOOD",
          np.where(r < invest, "INVESTIGATE", "AMBIGUOUS")))
    p["closure_class"] = lab

    def cause(row):
        if row.closure_class in ("GOOD", "N/A"):
            return None
        parts = []
        for side in ("left", "right"):
            gc = row.get(f"{side}_gap_class")
            bc = row.get(f"{side}_back")
            if gc and gc not in ("none", None):
                parts.append(f"{side}:{gc} behind footway")
            elif bc and bc != "building":
                parts.append(f"{side}:back edge is {bc}")
        nd = row.get("normal_disagree_deg", 0)
        if nd and nd > C.MAX_NORMAL_DISAGREE:
            parts.append(f"kerb normals disagree {nd:.0f} deg")
        elif nd and nd > 5:
            parts.append(f"street curving ({nd:.0f} deg between kerbs)")
        for side in ("left", "right"):
            if row.get(f"{side}_method") == "fan_fallback":
                parts.append(f"{side}:kerb fit rejected (fan fallback)")
        return "; ".join(parts) if parts else "unattributed"

    p["closure_cause"] = p.apply(cause, axis=1)
    p.loc[p.normal_disagree_deg > C.MAX_NORMAL_DISAGREE, "geometry_warning"] = (
        f"kerb normals disagree > {C.MAX_NORMAL_DISAGREE:.0f} deg")
    return p


def road_qa_summary(df: pd.DataFrame, paired: pd.DataFrame) -> dict:
    """Headline honesty metrics for one road."""
    tot = len(df)
    out = dict(
        road=df.road.iloc[0],
        n_samples=tot,
        pct_valid=100 * df.valid.mean(),
        pct_kerb_normal=100 * (df.method == "kerb_normal").mean(),
        pct_fan_fallback=100 * (df.method == "fan_fallback").mean(),
        median_uncertainty_m=float(np.nanmedian(df.loc[df.valid, "w_uncert"])),
        p90_uncertainty_m=float(np.nanpercentile(df.loc[df.valid, "w_uncert"].dropna(), 90))
        if df.loc[df.valid, "w_uncert"].notna().any() else np.nan,
    )
    for reason, n in df.reason.value_counts().items():
        out[f"rejected_{reason}"] = int(n)
    if len(paired):
        cc = paired.closure_class.value_counts(normalize=True) * 100
        for lab in CLOSURE_LABELS:
            out[f"closure_{lab}_pct"] = float(cc.get(lab, 0.0))
        out["median_closure_residual_m"] = float(np.nanmedian(paired.closure_residual))
    return out
