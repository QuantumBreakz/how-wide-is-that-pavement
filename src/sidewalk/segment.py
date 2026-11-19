"""Turning a per-2-metre width profile into segments.

The brief asks for segments "wherever the width changes meaningfully", not a
single average and not just a re-colouring of the classification bands. So we run
an actual change-point detection on the continuous width signal:

  * exact dynamic programming with an L2 (change-in-mean) cost,
  * a penalty per change point so we do not over-segment noise,
  * a hard minimum segment length,
  * a post-merge that removes any change smaller than MIN_MEAN_SHIFT.

Only then is each segment classified into the brief's UK width bands, from its
median (not mean) width.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _pelt_l2(y: np.ndarray, penalty: float, min_len: int) -> list[int]:
    """Exact change-point detection for a change in mean, L2 cost.

    Returns the interior breakpoint indices. O(n^2), which is nothing at our
    sizes (a few hundred samples per road side).
    """
    n = len(y)
    if n < 2 * min_len:
        return []
    cs = np.concatenate([[0.0], np.cumsum(y)])
    cs2 = np.concatenate([[0.0], np.cumsum(y * y)])

    def cost(i, j):                      # cost of y[i:j]
        m = j - i
        s, s2 = cs[j] - cs[i], cs2[j] - cs2[i]
        return s2 - s * s / m

    F = np.full(n + 1, np.inf)
    F[0] = -penalty
    prev = np.zeros(n + 1, dtype=int)
    for j in range(min_len, n + 1):
        for i in range(0, j - min_len + 1):
            if not np.isfinite(F[i]):
                continue
            c = F[i] + cost(i, j) + penalty
            if c < F[j]:
                F[j], prev[j] = c, i
    bkps, j = [], n
    while j > 0:
        j = prev[j]
        if j > 0:
            bkps.append(j)
    return sorted(bkps)


def _merge_small_shifts(y, bkps, min_shift):
    """Drop any breakpoint whose two sides differ by less than min_shift."""
    edges = [0] + list(bkps) + [len(y)]
    changed = True
    while changed and len(edges) > 2:
        changed = False
        means = [np.median(y[edges[i]:edges[i + 1]]) for i in range(len(edges) - 1)]
        diffs = [abs(means[i + 1] - means[i]) for i in range(len(means) - 1)]
        if diffs:
            k = int(np.argmin(diffs))
            if diffs[k] < min_shift:
                edges.pop(k + 1)
                changed = True
    return edges[1:-1]


def segment_series(s: np.ndarray, y: np.ndarray, spacing: float = C.SAMPLE_SPACING,
                   min_len_m: float = C.MIN_SEGMENT_LEN,
                   min_shift: float = C.MIN_MEAN_SHIFT,
                   pen_k: float = C.PELT_PENALTY_K,
                   max_gap_m: float = None) -> list[dict]:
    """Segment one (chainage, width) series, splitting first at data gaps."""
    if max_gap_m is None:
        max_gap_m = C.MAX_SEGMENT_GAP
    order = np.argsort(s)
    s, y = np.asarray(s)[order], np.asarray(y)[order]
    ok = np.isfinite(y)
    s, y = s[ok], y[ok]
    if len(y) == 0:
        return []

    # Split into runs of contiguous coverage; a gap is a hard break.
    breaks = np.where(np.diff(s) > max_gap_m)[0]
    runs = np.split(np.arange(len(s)), breaks + 1)

    min_len = max(int(round(min_len_m / spacing)), 2)
    out = []
    for run in runs:
        if len(run) == 0:
            continue
        ys, ss = y[run], s[run]
        if len(ys) < min_len:
            # A run of valid data shorter than the minimum segment length is not a
            # segment -- it is a scrap of street between two rejected stretches.
            # We keep it, but flagged, and exclude flagged scraps from the widest-
            # segment table where they would otherwise dominate.
            out.append({**_pack(ss, ys, spacing), "short_run": True})
            continue
        # Robust noise scale from lag-1 differences (immune to real steps).
        # Noise scale from lag-1 differences, floored at the survey noise. Without
        # the floor a smooth stretch looks noiseless and every wobble becomes a
        # "significant" change point.
        sigma = 1.4826 * np.median(np.abs(np.diff(ys))) / np.sqrt(2) if len(ys) > 2 else np.std(ys)
        sigma = max(float(sigma), C.SIGMA_SURVEY)
        penalty = pen_k * sigma ** 2 * np.log(max(len(ys), 2))
        bkps = _pelt_l2(ys, penalty, min_len)
        bkps = _merge_small_shifts(ys, bkps, min_shift)
        edges = [0] + list(bkps) + [len(ys)]
        for a, b in zip(edges[:-1], edges[1:], strict=False):
            out.append(_pack(ss[a:b], ys[a:b], spacing))
    return out


def _pack(ss, ys, spacing: float = C.SAMPLE_SPACING) -> dict:
    # A segment covering samples at 10, 11, 12 m spans 2 m between its end
    # samples but represents 3 m of street, so one sample spacing is added. That
    # has to be the spacing actually used, not the default: the borough-scale run
    # samples every 2 m, and hardcoding 1 m understated every segment there.
    return dict(
        short_run=False,
        s_start=float(ss.min()),
        s_end=float(ss.max()),
        length_m=float(ss.max() - ss.min() + spacing),
        n=int(len(ys)),
        width_median=float(np.median(ys)),
        width_mean=float(np.mean(ys)),
        width_p10=float(np.percentile(ys, 10)),
        width_p90=float(np.percentile(ys, 90)),
        width_min=float(np.min(ys)),
        width_max=float(np.max(ys)),
        width_iqr=float(np.percentile(ys, 75) - np.percentile(ys, 25)),
    )


def segment_road_side(df_side: pd.DataFrame, **kw) -> pd.DataFrame:
    """Segment one side of one road from its valid transect samples."""
    d = df_side[df_side.valid].sort_values("s")
    segs = segment_series(d.s.to_numpy(), d.foot_w.to_numpy(), **kw)
    if not segs:
        return pd.DataFrame()
    out = pd.DataFrame(segs)
    out["road"] = df_side.road.iloc[0]
    out["side"] = df_side.side.iloc[0]
    out["band"] = out.width_median.map(C.band_of)

    # Carry per-segment QA evidence through to the results table.
    for i, row in out.iterrows():
        m = (d.s >= row.s_start) & (d.s <= row.s_end)
        sub = d[m]
        out.loc[i, "pct_kerb_normal"] = 100 * (sub.method == "kerb_normal").mean()
        out.loc[i, "median_uncertainty_m"] = float(np.nanmedian(sub.w_uncert))
        out.loc[i, "back_edge_mode"] = sub.back_class.mode().iloc[0] if len(sub) else None
        out.loc[i, "pct_negligible"] = 100 * sub.negligible.mean()
    return out
