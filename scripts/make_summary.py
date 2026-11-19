"""Road-level cross-section summary: the brief's 'total street width' table."""
from __future__ import annotations

import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

TAB = pathlib.Path("outputs/tables")


def main():
    xs = pd.read_csv(TAB / "cross_sections.csv")
    # The brief wants the three components on the SAME cross-section, so the total
    # is computed only where both sides measured. Reported on its own, though, a
    # per-side median restricted that way is biased: on Brushfield Street the wide
    # north footway is measurable along stretches where the south side is not, so
    # requiring both inflates the left-hand figure. Each side therefore also gets
    # a median over its own valid cross-sections.
    both = xs[xs.left_valid & xs.right_valid].copy()
    side_med = {}
    for side in ("left", "right"):
        ok = xs[xs[f"{side}_valid"]]
        side_med[side] = ok.groupby("road")[f"{side}_w"].median()
    d = both

    def row(g):
        has_median = g.median_w > 0.05
        return pd.Series({
            "n_cross_sections": len(g),
            "left_footway_m": round(g.left_w.median(), 2),
            "left_footway_own_m": round(float(side_med["left"].get(g.name, np.nan)), 2),
            "carriageway_m": round(g.carriageway_w.median(), 2),
            "right_footway_m": round(g.right_w.median(), 2),
            "right_footway_own_m": round(float(side_med["right"].get(g.name, np.nan)), 2),
            "total_street_width_m": round(g.total_w.median(), 2),
            "total_p10_m": round(np.percentile(g.total_w, 10), 2),
            "total_p90_m": round(np.percentile(g.total_w, 90), 2),
            # A median strip exists on part of some streets only, so reporting the
            # median of a column that is mostly zero would read as "no median".
            "pct_with_median_strip": round(100 * has_median.mean(), 1),
            "median_strip_m_where_present": round(g.loc[has_median, "median_w"].median(), 2)
            if has_median.any() else np.nan,
            "frontage_to_frontage_m": round(g.frontage_w.median(), 2)
            if g.frontage_w.notna().any() else np.nan,
            "pct_closure_good": round(100 * (g.closure_class == "GOOD").mean(), 1),
        })

    t = d.groupby("road").apply(row).reset_index(drop=True) if False else \
        d.groupby("road").apply(row).reset_index()
    t.to_csv(TAB / "road_cross_section.csv", index=False)
    print(t.to_string(index=False))
    print("\nleft + carriageway + right = total by construction. Frontage-to-frontage is")
    print("measured independently to the building line; the gap is the closure residual.")
    return t


if __name__ == "__main__":
    main()
