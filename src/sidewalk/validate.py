"""Independent checks on the measurements.

None of these are self-referential: each compares our numbers against something
computed a different way, or published by someone else.

1. OS carriageway width  -- OS Highways publishes `averagewidth` / `minimumwidth`
   per RoadLink, derived by OS from their own data. If our kerb-to-kerb
   carriageway matches it, the transect machinery is sound -- and the same
   machinery produced the footway numbers.
2. Areal reconciliation  -- sum(width x length) from ray casting versus the true
   planar area of the footway polygons in that stretch, computed by polygon
   integration. Two genuinely different computations of the same quantity.
3. Sensitivity            -- how much the answer moves when the parameters move.
4. Street-level imagery   -- handled separately, in the validation notebook.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import STRtree
from shapely.ops import unary_union

from . import config as C


# --------------------------------------------------------------------------
def os_carriageway_check(xsect: pd.DataFrame, roadlinks: gpd.GeoDataFrame,
                         contexts: dict) -> pd.DataFrame:
    """Compare our measured carriageway width with OS's published attribute."""
    rows = []
    for road in contexts:
        links = roadlinks[roadlinks.name == road]
        links = links[links.averagewidth.notna()]
        if not len(links):
            continue
        xs = xsect[(xsect.road == road)].dropna(subset=["carriageway_w"])
        if not len(xs):
            continue
        pts = gpd.GeoSeries(gpd.points_from_xy(xs.x, xs.y), crs=C.CRS)
        # each cross-section takes the OS width of the nearest link of this road
        idx = [links.geometry.distance(p).idxmin() for p in pts]
        rows.append(pd.DataFrame({
            "road": road, "s": xs.s.values,
            "measured_carriageway_m": xs.carriageway_w.values,
            "median_w": xs.median_w.values,
            "os_averagewidth_m": links.loc[idx, "averagewidth"].values,
            "os_minimumwidth_m": links.loc[idx, "minimumwidth"].values,
            "formofway": links.loc[idx, "formofway"].values,
            "link_toid": links.loc[idx, "TOID"].values,
            "link_length_m": links.loc[idx, "length"].values,
        }))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    # OS's width is the trafficked carriageway; ours is kerb to kerb, so on a
    # dual carriageway ours legitimately includes the median. Compare like with
    # like by removing it.
    out["measured_excl_median_m"] = out.measured_carriageway_m - out.median_w.fillna(0)
    out["residual_m"] = out.measured_excl_median_m - out.os_averagewidth_m
    return out


def os_carriageway_by_link(check: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to the link before comparing.

    OS publishes ONE average width per RoadLink. Comparing it against individual
    cross-sections mixes our measurement error with the genuine variation along
    the link, which is unfair to both. Aggregating our cross-sections to the same
    link is the like-for-like test.
    """
    g = (check.dropna(subset=["residual_m"])
              .groupby(["road", "link_toid", "formofway"], as_index=False)
              .agg(n_xs=("residual_m", "size"),
                   ours_median_m=("measured_excl_median_m", "median"),
                   ours_iqr_m=("measured_excl_median_m", lambda x: x.quantile(.75) - x.quantile(.25)),
                   os_averagewidth_m=("os_averagewidth_m", "first"),
                   os_minimumwidth_m=("os_minimumwidth_m", "first"),
                   link_length_m=("link_length_m", "first")))
    g["residual_m"] = g.ours_median_m - g.os_averagewidth_m
    return g


def areal_reconciliation(samples: pd.DataFrame, segments: pd.DataFrame,
                         contexts: dict, topo: gpd.GeoDataFrame,
                         spacing: float = C.SAMPLE_SPACING) -> pd.DataFrame:
    """Sweep the 1-D measurements into 2-D and check they land on real pavement.

    Every cross-section gives two points: the front of the footway and the back
    of it. Joining them along a segment produces a ribbon -- the strip of ground
    we are claiming is pavement. We then intersect that ribbon with the actual
    OS footway polygons.

        coverage = area(ribbon and footway polygons) / area(ribbon)

    This is a genuinely independent test. The ribbon comes from ray casting along
    inferred kerb normals; the polygons come from the survey. If a transect ran
    oblique, cut a corner, or escaped into a forecourt, its ribbon drifts off the
    polygons and coverage falls. A segment that measures a real pavement
    correctly scores close to 1.
    """
    from shapely.geometry import Polygon
    foot = topo[topo.surface.isin(["roadside", "path"])]
    ftree = STRtree(list(foot.geometry.values))
    fgeoms = list(foot.geometry.values)
    rows = []
    for _, r in segments.iterrows():
        sub = samples[(samples.road == r.road) & (samples.side == r.side) &
                      (samples.s >= r.s_start) & (samples.s <= r.s_end) &
                      samples.valid].sort_values("s")
        sub = sub[np.isfinite(sub.fs_x) & np.isfinite(sub.back_x)]
        if len(sub) < 3:
            continue
        front = list(zip(sub.fs_x, sub.fs_y, strict=False))
        back = list(zip(sub.back_x, sub.back_y, strict=False))
        try:
            ribbon = Polygon(front + back[::-1]).buffer(0)
        except Exception:
            continue
        if ribbon.is_empty or ribbon.area < 1e-3:
            continue
        idx = ftree.query(ribbon, predicate="intersects")
        if len(idx) == 0:
            inter_area = 0.0
        else:
            inter_area = unary_union([fgeoms[int(i)].intersection(ribbon) for i in idx]).area
        rows.append(dict(segment_id=r.segment_id, road=r.road, side=r.side,
                         length_m=r.length_m, width_median=r.width_median,
                         ribbon_area_m2=ribbon.area,
                         on_footway_area_m2=inter_area,
                         # clamp: a ribbon lying wholly on footway polygons can
                         # come back as 1.0000000000000002 from floating point
                         coverage=float(min(1.0, max(0.0, inter_area / ribbon.area))),
                         implied_area_m2=float(sub.width_m.sum() * spacing)))
    out = pd.DataFrame(rows)
    if len(out):
        out["ribbon_vs_implied_pct"] = 100 * (out.implied_area_m2 - out.ribbon_area_m2) / out.ribbon_area_m2
    return out


def sensitivity(run_road_fn, road: str, topo, roadlinks, furniture, osm_ways,
                grid: dict = None) -> pd.DataFrame:
    """Re-measure one road under different parameter settings."""
    grid = grid or {
        "baseline": {},
        "spacing 0.5 m": dict(spacing=0.5),
        "spacing 2 m": dict(spacing=2.0),
        "fan +/-15 deg": dict(fan_half=15.0),
        "fan +/-45 deg": dict(fan_half=45.0),
        "kerb fit radius 2 m": dict(kerb_radius=2.0),
        "kerb fit radius 6 m": dict(kerb_radius=6.0),
        "kerb tolerance 0.05 m": dict(corner_tol=0.05),
        "kerb tolerance 0.20 m": dict(corner_tol=0.20),
        "tangent window 4 m": dict(tangent_window=4.0),
        "tangent window 16 m": dict(tangent_window=16.0),
        "no ray fan": dict(use_fan=False),
    }
    rows = []
    for label, kw in grid.items():
        spacing = kw.pop("spacing", C.SAMPLE_SPACING)
        df, xs, segs, summ, ctx = run_road_fn(
            road, topo, roadlinks, furniture, osm_ways, spacing=spacing, **kw)
        v = df[df.valid]
        rows.append(dict(setting=label, spacing_m=spacing,
                         pct_valid=100 * df.valid.mean(),
                         n_segments=len(segs),
                         median_width_m=float(np.nanmedian(v.width_m)),
                         p90_width_m=float(np.nanpercentile(v.width_m, 90)),
                         median_carriageway_m=float(np.nanmedian(xs.carriageway_w)),
                         pct_good_or_better=100 * float(
                             (v.width_m >= 2.0).mean())))
    out = pd.DataFrame(rows)
    base = out[out.setting == "baseline"].iloc[0]
    out["median_width_delta_m"] = out.median_width_m - base.median_width_m
    out["pct_band_delta"] = out.pct_good_or_better - base.pct_good_or_better
    return out
