"""End-to-end orchestration: one named street in, measured segments out."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from . import confidence as CF
from . import config as C
from . import dataio, qa
from . import segment as S
from . import transect as T


def prepare_layers(osm_ways=None):
    """Load once, classify once.

    Two reclassifications happen here and both matter:
      * traffic islands and central reservations are split out of the footway
        class geometrically (they are tagged `Roadside` exactly like a pavement);
      * segregated cycle tracks are split out of the `Path` class using OSM,
        because OS files them alongside alleys and precincts.
    """
    topo = dataio.load_topo()
    assert str(topo.crs).upper().endswith("27700"), f"unexpected CRS {topo.crs}"
    fw, isl = dataio.split_islands(topo)
    topo = topo.copy()
    topo.loc[isl.index, "surface"] = "island"
    n_cycle = 0
    if osm_ways is not None:
        topo, n_cycle = dataio.tag_cycletracks(topo, osm_ways)
    roadlinks = dataio.load_roadlinks()
    assert str(roadlinks.crs).upper().endswith("27700"), f"unexpected CRS {roadlinks.crs}"
    obstructions = dataio.load_obstructions()
    return topo, roadlinks, obstructions, isl, n_cycle


def run_road(name: str, topo, roadlinks, obstructions=None, osm=None,
             spacing: float = C.SAMPLE_SPACING, use_fan: bool = True,
             **measure_kw):
    """Measure one street. Returns (samples, cross_sections, segments, qa_summary, ctx)."""
    ctx = T.build_context(name, roadlinks, topo, obstructions)

    df = T.measure_road(ctx, spacing=spacing, use_fan=use_fan, **measure_kw)
    df = qa.neighbour_check(df, spacing=spacing)
    df = T.ensemble_stats(df, spacing=spacing)

    # Cross-section level: carriageway, median, total, closure.
    xs = T.pair_sides(df)
    for side in ("left", "right"):
        g = df[df.side == side].set_index("s")
        xs[f"{side}_gap_class"] = xs.s.map(g.gap_class)
        xs[f"{side}_back"] = xs.s.map(g.back_class)
    xs = qa.classify_closure(xs)

    # Feed closure back to the samples so it can enter the confidence model.
    cmap = xs.set_index("s").closure_class
    df["closure_class"] = df.s.map(cmap).fillna("N/A")

    if osm is not None:
        from . import osm as OSM
        df = OSM.sidewalk_evidence(osm, name, df)

    if obstructions is not None:
        from . import obstruction as OB
        df = OB.effective_width(df, ctx.obstructions, spacing=spacing)

    df = CF.sample_uncertainty(df)
    df = CF.sample_confidence(df)

    segs = pd.concat([_segment_side(df[df.side == sd], spacing) for sd in ("left", "right")],
                     ignore_index=True) if df.valid.any() else pd.DataFrame()

    summary = qa.road_qa_summary(df, xs)
    summary["axis_length_m"] = ctx.axis.length
    summary.update({f"axis_{k}": v for k, v in ctx.diag.items()
                    if k in ("n_links", "n_components_raw", "n_bridged")})
    return df, xs, segs, summary, ctx


def _segment_side(d_side: pd.DataFrame, spacing: float) -> pd.DataFrame:
    """Segment one side, then attach evidence, uncertainty and band-with-CI."""
    d = d_side[d_side.valid].sort_values("s")
    if len(d) < 3:
        return pd.DataFrame()
    segs = S.segment_series(d.s.to_numpy(), d.width_m.to_numpy(), spacing=spacing)
    if not segs:
        return pd.DataFrame()
    out = pd.DataFrame(segs)
    out["road"] = d_side.road.iloc[0]
    out["side"] = d_side.side.iloc[0]

    rows = []
    for _, row in out.iterrows():
        m = (d.s >= row.s_start) & (d.s <= row.s_end)
        sub = d[m]
        sigma, n_eff = CF.segment_uncertainty(sub, spacing=spacing)
        bu = CF.band_with_uncertainty(row.width_median, sigma)
        rows.append(dict(
            sigma_m=sigma, n_effective=n_eff,
            confidence=float(np.nanmean(sub.confidence)),
            pct_kerb_normal=100 * (sub.method == "kerb_normal").mean(),
            back_edge_mode=sub.back_class.mode().iloc[0] if len(sub) else None,
            pct_negligible=100 * sub.negligible.mean(),
            pct_closure_ambiguous=100 * (sub.closure_class == "AMBIGUOUS").mean(),
            mean_dist_other_road=float(np.nanmean(sub.dist_other_road)),
            median_sigma_method=float(np.nanmedian(sub.sigma_method)),
            # Obstructions are sparse point features, so the median effective
            # width barely moves. What matters for accessibility is the pinch
            # point, so we carry the 10th percentile and the minimum too.
            effective_width_median=(float(np.nanmedian(sub.effective_w))
                                    if "effective_w" in sub else np.nan),
            effective_width_p10=(float(np.nanpercentile(sub.effective_w.dropna(), 10))
                                 if "effective_w" in sub and sub.effective_w.notna().any() else np.nan),
            min_clear_width=(float(np.nanmin(sub.effective_w))
                             if "effective_w" in sub and sub.effective_w.notna().any() else np.nan),
            # The narrowest point of a segment and the narrowest point CAUSED BY
            # AN OBSTRUCTION are different quantities. min_clear_width is the
            # former and moves with ordinary width variation; this is the latter,
            # and is the only one that says anything about street furniture.
            max_obstruction_loss_m=(float(np.nanmax(sub.width_m - sub.effective_w))
                                    if "effective_w" in sub and sub.effective_w.notna().any() else np.nan),
            pct_obstructed=(100 * (sub.obstruction_count > 0).mean()
                            if "obstruction_count" in sub else np.nan),
            osm_agree=(float((sub.osm_says_footway == "yes").mean())
                       if "osm_says_footway" in sub else np.nan),
            x_mid=float(sub.x.mean()), y_mid=float(sub.y.mean()),
            **bu))
    out = pd.concat([out, pd.DataFrame(rows, index=out.index)], axis=1)
    out["short_run"] = out.get("short_run", False).fillna(False) if "short_run" in out else False
    out["segment_id"] = [f"{out.road.iloc[0]}|{out.side.iloc[0]}|{int(a)}-{int(b)}"
                         for a, b in zip(out.s_start, out.s_end, strict=False)]
    return out


def attach_geometry(segs: pd.DataFrame, contexts: dict, samples: pd.DataFrame) -> gpd.GeoDataFrame:
    """Give every segment a line geometry running down the middle of its footway.

    The segment is a stretch of the street axis offset sideways by
    (half-carriageway + half the footway width), so it plots where the pavement
    actually is rather than on the centreline.
    """
    from shapely.ops import substring
    geoms = []
    for _, r in segs.iterrows():
        ctx = contexts[r.road]
        base = substring(ctx.axis, float(r.s_start), float(r.s_end))
        sub = samples[(samples.road == r.road) & (samples.side == r.side) &
                      (samples.s >= r.s_start) & (samples.s <= r.s_end) & samples.valid]
        half = float(np.nanmedian(sub.half_corridor)) if len(sub) else 0.0
        # Include anything between the kerb and the pavement (a cycle track, a
        # planted strip) or the line is drawn on top of it rather than on the
        # footway it measures.
        strip = float(np.nanmedian(sub.kerbside_strip_w.fillna(0.0))) if len(sub) else 0.0
        off = half + strip + float(r.width_median) / 2.0
        try:
            g = base.offset_curve(off if r.side == "left" else -off)
            if g.is_empty:
                g = base
        except Exception:
            g = base
        geoms.append(g)
    return gpd.GeoDataFrame(segs.copy(), geometry=geoms, crs=C.CRS)
