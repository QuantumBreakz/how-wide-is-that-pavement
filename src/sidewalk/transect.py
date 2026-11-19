"""The measurement engine.

For every 2 m of street we build one cross-section and read the street off it:

    axis point -> carriageway -> KERB -> footway -> back edge -> (building frontage)

The transect direction is not simply perpendicular to the road centreline. Once
the ray finds the kerb we re-derive the direction from the *local carriageway
boundary*, because it is the kerb -- not the centreline -- that defines a
cross-section of the footway. The centreline-normal and a +/-30 deg ray fan are
retained as a fallback where the kerb is too curved or broken to fit (junction
radii, crossovers), and the spread of the fan is kept as an instability metric.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from . import config as C
from . import dataio
from . import geom as G
from .axis import build_axis, orient_axis, tangent_at

# Everything between the two pavements. A segregated cycle track sits inside the
# kerb line but is not walkable, so it belongs here, not in the footway.
CORRIDOR = {"carriageway", "island", "cycletrack"}
VEHICLE = CORRIDOR                      # kept for backwards compatibility
FOOT = {"roadside"}
FOOT_WITH_PATH = {"roadside", "path"}
# A strip that can legitimately sit between the kerb and the pavement.
KERBSIDE_STRIP = {"path", "verge"}
MAX_KERBSIDE_STRIP = 6.0                # m
MAX_BRIDGED_INTERRUPTION = 2.5          # m: a thin path crossing the footway
KERB_BACKOFF = 0.30                     # m the kerb-normal ray starts inside the road


@dataclass
class RoadContext:
    """Everything needed to measure one named street."""
    name: str
    axis: LineString
    diag: dict
    topo: gpd.GeoDataFrame
    index: G.SurfaceIndex
    corridor_boundary: list
    vehicle_union: object
    other_roads: STRtree
    other_geoms: list
    obstructions: gpd.GeoDataFrame = field(default=None)


def build_context(name: str, roadlinks, topo, obstructions=None, pad: float = 90.0) -> RoadContext:
    sel = roadlinks[roadlinks.name == name]
    if sel.empty:
        raise ValueError(f"road not found: {name}")
    axis, diag = build_axis(sel)
    axis = orient_axis(axis)

    minx, miny, maxx, maxy = sel.total_bounds
    win = (minx - pad, miny - pad, maxx + pad, maxy + pad)
    local = topo.cx[win[0]:win[2], win[1]:win[3]].copy()

    veh = local[local.surface.isin(VEHICLE)]
    veh_u = unary_union(veh.geometry.values) if len(veh) else None
    boundary_parts = []
    if veh_u is not None:
        b = veh_u.boundary
        boundary_parts = [b] if b.geom_type == "LineString" else list(getattr(b, "geoms", []))

    others = roadlinks[(roadlinks.name != name) & roadlinks.geometry.intersects(
        LineString(axis.coords).buffer(pad))]
    other_geoms = list(others.geometry.values)
    tree = STRtree(other_geoms) if other_geoms else None

    obs = None
    if obstructions is not None and len(obstructions):
        obs = obstructions.cx[win[0]:win[2], win[1]:win[3]].copy()

    return RoadContext(name, axis, diag, local, G.SurfaceIndex(local), boundary_parts,
                       veh_u, tree, other_geoms, obs)


# --------------------------------------------------------------------------
# Surfaces that can genuinely close the back of a footway. Anything else found
# behind the pavement means the ray is still inside the highway and has not yet
# reached the property line.
# Surfaces that genuinely close the back of a street.
TRUE_BACK = {"building", "general", "structure", "rail", "water", "natural",
             "other", "steps", "verge", "path"}
# `verge` and `path` are in that set, but conditionally: either can also sit
# BETWEEN the kerb and the pavement (a planted strip, a pedestrian threshold).
# Treating the first one encountered as the back of the street would report a
# 0 m footway with a perfectly good pavement behind it. So a verge or path is
# stepped over when, and only when, a `roadside` run follows it before anything
# that unambiguously closes the street. Otherwise it stops the ray -- which is
# what we want next to Altab Ali Park, where the path IS the back edge and the
# park is not a pavement.
SOFT_BACK = {"verge", "path"}
HARD_BACK = TRUE_BACK - SOFT_BACK


def _find_back_edge(prof, n):
    """Index of the interval that closes the back of the street (or None).

    A `verge` or `path` is ambiguous: it can be a strip between the kerb and the
    pavement, the pedestrian surface itself, or the thing the pavement opens onto.
    Which one it is depends on what else the cross-section contains, so all three
    are resolved explicitly:

      * a pavement appears behind it  -> step over it (kerb buffer, planted strip)
      * a pavement was already passed -> it closes the street (footway opens onto
                                         a park or a precinct)
      * no pavement anywhere          -> step over it and look for a hard edge;
                                         it is itself the walking surface
                                         (a pedestrianised street)
    """
    k, seen_roadside = 1, False
    while k < n:
        cls = prof[k][2]
        if cls in HARD_BACK:
            return k
        if cls == "roadside":
            seen_roadside = True
            k += 1
            continue
        if cls in SOFT_BACK:
            j = k + 1
            roadside_behind = False
            while j < n and prof[j][2] not in HARD_BACK:
                if prof[j][2] == "roadside":
                    roadside_behind = True
                    break
                j += 1
            if roadside_behind:
                k = j                    # strip between kerb and pavement
                continue
            if seen_roadside:
                return k                 # the pavement opens onto it
            # No pavement anywhere: this may be the walking surface itself. Step
            # over it and see whether something hard closes the street behind.
            while j < n and prof[j][2] not in HARD_BACK:
                j += 1
            return j if j < n else k
        k += 1
    return None


def _read_ray(ctx, origin, direction, maxlen=C.MAX_RAY, detail=False):
    """Read one cross-section ray and parse it into named components.

    Rather than assuming a fixed stack of surfaces, we locate the back of the
    street first and then work inwards:

        1. walk out until we reach a surface that can close the street --
           a building, a forecourt, private land, a park, a wall;
        2. the pavement is the pedestrian surface immediately in front of it;
        3. everything between the road centre and the pavement is the corridor,
           decomposed into carriageway, median and cycle track.

    This ordering matters. On Whitechapel High Street the real sequence is
    carriageway, then a segregated cycle track, then the pavement -- and in
    places a narrow kerb buffer sits in front of the cycle track too. A parser
    that expects "carriageway then pavement" reports a 0.5 m footway there. This
    one reports the pavement wherever it actually sits in the cross-section.

    All distances returned are measured from the ray origin.
    """
    prof = ctx.index.profile(origin, direction, maxlen)
    if not prof:
        return None
    n = len(prof)

    # ---- the corridor must start under our feet ---------------------------
    if prof[0][2] not in CORRIDOR or prof[0][0] > 0.05:
        return None

    # ---- 1. the back of the street ----------------------------------------
    edge_i = _find_back_edge(prof, n)
    if edge_i is None:
        t_edge, back = (prof[-1][1], "void")
        edge_i = n
    else:
        t_edge, back = prof[edge_i][0], prof[edge_i][2]

    # ---- 2. the pavement: the pedestrian run that ends at that back edge ---
    w, foot_start, from_path, bridged = 0.0, t_edge, False, 0.0
    j = edge_i - 1
    if j >= 0 and prof[j][2] in ("roadside", "path"):
        cls = prof[j][2]
        from_path = cls == "path"
        foot_start = prof[j][0]
        while j - 1 >= 0:
            pc, pa, pb = prof[j - 1][2], prof[j - 1][0], prof[j - 1][1]
            if pc == cls and foot_start - pb <= G.EPS_MERGE:
                foot_start = pa
                j -= 1
                continue
            # bridge a thin pedestrian-path interruption inside the footway
            if (cls == "roadside" and pc == "path" and pb - pa <= MAX_BRIDGED_INTERRUPTION
                    and j - 2 >= 0 and prof[j - 2][2] == "roadside"):
                bridged += pb - pa
                foot_start = prof[j - 2][0]
                j -= 2
                continue
            break
        w = t_edge - foot_start

    # ---- 3. the corridor, decomposed --------------------------------------
    t_corr = foot_start
    def span(cls):
        return sum(min(b, t_corr) - max(a, 0.0)
                   for a, b, c in prof if c == cls and a < t_corr and b > 0.0)

    carr_w, median_w, cycle_w = span("carriageway"), span("island"), span("cycletrack")
    # what sits between the kerb line and the pavement, if anything
    kerb_end = 0.0
    for a, b, c in prof:
        if c in CORRIDOR and a <= kerb_end + G.EPS_MERGE:
            kerb_end = max(kerb_end, b)
        elif a > kerb_end:
            break
    strip_w = max(0.0, foot_start - kerb_end)
    strip_cls = _dominant_class(prof, kerb_end, foot_start) if strip_w > 0.05 else None
    # If the ray re-enters the road between the kerb and the pavement it has
    # crossed a junction mouth and the "pavement" it found is on the far side of
    # a side street. Likewise a strip wider than a cycle track plus its buffer is
    # not a kerbside strip, it is a leak.
    # Only re-entering the *trafficked* surface signals a leak. Meeting the cycle
    # track here is normal: on Whitechapel High Street a narrow kerb buffer sits
    # between the carriageway and the CS2 track, so the track legitimately falls
    # in this span.
    strip_reentered = any(c in {"carriageway", "island"} for a, b, c in prof
                          if a >= kerb_end - G.EPS_MERGE and b <= foot_start + G.EPS_MERGE
                          and b - a > 0.05)

    # ---- 4. the building frontage, for the closure test -------------------
    t_front, gap_cls = np.nan, None
    for a, _b, c in prof:
        if c == "building" and a >= t_edge - 1e-6:
            if a - t_edge <= C.MAX_FRONTAGE_GAP:
                t_front = a
                gap_cls = _dominant_class(prof, t_edge, a) if a > t_edge + 0.05 else "none"
            break

    # ---- 5. is the footway run one surface, or several? -------------------
    n_polys, outer_w, poly_widths = 1, w, None
    if detail and w > 0.05:
        raw = ctx.index.profile_raw(origin, direction, maxlen)
        pieces = [(a, b, c, i) for a, b, c, i in raw
                  if b > foot_start + 0.02 and a < t_edge - 0.02 and c in ("roadside", "path")]
        if pieces:
            n_polys = len({i for *_, i in pieces})
            outer_w = max(min(b, t_edge) - max(a, foot_start) for a, b, c, i in pieces
                          if b >= t_edge - 0.05) if any(b >= t_edge - 0.05 for a, b, c, i in pieces) else w
            poly_widths = ";".join(f"{min(b, t_edge) - max(a, foot_start):.2f}"
                                   for a, b, c, i in pieces)

    w_path, _, _ = G.run_after(prof, kerb_end, FOOT_WITH_PATH)
    return dict(n_footway_polys=n_polys, outer_poly_w=outer_w, poly_widths=poly_widths,
                strip_reentered=bool(strip_reentered),
                t_kerb=kerb_end, corridor_w=t_corr, carr_w=carr_w, median=median_w,
                cycletrack=cycle_w, strip_class=strip_cls, strip_w=strip_w,
                foot_start=foot_start, w=w, t_back=t_edge, back=back,
                from_path=from_path, bridged=bridged,
                w_path=w_path, t_front=t_front, gap_class=gap_cls)


def _dominant_class(prof, t0, t1):
    """Which surface occupies most of (t0, t1) -- i.e. what the residual is made of."""
    if t1 - t0 < 1e-6:
        return "none"
    acc = {}
    for a, b, cls in prof:
        ov = min(b, t1) - max(a, t0)
        if ov > 0:
            acc[cls] = acc.get(cls, 0.0) + ov
    return max(acc, key=acc.get) if acc else "void"


def _dist_to_other_road(ctx, point) -> float:
    """Distance to the nearest *other* street centreline -- a junction proxy."""
    if ctx.other_roads is None or not ctx.other_geoms:
        return np.inf
    pt = Point(point)
    idx = ctx.other_roads.query_nearest(pt)
    if len(idx) == 0:
        return np.inf
    return float(min(ctx.other_geoms[int(i)].distance(pt) for i in np.atleast_1d(idx)))


def _crosses_other_road(ctx, origin, direction, length) -> bool:
    if ctx.other_roads is None or length <= 0:
        return False
    o = np.asarray(origin, float)
    seg = LineString([o, o + G.unit(direction) * length])
    idx = ctx.other_roads.query(seg, predicate="intersects")
    return len(idx) > 0


def measure_road(ctx: RoadContext, spacing: float = C.SAMPLE_SPACING,
                 fan_half: float = C.FAN_HALF_ANGLE, fan_step: float = C.FAN_STEP,
                 kerb_radius: float = C.KERB_FIT_RADIUS,
                 corner_tol: float = C.KERB_CORNER_TOL,
                 tangent_window: float = C.TANGENT_WINDOW,
                 use_fan: bool = True) -> pd.DataFrame:
    """One row per (chainage, side).

    Three estimators are computed for every sample:
      w_kerb  -- along the normal to the locally fitted kerb   (primary)
      w_perp  -- along the normal to the road centreline       (naive baseline)
      w_fan   -- minimum over a +/-30 deg ray fan              (conservative fallback)
    The primary is w_kerb where the kerb fit is sound and w_fan where it is not.
    The spread between estimators is kept as a per-sample uncertainty.
    """
    L = ctx.axis.length
    ss = np.arange(0.0, L + 1e-9, spacing)
    angles = np.arange(-fan_half, fan_half + 1e-9, fan_step)
    rows = []

    for s in ss:
        P = ctx.axis.interpolate(float(s))
        p = np.array([P.x, P.y])
        t = tangent_at(ctx.axis, float(s), tangent_window)
        n_left = G.perp(t)
        on_carr = ctx.vehicle_union is not None and ctx.vehicle_union.contains(P)

        for side, sgn in (("left", 1.0), ("right", -1.0)):
            n0 = G.unit(n_left * sgn)
            rec = dict(road=ctx.name, s=float(s), side=side, x=P.x, y=P.y,
                       tx=t[0], ty=t[1], n0x=n0[0], n0y=n0[1],
                       axis_on_carriageway=bool(on_carr))

            base = _read_ray(ctx, p, n0, detail=True)
            if base is None or base["t_kerb"] <= 0.01:
                rows.append({**rec, "valid": False, "reason": "axis_not_on_carriageway"})
                continue
            if base["t_kerb"] > C.MAX_PLAUSIBLE_HALF_CARRIAGEWAY:
                rows.append({**rec, "valid": False, "reason": "carriageway_leak",
                             "half_corridor": base["t_kerb"]})
                continue

            K = p + n0 * base["t_kerb"]

            # ---- estimator 1: normal to the locally fitted kerb ---------------
            nk, _, resid, fit_ok = G.local_kerb_frame(
                K, ctx.corridor_boundary, n0, radius=kerb_radius, corner_tol=corner_tol)
            # The kerb-normal ray must START INSIDE the carriageway: the parser
            # requires the cross-section to begin on a trafficked surface, so a
            # ray launched from just outside the kerb is rejected and everything
            # silently degrades to the fan. Back off 0.3 m into the road, then
            # re-derive the kerb point along the new direction.
            kerb, K_nk = None, K
            if fit_ok and nk is not None:
                kerb = _read_ray(ctx, K - nk * KERB_BACKOFF, nk,
                                 maxlen=C.MAX_RAY + KERB_BACKOFF, detail=True)
                if kerb is not None and abs(kerb["t_kerb"] - KERB_BACKOFF) > 1.0:
                    kerb = None            # the kerb moved: fit was unreliable
                elif kerb is not None:
                    K_nk = K + nk * (kerb["t_kerb"] - KERB_BACKOFF)
            nk_used = G.unit(nk) if kerb is not None else n0

            # ---- estimator 3: conservative ray fan ---------------------------
            fan_w = []
            if use_fan:
                for a in angles:
                    d = G.rotate(n0, a)
                    r = _read_ray(ctx, p, d)
                    if r and 0 < r["t_kerb"] <= C.MAX_PLAUSIBLE_HALF_CARRIAGEWAY and r["w"] > 0:
                        fan_w.append(r["w"])
            w_fan = float(np.min(fan_w)) if fan_w else np.nan
            fan_spread = float(np.max(fan_w) - np.min(fan_w)) if len(fan_w) > 1 else np.nan

            # NOTE ON ORIGINS. The kerb-normal ray starts at the kerb K, so its
            # distances are already measured from the kerb. The centreline ray
            # starts at the axis point P, so its distances must have the
            # half-carriageway subtracted before they can be compared. Getting
            # this wrong silently inflates every closure residual.
            w_perp = base["w"]
            if kerb is not None:
                method, w, src = "kerb_normal", kerb["w"], kerb
            elif np.isfinite(w_fan):
                method, w, src = "fan_fallback", w_fan, base
            else:
                method, w, src = "centreline_normal", w_perp, base
            # Every ray reports distances from its own origin; re-base them all
            # onto the kerb so the components are directly comparable.
            t_front = src["t_front"] - src["t_kerb"]
            back, gap_cls, w_path = src["back"], src["gap_class"], src["w_path"]
            strip_cls, strip_w = src["strip_class"], src["strip_w"]
            from_path, bridged = src["from_path"], src["bridged"]

            ests = [e for e in (kerb["w"] if kerb else np.nan, w_perp, w_fan) if np.isfinite(e)]
            w_uncert = float(np.max(ests) - np.min(ests)) if len(ests) > 1 else np.nan

            reach = base["t_kerb"] + strip_w + (w if np.isfinite(w) else 0.0) + 0.5
            junction = _crosses_other_road(ctx, p, n0, reach)
            d_junction = _dist_to_other_road(ctx, p)

            FS = K_nk + nk_used * strip_w      # front of the pavement
            B = FS + nk_used * (w if np.isfinite(w) else 0.0)
            F = K_nk + nk_used * t_front if np.isfinite(t_front) else np.array([np.nan, np.nan])

            reason = None
            if src.get("strip_reentered"):
                reason = "ray_reentered_road"
            elif src.get("cycletrack", 0.0) > C.MAX_CYCLETRACK:
                reason = "implausible_cycletrack"
            elif strip_w - src.get("cycletrack", 0.0) > MAX_KERBSIDE_STRIP:
                reason = "footway_beyond_plausible_strip"
            elif junction:
                reason = "junction_mouth"
            elif not np.isfinite(w):
                reason = "no_measurement"
            elif w > C.MAX_PLAUSIBLE_FOOTWAY:
                reason = "implausible_width"
            elif back not in dataio.VALID_BACK_EDGE:
                reason = f"bad_back_edge:{back}"

            rows.append({**rec,
                         "half_corridor": base["t_kerb"],
                         "median_side": base["median"],
                         "carriageway_side": base["carr_w"],
                         "cycletrack_w": src["cycletrack"],
                         "kerbside_strip_class": strip_cls,
                         "kerbside_strip_w": strip_w,
                         "footway_from_path": bool(from_path),
                         "n_footway_polys": src.get("n_footway_polys", 1),
                         "outer_poly_w": src.get("outer_poly_w", np.nan),
                         "poly_widths": src.get("poly_widths"),
                         "bridged_path_m": bridged,
                         "kerb_x": K_nk[0], "kerb_y": K_nk[1],
                         "fs_x": FS[0], "fs_y": FS[1],
                         "nkx": nk_used[0], "nky": nk_used[1],
                         "kerb_resid": resid, "kerb_fit_ok": bool(fit_ok),
                         "method": method,
                         "foot_w": w,
                         "w_kerb": kerb["w"] if kerb else np.nan,
                         "w_perp": w_perp, "w_fan": w_fan,
                         "w_uncert": w_uncert, "fan_spread": fan_spread,
                         "foot_w_with_path": w_path,
                         "path_extension": (w_path - w) if np.isfinite(w_path) and np.isfinite(w) else np.nan,
                         "back_class": back, "gap_class": gap_cls,
                         "back_x": B[0], "back_y": B[1],
                         "frontage_t": t_front, "front_x": F[0], "front_y": F[1],
                         "dist_other_road": d_junction,
                         "opens_onto_pedestrian_area": bool(back == "path"),
                         "separated_from_kerb": bool(strip_w > 0.2),
                         "negligible": bool(np.isfinite(w) and w < C.NEGLIGIBLE_FOOTWAY),
                         "valid": reason is None, "reason": reason})

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def pair_sides(df: pd.DataFrame) -> pd.DataFrame:
    """Combine the two sides of each cross-section into one street record.

    Carriageway width is measured kerb-to-kerb along the bisector of the two kerb
    normals, so a street that curves does not inflate its own carriageway.
    """
    out = []
    for s, grp in df.groupby("s"):
        L = grp[grp.side == "left"]
        R = grp[grp.side == "right"]
        if len(L) != 1 or len(R) != 1:
            continue
        left_row, r = L.iloc[0], R.iloc[0]
        rec = dict(road=left_row.road, s=s, x=left_row.x, y=left_row.y)

        have_kerbs = np.isfinite([left_row.get("kerb_x", np.nan), r.get("kerb_x", np.nan)]).all()
        if have_kerbs:
            nL = np.array([left_row.nkx, left_row.nky])
            nR = np.array([r.nkx, r.nky])
            nbar = G.unit(nL - nR)
            KL = np.array([left_row.kerb_x, left_row.kerb_y])
            KR = np.array([r.kerb_x, r.kerb_y])
            rec["carriageway_w"] = float(abs((KL - KR) @ nbar))
            rec["median_w"] = float((left_row.get("median_side") or 0) + (r.get("median_side") or 0))
            rec["normal_disagree_deg"] = float(np.degrees(
                np.arccos(np.clip(-(nL @ nR), -1, 1))))
            FL = np.array([left_row.front_x, left_row.front_y])
            FR = np.array([r.front_x, r.front_y])
            rec["frontage_w"] = float(abs((FL - FR) @ nbar)) if np.isfinite([FL, FR]).all() else np.nan
        else:
            rec.update(carriageway_w=np.nan, median_w=np.nan,
                       normal_disagree_deg=np.nan, frontage_w=np.nan)

        for tag, row in (("left", left_row), ("right", r)):
            rec[f"{tag}_w"] = row.get("foot_w", np.nan)
            rec[f"{tag}_valid"] = bool(row.get("valid", False))
            rec[f"{tag}_reason"] = row.get("reason")
            rec[f"{tag}_method"] = row.get("method")
            rec[f"{tag}_back"] = row.get("back_class")
            rec[f"{tag}_fan_spread"] = row.get("fan_spread", np.nan)
            rec[f"{tag}_w_perp"] = row.get("foot_w_perp", np.nan)

        rec["total_w"] = rec["left_w"] + rec["carriageway_w"] + rec["right_w"] \
            if (rec["left_valid"] and rec["right_valid"]) else np.nan
        rec["closure_residual"] = rec["frontage_w"] - rec["total_w"] \
            if np.isfinite(rec.get("frontage_w", np.nan)) else np.nan
        out.append(rec)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
def ensemble_stats(df: pd.DataFrame, half_window: float = C.ENSEMBLE_HALF_WINDOW,
                   spacing: float = C.SAMPLE_SPACING) -> pd.DataFrame:
    """Replace each single-ray width with the robust centre of nearby transects.

    One intersection is one intersection: a vehicle crossover polygon, a sliver,
    a gate opening or a parked-car bay boundary can all move it. So the reported
    width at chainage s is the median of every valid transect within +/- 2 m of s,
    each measured on its own kerb normal, and the MAD of that little population is
    carried forward as one component of the uncertainty budget.
    """
    df = df.copy()
    k = max(int(round(half_window / spacing)), 1)
    win = 2 * k + 1
    for col in ("ens_median", "ens_mad", "ens_n", "ens_iqr"):
        df[col] = np.nan

    for _key, grp in df.groupby(["road", "side"], sort=False):
        g = grp.sort_values("s")
        w = g.foot_w.where(g.valid).astype(float)
        r = w.rolling(win, center=True, min_periods=2)
        med = r.median()
        df.loc[g.index, "ens_median"] = med
        df.loc[g.index, "ens_n"] = r.count()
        df.loc[g.index, "ens_iqr"] = r.quantile(0.75) - r.quantile(0.25)
        df.loc[g.index, "ens_mad"] = (w - med).abs().rolling(
            win, center=True, min_periods=2).median()

    # The reported width. Falls back to the single ray where the window is empty.
    df["width_m"] = df.ens_median.where(df.ens_median.notna(), df.foot_w)
    return df
