"""Low-level geometry: ray casting, surface profiles, and local kerb direction.

Everything in here works in British National Grid metres, so a distance in the
code is a distance on the ground.
"""
from __future__ import annotations

import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import substring

EPS_MERGE = 0.05   # m: two runs of the same class closer than this are one run
                   # (OS polygon coverage occasionally leaves sub-centimetre slivers)


# --------------------------------------------------------------------------
# Vector helpers
# --------------------------------------------------------------------------
def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.hypot(v[0], v[1])
    return v / n if n > 0 else np.array([0.0, 0.0])


def perp(v) -> np.ndarray:
    """Rotate 90 degrees anticlockwise (gives the 'left' normal of a tangent)."""
    return np.array([-v[1], v[0]])


def rotate(v, deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


# --------------------------------------------------------------------------
# Surface profile along a ray
# --------------------------------------------------------------------------
class SurfaceIndex:
    """Spatial index over a classified polygon tessellation.

    OS MasterMap Topography tiles the ground with no gaps, so a ray fired across
    a street reads out an ordered stack of surfaces. This class turns a ray into
    that stack.
    """

    def __init__(self, gdf, class_col: str = "surface"):
        self.geoms = list(gdf.geometry.values)
        self.classes = list(gdf[class_col].values)
        self.tree = STRtree(self.geoms)

    def profile_raw(self, origin, direction, maxlen: float):
        """Un-merged intervals, one per source polygon, with the polygon index.

        OS `Roadside` covers everything between the kerb and the property line:
        the footway, but also parking laybys, loading bays, verges and forecourt
        paving. Where OS has drawn those as separate polygons the merged profile
        hides the fact. This exposes it, so a 4.6 m "footway" that is really a
        2.1 m footway plus a 2.5 m parking layby can be flagged.
        """
        o = np.asarray(origin, dtype=float)
        d = unit(direction)
        ray = LineString([o, o + d * maxlen])
        out = []
        for i in self.tree.query(ray, predicate="intersects"):
            inter = ray.intersection(self.geoms[i])
            if inter.is_empty:
                continue
            parts = [inter] if inter.geom_type == "LineString" else list(getattr(inter, "geoms", []))
            for p in parts:
                if p.geom_type != "LineString" or p.length <= 1e-9:
                    continue
                co = np.asarray(p.coords, dtype=float)
                t = (co - o) @ d
                out.append((float(t.min()), float(t.max()), self.classes[i], int(i)))
        return sorted(out, key=lambda r: r[0])

    def profile(self, origin, direction, maxlen: float) -> list[tuple[float, float, str]]:
        """Ordered [(t_start, t_end, surface_class), ...] along the ray.

        t is distance in metres from `origin` along unit vector `direction`.
        Consecutive runs of the same class are merged.
        """
        o = np.asarray(origin, dtype=float)
        d = unit(direction)
        ray = LineString([o, o + d * maxlen])

        idx = self.tree.query(ray, predicate="intersects")
        intervals: list[tuple[float, float, str]] = []
        for i in idx:
            inter = ray.intersection(self.geoms[i])
            if inter.is_empty:
                continue
            parts = [inter] if inter.geom_type == "LineString" else list(getattr(inter, "geoms", []))
            for p in parts:
                if p.geom_type != "LineString" or p.length <= 1e-9:
                    continue
                co = np.asarray(p.coords, dtype=float)
                t = (co - o) @ d
                intervals.append((float(t.min()), float(t.max()), self.classes[i]))

        if not intervals:
            return []
        intervals.sort(key=lambda x: x[0])

        merged: list[list] = []
        for t0, t1, cls in intervals:
            if merged and merged[-1][2] == cls and t0 - merged[-1][1] <= EPS_MERGE:
                merged[-1][1] = max(merged[-1][1], t1)
            else:
                merged.append([t0, t1, cls])
        return [(a, b, c) for a, b, c in merged]


def run_from_zero(profile, classes: set[str], tol: float = 0.05) -> float:
    """Length of the contiguous run of `classes` that starts at t=0."""
    end = 0.0
    started = False
    for t0, t1, cls in profile:
        if cls not in classes:
            if started or t0 > tol:
                break
            continue
        if not started:
            if t0 > tol:
                return 0.0
            started = True
        elif t0 - end > EPS_MERGE:
            break
        end = max(end, t1)
    return end if started else 0.0


def run_after(profile, start_t: float, classes: set[str], tol: float = 0.10):
    """Contiguous run of `classes` beginning at (or just after) `start_t`.

    Returns (length, end_t, terminating_class). The terminating class is what the
    ray hits immediately after the run -- the back edge of the footway.
    """
    end = None
    for t0, t1, cls in profile:
        if t1 <= start_t + 1e-9:
            continue
        if cls in classes:
            if end is None:
                if t0 > start_t + tol:
                    return 0.0, start_t, _class_at(profile, start_t + 1e-3)
                end = max(t1, start_t)
            elif t0 - end > EPS_MERGE:
                return end - start_t, end, cls
            else:
                end = max(end, t1)
        else:
            if end is None:
                return 0.0, start_t, cls
            return end - start_t, end, cls
    if end is None:
        return 0.0, start_t, "void"
    return end - start_t, end, "void"   # ran off the end of the data


def _class_at(profile, t: float) -> str:
    for t0, t1, cls in profile:
        if t0 <= t <= t1:
            return cls
    return "void"


# --------------------------------------------------------------------------
# Local kerb direction
# --------------------------------------------------------------------------
def _components(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    if geom.geom_type in ("MultiLineString", "GeometryCollection"):
        return [g for g in geom.geoms if g.geom_type == "LineString"]
    return []


def local_kerb_frame(kerb_pt, corridor_boundary_parts, outward_hint,
                     radius: float = 4.0, n_samples: int = 41,
                     corner_tol: float = 0.06):
    """Infer the true kerb tangent/normal at a point on the carriageway edge.

    The transect direction that matters is perpendicular to the *kerb*, not to
    the road centreline. On a bend or a widening those differ, and a
    centreline-normal transect overstates the footway by 1/cos(theta).

    Method
    ------
    Take the stretch of carriageway boundary within `radius` metres *along the
    kerb* of the hit point. Put it in a local frame, then fit a weighted
    quadratic b = c0 + c1*a + c2*a^2. A quadratic (not a straight line) matters:
    a smoothly curving kerb is perfectly well defined and should NOT be rejected,
    whereas a genuine discontinuity -- a junction radius, a vehicle crossover,
    a build-out -- cannot be fitted and should be. The residual to the quadratic
    therefore separates *curvature* (fine) from *corners* (not fine).

    Returns (normal, tangent, rms_residual, ok). `ok` is False when the kerb is
    broken or cornered, in which case the caller falls back to the ray fan.
    """
    P = Point(kerb_pt)
    best, best_d = None, np.inf
    for part in corridor_boundary_parts:
        d = part.distance(P)
        if d < best_d:
            best, best_d = part, d
    if best is None or best_d > 0.5:
        return None, None, np.inf, False

    s = best.project(P)
    sub = substring(best, max(0.0, s - radius), min(best.length, s + radius))
    if sub.is_empty or sub.length < 1.5:
        return None, None, np.inf, False

    ts = np.linspace(0, sub.length, n_samples)
    pts = np.array([[p.x, p.y] for p in (sub.interpolate(t) for t in ts)])
    K = np.asarray(kerb_pt, dtype=float)

    # Initial frame from principal axis of the local boundary.
    centre = pts.mean(axis=0)
    X = pts - centre
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    u = unit(Vt[0])           # provisional tangent
    v = perp(u)               # provisional normal

    a = (pts - K) @ u
    b = (pts - K) @ v
    w = np.exp(-((a / radius) ** 2))          # weight the kerb near the hit point

    A = np.column_stack([np.ones_like(a), a, a * a])
    W = np.sqrt(w)
    try:
        coef, *_ = np.linalg.lstsq(A * W[:, None], b * W, rcond=None)
    except np.linalg.LinAlgError:
        return None, None, np.inf, False
    resid = float(np.sqrt(np.average((b - A @ coef) ** 2, weights=w)))

    # Tangent at the hit point (a = 0) is (1, db/da) = (1, c1) in the local frame.
    c1 = coef[1]
    tangent = unit(u + c1 * v)
    normal = perp(tangent)
    if np.dot(normal, outward_hint) < 0:
        normal = -normal
        tangent = -tangent

    ok = resid <= corner_tol
    return normal, tangent, resid, ok
