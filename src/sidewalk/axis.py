"""Building a single measurement axis for a named street.

OS Highways stores a street as many short RoadLinks. A naive `linemerge` breaks
into fragments at every junction, and on a dual carriageway you get two parallel
lines for one street. We need one continuous, sensibly-oriented line to walk
along and to define chainage against.

Approach: build the street's link graph, take the path between the two most
distant nodes (the graph diameter path). That is the through-route. On a dual
carriageway the diameter path follows one carriageway, which is what we want --
the transect engine reads across the median from there and reports it separately.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from shapely.geometry import LineString

SNAP = 0.05  # m: endpoints closer than this are the same node


def _key(xy):
    return (round(xy[0] / SNAP) * SNAP, round(xy[1] / SNAP) * SNAP)


def build_axis(links_gdf, smooth_window: float = 7.0, resample: float = 1.0,
               bridge_gap: float = 40.0):
    """Return (axis LineString, diagnostics dict) for one street's RoadLinks.

    Two practical complications are handled here.

    1. A street is often split into several disconnected pieces because the links
       inside a crossroads are attributed to the *other* street. We bridge pieces
       whose ends are within `bridge_gap` metres with a straight connector. Those
       connectors sit inside junctions, where the QA rules discard transects
       anyway, so they add chainage continuity without adding measurements.
    2. On a dual carriageway the diameter path follows one carriageway. That is
       deliberate: the transect engine then reads across the median from there
       and reports the median separately from the footways.
    """
    G = nx.Graph()
    for i, row in links_gdf.iterrows():
        g = row.geometry
        if g is None or g.is_empty:
            continue
        parts = [g] if g.geom_type == "LineString" else list(g.geoms)
        # Prefer OS's own node identifiers; fall back to snapped coordinates.
        a0 = getattr(row, "startnode_href", None)
        b0 = getattr(row, "endnode_href", None)
        for p in parts:
            a = a0 if (a0 and len(parts) == 1) else _key(p.coords[0])
            b = b0 if (b0 and len(parts) == 1) else _key(p.coords[-1])
            if a == b:
                continue
            if G.has_edge(a, b) and G[a][b]["weight"] <= p.length:
                continue
            G.add_edge(a, b, weight=p.length, geom=p, idx=i, synthetic=False)

    if G.number_of_edges() == 0:
        raise ValueError("no usable links")

    # Give every node a position so we can bridge components.
    pos = {}
    for u, v, d in G.edges(data=True):
        c = list(d["geom"].coords)
        pos.setdefault(u, c[0])
        pos.setdefault(v, c[-1])
        if np.hypot(*(np.array(pos[u]) - np.array(c[0]))) > np.hypot(*(np.array(pos[u]) - np.array(c[-1]))):
            pos[u] = c[-1]
            pos[v] = c[0]

    n_bridged = 0
    while True:
        comps = list(nx.connected_components(G))
        if len(comps) < 2:
            break
        best = None
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                for u in comps[i]:
                    for v in comps[j]:
                        d = float(np.hypot(*(np.array(pos[u]) - np.array(pos[v]))))
                        if best is None or d < best[0]:
                            best = (d, u, v)
        if best is None or best[0] > bridge_gap:
            break
        d, u, v = best
        G.add_edge(u, v, weight=d, geom=LineString([pos[u], pos[v]]), idx=-1, synthetic=True)
        n_bridged += 1

    comp = max(nx.connected_components(G), key=lambda c: sum(
        G[u][v]["weight"] for u, v in G.subgraph(c).edges()))
    H = G.subgraph(comp).copy()

    # Diameter path: the longest of all shortest paths. This is the through-route.
    endpoints = [n for n in H.nodes if H.degree(n) == 1] or list(H.nodes)
    best, best_len = None, -1.0
    lengths = dict(nx.all_pairs_dijkstra_path_length(H, weight="weight"))
    for u in endpoints:
        for v, L in lengths.get(u, {}).items():
            if L > best_len:
                best, best_len = (u, v), L
    path = nx.shortest_path(H, best[0], best[1], weight="weight")

    coords: list[tuple[float, float]] = []
    used, synth_ranges = [], []
    run = 0.0
    for u, v in zip(path[:-1], path[1:], strict=False):
        p = H[u][v]["geom"]
        used.append(H[u][v]["idx"])
        c = list(p.coords)
        if np.hypot(*(np.array(c[0]) - np.array(pos[u]))) > np.hypot(*(np.array(c[-1]) - np.array(pos[u]))):
            c = c[::-1]
        if H[u][v]["synthetic"]:
            synth_ranges.append((run, run + p.length))
        run += p.length
        coords.extend(c if not coords else c[1:])

    raw = LineString(coords)
    axis = _resample_smooth(raw, resample, smooth_window)

    total_len = sum(d["weight"] for _, _, d in G.edges(data=True) if not d["synthetic"])
    diag = dict(
        n_links=len(links_gdf),
        axis_length=axis.length,
        network_length=total_len,
        n_components_raw=len(list(nx.connected_components(nx.Graph(
            [(u, v) for u, v, d in G.edges(data=True) if not d["synthetic"]])))),
        n_bridged=n_bridged,
        synthetic_ranges=synth_ranges,
        used_link_idx=sorted(set(i for i in used if i >= 0)),
    )
    return axis, diag


def _resample_smooth(line: LineString, step: float, window: float) -> LineString:
    """Resample at fixed spacing then lightly smooth, keeping the endpoints fixed.

    Smoothing is deliberately gentle: it removes digitising noise that would make
    the local tangent jitter, without dragging the axis off the carriageway.
    """
    n = max(int(np.ceil(line.length / step)) + 1, 2)
    ts = np.linspace(0, line.length, n)
    pts = np.array([[p.x, p.y] for p in (line.interpolate(t) for t in ts)])
    k = max(int(round(window / step)) | 1, 3)  # odd window
    if len(pts) > k + 2:
        kern = np.ones(k) / k
        pad = k // 2
        padded = np.vstack([np.repeat(pts[:1], pad, axis=0), pts,
                            np.repeat(pts[-1:], pad, axis=0)])
        sm = np.column_stack([np.convolve(padded[:, i], kern, mode="valid") for i in (0, 1)])
        sm[0], sm[-1] = pts[0], pts[-1]
        pts = sm
    return LineString(pts)


def tangent_at(line: LineString, s: float, window: float) -> np.ndarray:
    """Smoothed unit tangent at chainage `s`, from a central difference."""
    L = line.length
    a = line.interpolate(max(0.0, s - window))
    b = line.interpolate(min(L, s + window))
    v = np.array([b.x - a.x, b.y - a.y])
    n = np.hypot(*v)
    if n < 1e-9:
        a = line.interpolate(max(0.0, s - 1.0))
        b = line.interpolate(min(L, s + 1.0))
        v = np.array([b.x - a.x, b.y - a.y])
        n = np.hypot(*v)
    return v / n if n > 0 else np.array([1.0, 0.0])


def orient_axis(axis: LineString) -> LineString:
    """Orient consistently (south->north, else west->east) so 'left'/'right' are stable."""
    (x0, y0), (x1, y1) = axis.coords[0], axis.coords[-1]
    dx, dy = x1 - x0, y1 - y0
    flip = (abs(dy) >= abs(dx) and dy < 0) or (abs(dx) > abs(dy) and dx < 0)
    return LineString(list(axis.coords)[::-1]) if flip else axis
