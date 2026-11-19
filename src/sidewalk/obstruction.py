"""Effective walkable width -- an OPTIONAL accessibility derivative.

IMPORTANT SCOPE NOTE, repeated wherever this number appears:

    Effective walkable width is an optional accessibility-oriented derivative.
    It is reported ALONGSIDE the geometric footway width the assessment asks for
    and is never substituted for it. All banding, all segment classification and
    all headline results use the geometric width.

What it does: OS surveys street furniture as Topographic Points -- 332 posts and
bollards and 264 positioned street trees inside our study area. A 2.4 m footway
with a lamp column in the middle of it does not offer 2.4 m of walking room. For
each cross-section we take the footway interval [kerb, back] and remove the part
of it swept by any surveyed obstruction, then report the largest surviving gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from shapely import STRtree
from shapely.geometry import LineString

from . import config as C


def effective_width(df: pd.DataFrame, obstructions, spacing: float = C.SAMPLE_SPACING,
                    radii: dict = None) -> pd.DataFrame:
    """Add `effective_w` and `obstruction_count` to the transect table."""
    radii = radii or C.OBSTRUCTION_BUFFER
    d = df.copy()
    d["effective_w"] = d["width_m"]
    d["obstruction_count"] = 0
    d["obstruction_types"] = None
    if obstructions is None or len(obstructions) == 0:
        return d

    geoms = list(obstructions.geometry.values)
    kinds = list(obstructions.obstruction.values)
    tree = STRtree(geoms)
    lateral = spacing / 2.0

    for i, row in d.iterrows():
        if not row.valid or not np.isfinite(row.get("width_m", np.nan)):
            continue
        # Start at the FRONT OF THE PAVEMENT, not at the kerb. Where a cycle
        # track or a planted strip sits inside the kerb line the two are metres
        # apart, and sweeping from the kerb both misses obstructions at the back
        # of the footway and blames the footway for ones in the cycle track.
        FS = np.array([row.fs_x, row.fs_y])
        if not np.isfinite(FS).all():
            FS = np.array([row.kerb_x, row.kerb_y])
        n = np.array([row.nkx, row.nky])
        w = float(row.width_m)
        if w <= 0:
            continue
        line = LineString([FS, FS + n * w])
        cand = tree.query(line.buffer(lateral + 1.0), predicate="intersects")
        blocked, types = [], []
        for j in np.atleast_1d(cand):
            g = geoms[int(j)]
            r = radii.get(kinds[int(j)], C.DEFAULT_OBSTRUCTION_RADIUS)
            p = np.array([g.x, g.y])
            along = float((p - FS) @ n)
            across = float(np.linalg.norm((p - FS) - along * n))
            if across > lateral + r or along < -r or along > w + r:
                continue
            blocked.append((max(0.0, along - r), min(w, along + r)))
            types.append(kinds[int(j)])
        if not blocked:
            continue
        # Largest free run of the footway once obstructions are removed.
        blocked.sort()
        free, cursor = [], 0.0
        for a, b in blocked:
            if a > cursor:
                free.append(a - cursor)
            cursor = max(cursor, b)
        if cursor < w:
            free.append(w - cursor)
        d.at[i, "effective_w"] = float(max(free)) if free else 0.0
        d.at[i, "obstruction_count"] = len(blocked)
        d.at[i, "obstruction_types"] = ",".join(sorted(set(types)))
    return d
