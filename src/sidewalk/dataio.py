"""Loading and classifying the source data.

The whole method rests on one observation: OS MasterMap Topography is a *seamless
polygon tessellation* of the ground surface. Every square metre of the study area
belongs to exactly one TopographicArea. So if we walk a straight line across a
street we can read off, in order, exactly what surfaces we cross -- carriageway,
kerb, footway, then whatever bounds the footway at the back.

This module turns the raw MasterMap tables into that tessellation, tagged with a
small vocabulary of surface classes we can reason about.
"""
from __future__ import annotations

import functools
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

from . import config as C

# --------------------------------------------------------------------------
# Name handling
# --------------------------------------------------------------------------
_ARRAY_RE = re.compile(r'"([^"]*)"|([^,]+)')


def clean_name(raw) -> str | None:
    """OS Highways stores names as PostgreSQL-style arrays: '{"Brick Lane"}'."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip()
    if not s or s in ("{}", "None", "nan"):
        return None
    s = s.strip("{}")
    vals = [a or b for a, b in _ARRAY_RE.findall(s)]
    vals = [v.strip().strip('"') for v in vals if v and v.strip()]
    return vals[0] if vals else None


def clean_names_all(raw) -> list[str]:
    """All names on a link (a link can carry alternates)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    s = str(raw).strip().strip("{}")
    vals = [a or b for a, b in _ARRAY_RE.findall(s)]
    return [v.strip().strip('"') for v in vals if v and v.strip()]


# --------------------------------------------------------------------------
# Surface classification
# --------------------------------------------------------------------------
# A TopographicArea's descriptivegroup can be a comma-separated *set*
# (e.g. "Roadside,Structure" = a footway carried on a bridge deck).
def _groups(s) -> set[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return set()
    return {g.strip() for g in str(s).split(",") if g.strip()}


def _terms(s) -> set[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return set()
    return {t.strip() for t in str(s).split(",") if t.strip()}


def classify_surface(row) -> str:
    """Map an OS TopographicArea onto our surface vocabulary.

    Classes
    -------
    carriageway : trafficked surface ("Road Or Track"), incl. traffic calming
    roadside    : OS "Roadside" -- the raw candidate footway class. This bucket
                  still contains traffic islands and central reservations; those
                  are separated out geometrically in `split_islands`.
    verge       : "Roadside" but naturally surfaced -- grass/planting, not walkable
    steps       : stepped surfaces, excluded from level footway width
    path        : off-carriageway pedestrian areas (plazas, alleys, park paths)
    building / general / structure / rail / water / other
    """
    g = _groups(row.descriptivegroup)
    t = _terms(row.descriptiveterm)
    make = row.make

    if "Step" in t:
        return "steps"
    if "Road Or Track" in g:
        return "carriageway"
    if "Roadside" in g:
        if make == "Natural":
            return "verge"
        return "roadside"
    if "Path" in g:
        return "path"
    if "Building" in g or "Glasshouse" in g:
        return "building"
    if "Rail" in g:
        return "rail"
    if "Inland Water" in g or "Tidal Water" in g:
        return "water"
    if "General Surface" in g:
        return "general"
    if "Structure" in g:
        return "structure"
    if "Natural Environment" in g or "Landform" in g:
        return "natural"
    return "other"


# Surfaces that can legitimately terminate the back of a footway.
# If a transect's footway run ends on anything NOT in this set it has leaked.
# `path` is included deliberately. Where a footway opens onto a pedestrian area
# -- Altab Ali Park off Whitechapel High Street, the Gunthorpe Street precinct --
# the Roadside polygon genuinely ends against pedestrian space. That is a real
# back edge, not a leak: the pavement measurement up to it stands, and the sample
# is flagged `opens_onto_pedestrian_area` so the wider walking space is visible.
# `carriageway`, `island` and `roadside` remain invalid: those mean the ray has
# escaped across a junction or through a sliver in the polygon coverage.
VALID_BACK_EDGE = {
    "building", "general", "structure", "rail", "water",
    "natural", "verge", "steps", "other", "void", "path",
}
# `carriageway`, `island`, `cycletrack` and `roadside` are deliberately absent:
# seeing one of those behind the pavement means the ray never reached the
# property line and has escaped across a junction or through a polygon sliver.


@functools.lru_cache(maxsize=4)
def load_topo(bbox: tuple | None = None) -> gpd.GeoDataFrame:
    """Load OS MasterMap TopographicArea and attach our surface class."""
    ta = gpd.read_file(C.TOPO_GPKG, layer="Topographicarea", bbox=bbox)
    ta = ta[ta.geometry.notna() & ~ta.geometry.is_empty].copy()
    ta["geometry"] = ta.geometry.buffer(0)           # repair any self-touching rings
    ta["surface"] = ta.apply(classify_surface, axis=1)
    ta["area_m2"] = ta.geometry.area
    return ta.reset_index(drop=True)


@functools.lru_cache(maxsize=4)
def load_roadlinks() -> gpd.GeoDataFrame:
    """OS MasterMap Highways RoadLink: centrelines + OS's own width attributes."""
    rl = gpd.read_file(C.HWY_GPKG, layer="RoadLink")
    rl["name"] = rl.roadname.map(clean_name)
    rl["alt_names"] = rl.roadname.map(clean_names_all)
    for col in ("averagewidth", "minimumwidth", "length"):
        rl[col] = pd.to_numeric(rl[col], errors="coerce")
    return rl


@functools.lru_cache(maxsize=4)
def load_obstructions(bbox: tuple | None = None) -> gpd.GeoDataFrame:
    """Surveyed street furniture: posts/bollards and positioned street trees.

    Used ONLY for the optional 'effective walkable width' derivative.
    """
    tp = gpd.read_file(C.TOPO_GPKG, layer="Topographicpoint", bbox=bbox)
    tp = tp[tp.geometry.notna()].copy()

    def kind(row):
        g, t = _groups(row.descriptivegroup), _terms(row.descriptiveterm)
        if "Post" in t and ("Roadside" in g or "Structure" in g):
            return "Post"
        if "Positioned Nonconiferous Tree" in t or "Positioned Coniferous Tree" in t:
            return "Tree"
        if "Roadside" in g and "Structure" in t:
            return "Structure"
        return None

    tp["obstruction"] = tp.apply(kind, axis=1)
    return tp[tp.obstruction.notna()].reset_index(drop=True)


@functools.lru_cache(maxsize=2)
def load_lsoa() -> gpd.GeoDataFrame:
    return gpd.read_file(C.LSOA_SHP).to_crs(C.CRS)


# --------------------------------------------------------------------------
# Traffic islands / central reservations
# --------------------------------------------------------------------------
def split_islands(topo: gpd.GeoDataFrame, share_threshold: float = 0.85):
    """Separate true footway from traffic islands inside the 'roadside' class.

    A central reservation or pedestrian refuge is tagged by OS exactly like a
    pavement ("Roadside", Manmade). Geometrically though it is an island: almost
    its entire boundary is shared with the carriageway. Real footway always has a
    back edge against buildings, walls or private land.

    Returns (footway_gdf, island_gdf).
    """
    carr = topo[topo.surface == "carriageway"]
    if len(carr) == 0:
        rs = topo[topo.surface == "roadside"]
        return rs, rs.iloc[0:0]
    carr_u = unary_union(carr.geometry.values)
    carr_boundary = carr_u.boundary

    rs = topo[topo.surface == "roadside"].copy()
    if len(rs) == 0:
        return rs, rs

    # Fraction of each roadside polygon's perimeter that lies on the carriageway edge.
    shared = rs.geometry.boundary.intersection(carr_boundary.buffer(0.05)).length
    rs["carr_share"] = (shared / rs.geometry.boundary.length).clip(0, 1)
    is_island = rs.carr_share >= share_threshold
    return rs[~is_island].copy(), rs[is_island].copy()


# --------------------------------------------------------------------------
def tag_cycletracks(topo: gpd.GeoDataFrame, osm_ways, buffer_m: float = 1.5):
    """Separate segregated cycle tracks out of the OS `Path` class using OSM.

    This is the clearest case in the project where neither source can do the job
    alone. OS MasterMap gives us an accurate polygon for the strip between the
    kerb and the pavement on Whitechapel High Street, but files it under `Path`
    -- the same class it uses for alleys, park paths and pedestrian precincts.
    OSM knows that strip is Cycleway CS2 but carries no usable width for it.

    Crossing the two: an OS `Path` polygon that an OSM `highway=cycleway` runs
    through is a cycle track, and must not be counted as pavement. Nine of the
    fifteen Path polygons beside Whitechapel High Street fail that test.

    Returns (topo_with_cycletrack_class, n_reclassified).
    """
    if osm_ways is None or len(osm_ways) == 0 or "highway" not in osm_ways.columns:
        return topo, 0
    cyc = osm_ways[osm_ways.highway == "cycleway"]
    if len(cyc) == 0:
        return topo, 0
    band = unary_union(cyc.geometry.values).buffer(buffer_m)
    out = topo.copy()
    mask = (out.surface == "path") & out.geometry.intersects(band)
    out.loc[mask, "surface"] = "cycletrack"
    return out, int(mask.sum())
