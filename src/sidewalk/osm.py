"""OpenStreetMap as a corroboration layer -- explicitly NOT as ground truth.

OSM does not reliably carry pavement widths in Tower Hamlets, so it cannot
measure anything for us. What it can do is answer independent yes/no questions
that a surveyed polygon dataset cannot answer on its own:

    Does anyone else think there is a footway on this side of this street?
    Has anyone mapped it as a separate way, and does it sit where we say it does?
    Where a width tag does exist, does it agree with our number?

Answers feed the `source` term of the confidence model and nothing else. A
disagreement lowers confidence and gets reported; it never overrides the
measurement.
"""
from __future__ import annotations

import time

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point

from . import config as C

CACHE = C.DATA / "cache"
OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]

_to_wgs = Transformer.from_crs(C.CRS, "EPSG:4326", always_xy=True)
_to_bng = Transformer.from_crs("EPSG:4326", C.CRS, always_xy=True)


def bng_to_wgs(x, y):
    lon, lat = _to_wgs.transform(x, y)
    return lon, lat


def fetch_osm(bbox_bng, refresh: bool = False) -> gpd.GeoDataFrame:
    """Highway ways in the bbox, cached on disk so the notebook is reproducible."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = "_".join(f"{v:.0f}" for v in bbox_bng)
    path = CACHE / f"osm_{key}.geojson"
    if path.exists() and not refresh:
        return gpd.read_file(path).to_crs(C.CRS)

    import requests
    minx, miny, maxx, maxy = bbox_bng
    lon0, lat0 = bng_to_wgs(minx, miny)
    lon1, lat1 = bng_to_wgs(maxx, maxy)
    q = f"""
    [out:json][timeout:180];
    ( way["highway"]({lat0:.6f},{lon0:.6f},{lat1:.6f},{lon1:.6f}); );
    out geom tags;
    """
    data = None
    for url in OVERPASS:
        try:
            r = requests.post(url, data={"data": q},
                              headers={"User-Agent": "sidewalk-width-assessment/1.0"},
                              timeout=180)
            if r.status_code == 200:
                data = r.json()
                break
        except Exception:
            time.sleep(2)
    if data is None:
        raise RuntimeError("Overpass unavailable - OSM corroboration layer skipped")

    rows = []
    for el in data.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        pts = [_to_bng.transform(p["lon"], p["lat"]) for p in el["geometry"]]
        if len(pts) < 2:
            continue
        tags = el.get("tags", {})
        rows.append({"osm_id": el["id"], "geometry": LineString(pts),
                     **{k: v for k, v in tags.items()
                        if k in ("highway", "name", "footway", "sidewalk",
                                 "sidewalk:left", "sidewalk:right", "sidewalk:both",
                                 "width", "sidewalk:left:width", "sidewalk:right:width",
                                 "surface", "foot")}})
    gdf = gpd.GeoDataFrame(rows, crs=C.CRS)
    gdf.to_file(path, driver="GeoJSON")
    return gdf


# --------------------------------------------------------------------------
def _num(v):
    try:
        return float(str(v).split()[0])
    except Exception:
        return np.nan


def sidewalk_evidence(osm: gpd.GeoDataFrame, road_name: str, samples: pd.DataFrame,
                      search: float = 10.0) -> pd.DataFrame:
    """Attach per-sample OSM evidence to the transect table.

    Side-aware: a separately mapped sidewalk only counts as evidence for *this*
    side if it actually lies on this side of the centreline. Streets here are
    around 10 m wide, so a naive nearest-way test would happily corroborate the
    left footway with the right footway's OSM geometry.
    """
    from shapely import STRtree

    s = samples.copy()
    s["osm_score"] = 0.80          # neutral: OSM is silent
    s["osm_says_footway"] = None
    s["osm_width_m"] = np.nan
    if osm is None or len(osm) == 0:
        return s

    name_col = osm["name"].astype(str).str.lower() if "name" in osm.columns else pd.Series("", index=osm.index)
    named = osm[name_col == road_name.lower()]

    seps = osm[osm.get("highway", pd.Series(dtype=str)) == "footway"]
    if "footway" in seps.columns:
        seps = seps[seps.footway.isna() | (seps.footway == "sidewalk")]
    sep_geoms = list(seps.geometry.values)
    sep_tree = STRtree(sep_geoms) if sep_geoms else None
    sep_width = ([_num(v) for v in seps["width"]] if "width" in seps.columns
                 else [np.nan] * len(seps))

    tag_for = {"left": ["sidewalk:left", "sidewalk:both", "sidewalk"],
               "right": ["sidewalk:right", "sidewalk:both", "sidewalk"]}
    wtag_for = {"left": "sidewalk:left:width", "right": "sidewalk:right:width"}
    named_geoms = list(named.geometry.values)
    named_tree = STRtree(named_geoms) if named_geoms else None

    for i, row in s.iterrows():
        if not np.isfinite(row.get("back_x", np.nan)):
            continue
        mid = Point((row.kerb_x + row.back_x) / 2, (row.kerb_y + row.back_y) / 2)
        n = np.array([row.n0x, row.n0y])
        axis_pt = np.array([row.x, row.y])
        says = None

        if named_tree is not None:
            j = named.index[int(np.atleast_1d(named_tree.query_nearest(Point(row.x, row.y)))[0])]
            for t in tag_for[row.side]:
                if t in named.columns and pd.notna(named.at[j, t]):
                    v = str(named.at[j, t]).lower()
                    if v in ("yes", "both", "left", "right"):
                        says = "yes"
                    elif v in ("no", "none"):
                        says = "no"
                    elif v == "separate":
                        says = None          # mapped as its own way; test for it below
                    break
            wt = wtag_for[row.side]
            if wt in named.columns and pd.notna(named.at[j, wt]):
                s.at[i, "osm_width_m"] = _num(named.at[j, wt])

        if sep_tree is not None:
            for k in np.atleast_1d(sep_tree.query(mid.buffer(search), predicate="intersects")):
                g = sep_geoms[int(k)]
                cp = g.interpolate(g.project(mid))
                if np.dot(np.array([cp.x, cp.y]) - axis_pt, n) > 0:   # same side
                    says = "yes"
                    if np.isfinite(sep_width[int(k)]) and not np.isfinite(s.at[i, "osm_width_m"]):
                        s.at[i, "osm_width_m"] = sep_width[int(k)]
                    break

        s.at[i, "osm_says_footway"] = says
        measured = row.get("width_m", np.nan)
        has = np.isfinite(measured) and measured > C.NEGLIGIBLE_FOOTWAY
        if says == "yes":
            s.at[i, "osm_score"] = 1.00 if has else 0.45
        elif says == "no":
            s.at[i, "osm_score"] = 0.45 if has else 1.00
    return s


# --------------------------------------------------------------------------
FURNITURE_QUERY = """
[out:json][timeout:180];
(
  node["barrier"="bollard"]({bb});
  node["natural"="tree"]({bb});
  node["highway"="street_lamp"]({bb});
  node["amenity"~"^(bench|waste_basket|bicycle_parking|telephone|post_box|drinking_fountain)$"]({bb});
  node["man_made"~"^(street_cabinet|utility_pole|flagpole|monitoring_station)$"]({bb});
  node["emergency"="fire_hydrant"]({bb});
  node["advertising"]({bb});
  node["highway"="traffic_signals"]({bb});
);
out body;
"""

FURNITURE_RADIUS = {
    "tree": 0.75, "bollard": 0.20, "street_lamp": 0.25, "bench": 0.60,
    "waste_basket": 0.30, "bicycle_parking": 0.60, "telephone": 0.45,
    "post_box": 0.35, "street_cabinet": 0.45, "utility_pole": 0.25,
    "traffic_signals": 0.25, "advertising": 0.40, "fire_hydrant": 0.20,
    "drinking_fountain": 0.35, "flagpole": 0.20, "monitoring_station": 0.30,
}


def fetch_osm_furniture(bbox_bng, refresh: bool = False) -> gpd.GeoDataFrame:
    """Street furniture from OSM.

    OS MasterMap Topography carries only a sparse selection of street furniture
    (about one surveyed point per 10,000 square metres across this extent, and
    only 29 points along our ten streets). OSM contributors have mapped it far
    more densely in Spitalfields and Whitechapel. We therefore use OSM as the
    primary furniture source and OS as corroboration -- and we state plainly that
    neither is a complete inventory, so effective walkable width is an upper
    bound on obstruction, i.e. a lower bound on the true clear width.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    key = "_".join(f"{v:.0f}" for v in bbox_bng)
    path = CACHE / f"osm_furniture_{key}.geojson"
    if path.exists() and not refresh:
        return gpd.read_file(path).to_crs(C.CRS)

    import requests
    minx, miny, maxx, maxy = bbox_bng
    lon0, lat0 = bng_to_wgs(minx, miny)
    lon1, lat1 = bng_to_wgs(maxx, maxy)
    bb = f"{lat0:.6f},{lon0:.6f},{lat1:.6f},{lon1:.6f}"
    q = FURNITURE_QUERY.format(bb=bb)

    data = None
    for url in OVERPASS:
        try:
            r = requests.post(url, data={"data": q},
                              headers={"User-Agent": "sidewalk-width-assessment/1.0"},
                              timeout=180)
            if r.status_code == 200:
                data = r.json()
                break
        except Exception:
            time.sleep(2)
    if data is None:
        raise RuntimeError("Overpass unavailable - OSM furniture layer skipped")

    rows = []
    for el in data.get("elements", []):
        if el.get("type") != "node":
            continue
        t = el.get("tags", {})
        kind = (t.get("barrier") or t.get("natural") or t.get("amenity")
                or t.get("man_made") or t.get("emergency")
                or ("advertising" if "advertising" in t else None)
                or (t.get("highway") if t.get("highway") in ("street_lamp", "traffic_signals") else None))
        if not kind:
            continue
        x, y = _to_bng.transform(el["lon"], el["lat"])
        rows.append({"osm_id": el["id"], "obstruction": kind,
                     "geometry": Point(x, y)})
    gdf = gpd.GeoDataFrame(rows, crs=C.CRS)
    if len(gdf):
        gdf.to_file(path, driver="GeoJSON")
    return gdf


def combine_furniture(os_points: gpd.GeoDataFrame, osm_points: gpd.GeoDataFrame,
                      dedupe_m: float = 1.5) -> gpd.GeoDataFrame:
    """Merge the OS and OSM furniture inventories, dropping near-duplicates."""
    from shapely import STRtree
    frames = []
    if os_points is not None and len(os_points):
        a = os_points[["obstruction", "geometry"]].copy()
        a["source"] = "OS MasterMap"
        frames.append(a)
    if osm_points is not None and len(osm_points):
        b = osm_points[["obstruction", "geometry"]].copy()
        b["source"] = "OSM"
        frames.append(b)
    if not frames:
        return gpd.GeoDataFrame(columns=["obstruction", "source", "geometry"], crs=C.CRS)
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)

    a, b = frames
    tree = STRtree(list(a.geometry.values))
    keep = []
    for g in b.geometry.values:
        hit = tree.query(g.buffer(dedupe_m), predicate="intersects")
        keep.append(len(hit) == 0)
    out = gpd.GeoDataFrame(pd.concat([a, b[np.array(keep)]], ignore_index=True), crs=C.CRS)
    return out
