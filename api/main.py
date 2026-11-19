"""Read-only API over the measured results.

Deliberately small. The pipeline is a batch job; this serves what it produced so
a dashboard, a GIS client or a colleague can query segments without re-running
anything. No measurement happens here.
"""
from __future__ import annotations

import json
import pathlib
import sys
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from sidewalk import config as C  # noqa: E402

TAB = C.OUTPUTS / "tables"
GEO = C.OUTPUTS / "geo"

app = FastAPI(
    title="Sidewalk Width API",
    version="1.0.0",
    description=("Pavement widths measured from OS MasterMap for Tower Hamlets and "
                 "Spitalfields. Widths are geometric footway widths; effective walkable "
                 "width is an optional accessibility derivative and is never substituted "
                 "for it."),
)


@lru_cache(maxsize=1)
def _segments() -> pd.DataFrame:
    p = TAB / "segments.csv"
    if not p.exists():
        raise HTTPException(503, "results not built - run scripts/run_all.py")
    return pd.read_csv(p)


@lru_cache(maxsize=1)
def _meta() -> dict:
    p = TAB / "run_metadata.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _clean(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.replace({float("nan"): None}).to_json(orient="records"))


@app.get("/health")
def health():
    ok = (TAB / "segments.csv").exists()
    return {"status": "ok" if ok else "results missing", "version": app.version}


@app.get("/meta")
def meta():
    """Pipeline parameters and provenance for the current results."""
    return _meta()


@app.get("/roads")
def roads():
    df = _segments()
    out = (df.groupby("road")
             .apply(lambda g: pd.Series({
                 "segments": len(g),
                 "measured_length_m": round(g.length_m.sum(), 1),
                 "median_width_m": round(g.width_median.median(), 2),
                 "median_sigma_m": round(g.sigma_m.median(), 2),
                 "pct_length_below_1_5m": round(
                     100 * g.loc[g.width_median < 1.5, "length_m"].sum() / g.length_m.sum(), 1),
                 "median_confidence": round(g.confidence.median(), 2),
             }), include_groups=False)
             .reset_index())
    return _clean(out)


@app.get("/roads/{road}/segments")
def road_segments(road: str,
                  side: str | None = Query(None, pattern="^(left|right)$"),
                  min_confidence: float = 0.0):
    df = _segments()
    sub = df[df.road.str.lower() == road.lower()]
    if sub.empty:
        raise HTTPException(404, f"road not measured: {road}")
    if side:
        sub = sub[sub.side == side]
    sub = sub[sub.confidence.fillna(0) >= min_confidence]
    return _clean(sub.sort_values(["side", "s_start"]))


@app.get("/segments/{segment_id}")
def segment(segment_id: str):
    df = _segments()
    row = df[df.segment_id == segment_id]
    if row.empty:
        raise HTTPException(404, f"no such segment: {segment_id}")
    return _clean(row)[0]


@app.get("/segments")
def segments(band: str | None = None, min_width: float | None = None,
             max_artefact_risk: float | None = None, limit: int = 200):
    df = _segments()
    if band:
        df = df[df.band == band]
    if min_width is not None:
        df = df[df.width_median >= min_width]
    if max_artefact_risk is not None:
        df = df[df.artefact_risk <= max_artefact_risk]
    return _clean(df.nlargest(min(limit, 1000), "width_median"))


@app.get("/validation")
def validation():
    p = TAB / "validation_summary.json"
    if not p.exists():
        raise HTTPException(503, "validation not run - run scripts/run_validation.py")
    return json.loads(p.read_text())


@app.get("/geojson")
def geojson(road: str | None = None):
    p = GEO / "segments_wgs84.geojson"
    if not p.exists():
        raise HTTPException(503, "geometry not built")
    data = json.loads(p.read_text())
    if road:
        data["features"] = [f for f in data["features"]
                            if f["properties"].get("road", "").lower() == road.lower()]
    return JSONResponse(data, media_type="application/geo+json")
