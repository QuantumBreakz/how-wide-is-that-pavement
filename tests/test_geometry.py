"""Unit tests for the geometry primitives.

These use synthetic streets with known answers, so a failure means the maths is
wrong rather than the data being awkward.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from sidewalk import confidence as CF
from sidewalk import config as C
from sidewalk import geom as G
from sidewalk import segment as S


# --------------------------------------------------------------------------
def _straight_street(carr_half=3.0, foot=2.0, length=100.0):
    """A perfectly straight street with known widths, as a classified tessellation."""
    rows = []
    rows.append(dict(surface="carriageway",
                     geometry=Polygon([(0, -carr_half), (length, -carr_half),
                                       (length, carr_half), (0, carr_half)])))
    for sgn in (1, -1):
        a, b = sgn * carr_half, sgn * (carr_half + foot)
        rows.append(dict(surface="roadside",
                         geometry=Polygon([(0, a), (length, a), (length, b), (0, b)])))
        c = sgn * (carr_half + foot + 10)
        rows.append(dict(surface="building",
                         geometry=Polygon([(0, b), (length, b), (length, c), (0, c)])))
    return gpd.GeoDataFrame(rows, crs=C.CRS)


def test_unit_and_perp():
    assert np.allclose(G.unit([3, 4]), [0.6, 0.8])
    assert np.allclose(G.perp([1, 0]), [0, 1])
    assert np.allclose(G.rotate([1, 0], 90), [0, 1], atol=1e-9)


def test_profile_reads_the_stack_in_order():
    idx = G.SurfaceIndex(_straight_street())
    prof = idx.profile((50.0, 0.0), (0, 1), 40)
    classes = [c for _, _, c in prof]
    assert classes[:3] == ["carriageway", "roadside", "building"]
    assert G.run_from_zero(prof, {"carriageway"}) == pytest.approx(3.0, abs=1e-6)


def test_run_after_measures_the_footway_and_names_the_back_edge():
    idx = G.SurfaceIndex(_straight_street(carr_half=3.0, foot=2.0))
    prof = idx.profile((50.0, 0.0), (0, 1), 40)
    w, t_back, back = G.run_after(prof, 3.0, {"roadside"})
    assert w == pytest.approx(2.0, abs=1e-6)
    assert back == "building"


def test_kerb_frame_recovers_the_normal_of_a_straight_kerb():
    street = _straight_street()
    carr = street[street.surface == "carriageway"].geometry.iloc[0]
    parts = [carr.boundary]
    n, t, resid, ok = G.local_kerb_frame((50.0, 3.0), parts, np.array([0.0, 1.0]))
    assert ok and resid < 1e-6
    assert np.allclose(n, [0, 1], atol=1e-6)


def test_kerb_frame_follows_a_curve_instead_of_rejecting_it():
    """A gently curving kerb is well defined; the quadratic fit must accept it."""
    th = np.linspace(-0.6, 0.6, 200)
    R = 60.0
    kerb = LineString(np.column_stack([R * np.sin(th), R * np.cos(th) - R]))
    n, t, resid, ok = G.local_kerb_frame((0.0, 0.0), [kerb], np.array([0.0, -1.0]))
    assert ok, f"a {R:.0f} m radius kerb should still be fittable (resid {resid:.4f})"
    assert np.allclose(np.abs(n), [0, 1], atol=1e-3)


def test_kerb_frame_rejects_a_corner():
    corner = LineString([(-5, 0), (0, 0), (0, 5)])
    n, t, resid, ok = G.local_kerb_frame((0.0, 0.0), [corner], np.array([0.0, -1.0]))
    assert not ok, "a right-angled junction radius must fall back to the fan"


# --------------------------------------------------------------------------
def test_segmentation_finds_a_known_step():
    s = np.arange(0, 120, 1.0)
    y = np.where(s < 60, 2.0, 3.2) + np.random.default_rng(0).normal(0, 0.02, len(s))
    segs = S.segment_series(s, y, spacing=1.0)
    assert len(segs) == 2
    assert abs(segs[0]["width_median"] - 2.0) < 0.05
    assert abs(segs[1]["width_median"] - 3.2) < 0.05
    assert abs(segs[0]["s_end"] - 59) < 3


def test_segmentation_ignores_a_change_smaller_than_the_error_budget():
    s = np.arange(0, 120, 1.0)
    y = np.where(s < 60, 2.00, 2.10)
    segs = S.segment_series(s, y, spacing=1.0)
    assert len(segs) == 1, "a 0.10 m step is inside our uncertainty and must not split"


def test_segmentation_splits_at_a_data_gap():
    s = np.r_[np.arange(0, 40, 1.0), np.arange(80, 120, 1.0)]
    y = np.full(len(s), 2.5)
    segs = S.segment_series(s, y, spacing=1.0)
    assert len(segs) == 2


# --------------------------------------------------------------------------
def test_bands_match_the_brief():
    assert C.band_of(0.9) == "below minimum"
    assert C.band_of(1.2) == "substandard"
    assert C.band_of(1.7) == "minimum acceptable"
    assert C.band_of(2.5) == "good"
    assert C.band_of(4.0) == "high capacity"
    # boundaries are inclusive at the bottom
    assert C.band_of(1.0) == "substandard"
    assert C.band_of(3.3) == "high capacity"


def test_borderline_band_is_declared():
    r = CF.band_with_uncertainty(1.51, 0.08)
    assert r["band"] == "minimum acceptable"
    assert r["band_borderline"] is True
    r2 = CF.band_with_uncertainty(2.40, 0.10)
    assert r2["band_borderline"] is False


# --------------------------------------------------------------------------
# Back-edge resolution. `verge` and `path` are ambiguous in OS MasterMap: each can
# be a strip between the kerb and the pavement, the walking surface itself, or the
# thing the pavement opens onto. These are the eight shapes that actually occur.
BACK_EDGE_CASES = [
    ("plain street",
     [(0, 3, "carriageway"), (3, 5, "roadside"), (5, 20, "building")], 2, 2.0),
    ("segregated cycle track inside the kerb line",
     [(0, 3, "carriageway"), (3, 5.5, "cycletrack"), (5.5, 8, "roadside"),
      (8, 20, "building")], 3, 2.5),
    ("kerb buffer then cycle track then pavement",
     [(0, 3, "carriageway"), (3, 3.4, "roadside"), (3.4, 5.9, "cycletrack"),
      (5.9, 8.4, "roadside"), (8.4, 20, "building")], 4, 2.5),
    ("planted verge between kerb and pavement",
     [(0, 3, "carriageway"), (3, 4, "verge"), (4, 6, "roadside"),
      (6, 20, "building")], 3, 2.0),
    ("pavement opens onto a park path",
     [(0, 3, "carriageway"), (3, 5, "roadside"), (5, 25, "path")], 2, 2.0),
    ("pedestrianised street: the path IS the walking surface",
     [(0, 3, "carriageway"), (3, 6, "path"), (6, 20, "building")], 2, 3.0),
    ("no footway at all",
     [(0, 3, "carriageway"), (3, 20, "building")], 1, 0.0),
    ("path with nothing behind it",
     [(0, 3, "carriageway"), (3, 9, "path")], 1, 0.0),
]


@pytest.mark.parametrize("name,prof,exp_edge,exp_w", BACK_EDGE_CASES,
                         ids=[c[0] for c in BACK_EDGE_CASES])
def test_back_edge_resolution(name, prof, exp_edge, exp_w):
    from sidewalk import transect as TR
    n = len(prof)
    edge = TR._find_back_edge(prof, n)
    assert edge == exp_edge, f"{name}: back edge at {edge}, expected {exp_edge}"
    t_edge = prof[edge][0] if edge is not None else prof[-1][1]
    j = (edge if edge is not None else n) - 1
    w = (t_edge - prof[j][0]) if (j >= 0 and prof[j][2] in ("roadside", "path")) else 0.0
    assert w == pytest.approx(exp_w, abs=1e-6), f"{name}: footway {w}, expected {exp_w}"
