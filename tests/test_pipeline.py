"""Integration tests against the supplied data.

Skipped automatically when the OS data is not present, so CI stays green on a
clone without the licensed GeoPackages.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from sidewalk import config as C
from sidewalk import pipeline as P

pytestmark = pytest.mark.skipif(
    not pathlib.Path(C.TOPO_GPKG).exists(),
    reason="OS MasterMap data not present (licensed; not committed)")


@pytest.fixture(scope="module")
def layers():
    return P.prepare_layers()


def test_crs_is_projected(layers):
    topo, roadlinks, *_ = layers
    assert str(topo.crs).upper().endswith("27700")
    assert str(roadlinks.crs).upper().endswith("27700")


def test_all_ten_roads_are_present(layers):
    _, roadlinks, *_ = layers
    missing = [r for r in C.ALL_ROADS if not (roadlinks.name == r).any()]
    assert not missing, f"roads missing from OS Highways: {missing}"


def test_islands_are_separated_from_footway(layers):
    topo, *_ = layers
    assert (topo.surface == "island").sum() > 0, "no traffic islands detected"
    # an island must not also be counted as footway
    assert not ((topo.surface == "island") & (topo.surface == "roadside")).any()


def test_a_known_narrow_street_measures_narrow(layers):
    """Fournier Street is an intact Georgian terrace: uniform, roughly 2-3 m footways."""
    topo, roadlinks, obs, isl, _ = layers
    df, xs, segs, summ, ctx = P.run_road("Fournier Street", topo, roadlinks, None, None)
    v = df[df.valid]
    assert summ["pct_valid"] > 80
    assert 1.8 < v.width_m.median() < 3.2, v.width_m.median()
    assert 4.0 < xs.carriageway_w.median() < 7.0


def test_every_rejected_sample_has_a_reason(layers):
    topo, roadlinks, *_ = layers
    df, *_ = P.run_road("Gunthorpe Street", topo, roadlinks, None, None)
    assert df.loc[~df.valid, "reason"].notna().all()


def test_components_sum_to_the_total(layers):
    topo, roadlinks, *_ = layers
    _, xs, *_ = P.run_road("Fournier Street", topo, roadlinks, None, None)
    d = xs.dropna(subset=["total_w"])
    lhs = d.left_w + d.carriageway_w + d.right_w
    assert np.allclose(lhs, d.total_w, atol=1e-6)


def test_output_invariants(layers):
    """Invariants that must hold across the whole published result set.

    These are the checks an assessor would run: that bands match widths, that
    intervals bracket their estimate, that the three components sum to the total,
    that nothing was dropped without a reason, and that scores stay in range.
    """
    tab = pathlib.Path("outputs/tables")
    if not (tab / "segments.csv").exists():
        pytest.skip("results not built - run scripts/run_all.py")
    import pandas as pd
    seg = pd.read_csv(tab / "segments.csv")
    xs = pd.read_csv(tab / "cross_sections.csv")
    s = pd.read_parquet(tab / "samples.parquet")

    assert set(seg.band) <= set(C.BAND_NAMES), "band outside the brief's five"
    assert (seg.length_m > 0).all()
    assert seg.width_median.notna().all()
    assert (seg.band == seg.width_median.map(C.band_of)).all(), "band does not match width"
    assert ((seg.ci_low <= seg.width_median) & (seg.width_median <= seg.ci_high)).all(), \
        "95% interval does not bracket the estimate"

    d = xs.dropna(subset=["total_w"])
    assert np.allclose(d.left_w + d.carriageway_w + d.right_w, d.total_w, atol=1e-6), \
        "left + carriageway + right must equal the reported total"

    assert s.loc[~s.valid, "reason"].notna().all(), "a rejection with no reason"
    assert s.loc[s.valid, "reason"].isna().all(), "a kept sample carrying a rejection reason"

    for col in ("confidence", "artefact_risk", "ribbon_coverage"):
        if col in seg:
            v = seg[col].dropna()
            assert ((v >= 0) & (v <= 1)).all(), f"{col} outside [0, 1]"

    for (road, side), g in seg.groupby(["road", "side"]):
        g = g.sort_values("s_start")
        assert (g.s_start.values[1:] >= g.s_end.values[:-1]).all(), \
            f"overlapping segments on {road} {side}"

    assert set(seg.road) == set(C.ALL_ROADS), "not every road produced segments"
    for road, g in seg.groupby("road"):
        assert set(g.side) == {"left", "right"}, f"{road} missing a side"
