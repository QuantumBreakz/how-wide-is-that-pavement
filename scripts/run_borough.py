"""Borough-scale extension: every named street in the Tower Hamlets study area.

The ten roads are the detailed deliverable. This shows the same pipeline running
unattended at scale, which is the real test of whether a measurement method is
usable rather than hand-tuned. Results are joined to LSOA 2021 boundaries so
sub-standard footway can be located, not just counted.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import geopandas as gpd
import numpy as np
import pandas as pd

from sidewalk import confidence as CF
from sidewalk import config as C
from sidewalk import dataio
from sidewalk import osm as OSM
from sidewalk import pipeline as P

OUT = pathlib.Path("outputs")
MIN_LEN = 60.0          # m: skip courts, yards and stubs
SPACING = 2.0           # coarser than the headline run; no ray fan


def main():
    t0 = time.time()
    lsoa = dataio.load_lsoa()
    study = lsoa.union_all()
    rl0 = dataio.load_roadlinks()
    inside = rl0[rl0.geometry.intersects(study)]
    lengths = inside.groupby("name").length.sum()
    roads = sorted(lengths[lengths >= MIN_LEN].index)
    print(f"[scope] {len(roads)} named streets >= {MIN_LEN:.0f} m inside Tower Hamlets LSOAs")

    b = inside.total_bounds
    bbox = (b[0] - 200, b[1] - 200, b[2] + 200, b[3] + 200)
    try:
        osm_ways = OSM.fetch_osm(bbox)
    except Exception as e:
        print("[osm] unavailable:", e); osm_ways = None
    topo, roadlinks, os_pts, isl, n_cycle = P.prepare_layers(osm_ways)
    print(f"[layers] islands={len(isl)} cycletracks={n_cycle}")

    segs, summaries, failed = [], [], []
    for i, road in enumerate(roads, 1):
        try:
            df, xs, sg, summ, ctx = P.run_road(road, topo, roadlinks, None, None,
                                               spacing=SPACING, use_fan=False)
            if len(sg):
                segs.append(sg)
            summ["axis_length_m"] = ctx.axis.length
            summaries.append(summ)
        except Exception as e:
            failed.append((road, str(e)[:80]))
        if i % 25 == 0:
            print(f"  {i}/{len(roads)}  {time.time()-t0:.0f}s")

    segments = pd.concat(segs, ignore_index=True)
    segments = CF.artefact_risk(segments)
    segments.to_csv(OUT / "tables/borough_segments.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "tables/borough_road_summary.csv", index=False)

    # --- LSOA join ---------------------------------------------------------
    pts = gpd.GeoDataFrame(segments.copy(),
                           geometry=gpd.points_from_xy(segments.x_mid, segments.y_mid),
                           crs=C.CRS)
    j = gpd.sjoin(pts, lsoa[["lsoa21cd", "lsoa21nm", "geometry"]], how="inner", predicate="within")
    j["sub_standard"] = j.width_median < 1.5
    agg = j.groupby(["lsoa21cd", "lsoa21nm"]).apply(lambda g: pd.Series({
        "n_segments": len(g),
        "measured_length_m": g.length_m.sum(),
        "median_width_m": np.median(np.repeat(g.width_median, g.length_m.round().astype(int).clip(lower=1))),
        "pct_length_below_1m": 100 * g.loc[g.width_median < 1.0, "length_m"].sum() / g.length_m.sum(),
        "pct_length_sub_standard": 100 * g.loc[g.sub_standard, "length_m"].sum() / g.length_m.sum(),
        "pct_length_good_plus": 100 * g.loc[g.width_median >= 2.0, "length_m"].sum() / g.length_m.sum(),
        "mean_confidence": g.confidence.mean(),
    })).reset_index()
    agg.to_csv(OUT / "tables/borough_lsoa.csv", index=False)
    gl = lsoa.merge(agg, on=["lsoa21cd", "lsoa21nm"], how="left")
    gl.to_file(OUT / "geo/borough_lsoa.geojson", driver="GeoJSON")

    meta = dict(streets_attempted=len(roads), streets_measured=len(summaries),
                failed=failed[:20], n_failed=len(failed),
                segments=int(len(segments)),
                measured_length_km=round(float(segments.length_m.sum()) / 1000, 2),
                lsoas_with_data=int(agg.shape[0]), runtime_s=round(time.time() - t0, 1))
    json.dump(meta, open(OUT / "tables/borough_metadata.json", "w"), indent=2)
    print(f"\n[done] {meta['streets_measured']}/{len(roads)} streets, "
          f"{meta['segments']} segments, {meta['measured_length_km']} km, "
          f"{meta['lsoas_with_data']} LSOAs, {meta['runtime_s']}s")
    if failed:
        print(f"[failed] {len(failed)}: {failed[:5]}")


if __name__ == "__main__":
    main()
