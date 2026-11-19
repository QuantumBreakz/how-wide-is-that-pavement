"""Run the full pipeline over all ten study roads and write every output table."""
from __future__ import annotations

import json
import pathlib
import sys
import time
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sidewalk import confidence as CF
from sidewalk import config as C
from sidewalk import osm as OSM
from sidewalk import pipeline as P
from sidewalk import viz

OUT = pathlib.Path("outputs")
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)
(OUT / "geo").mkdir(parents=True, exist_ok=True)


def main(roads=None, with_osm=True):
    roads = roads or C.ALL_ROADS
    t0 = time.time()
    # OSM is fetched first: it is needed to separate cycle tracks out of the OS
    # `Path` class before any measurement happens.
    tmp_links = __import__("sidewalk.dataio", fromlist=["x"]).load_roadlinks()
    sel = tmp_links[tmp_links.name.isin(roads)]
    b = sel.total_bounds
    bbox = (b[0] - 200, b[1] - 200, b[2] + 200, b[3] + 200)

    osm_ways = furniture = None
    if with_osm:
        try:
            osm_ways = OSM.fetch_osm(bbox)
        except Exception as e:
            print(f"[osm] ways unavailable: {e}")

    topo, roadlinks, os_points, islands, n_cycle = P.prepare_layers(osm_ways)
    print(f"[layers] topo={len(topo)} roadlinks={len(roadlinks)} "
          f"islands_excluded={len(islands)} ({islands.geometry.area.sum():,.0f} m2) "
          f"cycletracks_reclassified={n_cycle}")

    if with_osm and osm_ways is not None:
        try:
            furniture = OSM.combine_furniture(
                os_points.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]],
                OSM.fetch_osm_furniture(bbox))
            print(f"[osm] ways={len(osm_ways)} furniture={len(furniture)} "
                  f"({furniture.source.value_counts().to_dict()})")
        except Exception as e:
            print(f"[osm] furniture unavailable: {e}")

    all_samples, all_xs, all_segs, summaries, contexts = [], [], [], [], {}
    for road in roads:
        t = time.time()
        df, xs, segs, summ, ctx = P.run_road(road, topo, roadlinks, furniture, osm_ways)
        contexts[road] = ctx
        all_samples.append(df); all_xs.append(xs); all_segs.append(segs)
        summaries.append(summ)
        print(f"[{road:26s}] {ctx.axis.length:7.1f} m  "
              f"{summ['pct_valid']:5.1f}% valid  {len(segs):3d} segments  "
              f"{time.time()-t:5.1f}s")

    samples = pd.concat(all_samples, ignore_index=True)
    xsect = pd.concat(all_xs, ignore_index=True)
    segments = pd.concat(all_segs, ignore_index=True)
    road_summary = pd.DataFrame(summaries)

    # --- independent areal check, folded back into the evidence -------------
    from sidewalk import validate as V
    ar = V.areal_reconciliation(samples, segments, contexts, topo)
    if len(ar):
        segments = segments.merge(
            ar[["segment_id", "coverage", "ribbon_area_m2", "on_footway_area_m2"]]
            .rename(columns={"coverage": "ribbon_coverage"}), on="segment_id", how="left")
        # A segment whose swept ribbon does not land on surveyed footway polygons
        # cannot be trusted, however tidy its width profile looked.
        segments["confidence"] = segments.confidence * np.clip(
            segments.ribbon_coverage.fillna(0.9) ** 0.5, 0.25, 1.0)
        ar.to_csv(OUT / "tables/validation_areal.csv", index=False)

    # --- artefact risk needs the whole population ---------------------------
    segments = CF.artefact_risk(segments)
    segments = segments.sort_values(["road", "side", "s_start"]).reset_index(drop=True)

    samples.to_parquet(OUT / "tables/samples.parquet", index=False)
    xsect.to_csv(OUT / "tables/cross_sections.csv", index=False)
    segments.to_csv(OUT / "tables/segments.csv", index=False)
    road_summary.to_csv(OUT / "tables/road_summary.csv", index=False)

    gseg = P.attach_geometry(segments, contexts, samples)
    gseg.to_file(OUT / "geo/segments.geojson", driver="GeoJSON")
    gseg.to_crs(4326).to_file(OUT / "geo/segments_wgs84.geojson", driver="GeoJSON")

    axes = gpd.GeoDataFrame(
        {"road": list(contexts)}, geometry=[c.axis for c in contexts.values()], crs=C.CRS)
    axes.to_file(OUT / "geo/axes.geojson", driver="GeoJSON")

    # --- strip maps ---------------------------------------------------------
    for road in roads:
        sg = segments[segments.road == road]
        if not len(sg):
            continue
        fig, ax = plt.subplots(2, 1, figsize=(15, 5.6),
                               gridspec_kw={"height_ratios": [3.4, 1]})
        viz.strip_map(sg[sg.side == "left"], sg[sg.side == "right"], road,
                      contexts[road].axis.length,
                      carriageway=xsect[xsect.road == road], ax=ax[0])
        viz.band_legend(ax[0], loc="upper center", ncol=6)
        viz.confidence_strip(sg[sg.side == "left"], sg[sg.side == "right"], road,
                             contexts[road].axis.length, ax=ax[1])
        fig.tight_layout()
        slug = road.lower().replace(" ", "_")
        fig.savefig(OUT / f"figures/strip_{slug}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    meta = dict(
        generated=pd.Timestamp.utcnow().isoformat(),
        roads=list(roads), crs=C.CRS,
        sample_spacing_m=C.SAMPLE_SPACING,
        ensemble_half_window_m=C.ENSEMBLE_HALF_WINDOW,
        fan_half_angle_deg=C.FAN_HALF_ANGLE,
        kerb_fit_radius_m=C.KERB_FIT_RADIUS,
        kerb_corner_tol_m=C.KERB_CORNER_TOL,
        min_segment_len_m=C.MIN_SEGMENT_LEN,
        min_mean_shift_m=C.MIN_MEAN_SHIFT,
        sigma_survey_m=C.SIGMA_SURVEY,
        closure_good_m=C.CLOSURE_GOOD, closure_investigate_m=C.CLOSURE_INVESTIGATE,
        n_samples=int(len(samples)), n_segments=int(len(segments)),
        total_axis_length_m=float(sum(c.axis.length for c in contexts.values())),
        runtime_s=round(time.time() - t0, 1),
    )
    (OUT / "tables/run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[done] {len(samples)} samples, {len(segments)} segments, "
          f"{meta['total_axis_length_m']:.0f} m of street, {meta['runtime_s']}s")
    return samples, xsect, segments, road_summary, contexts


if __name__ == "__main__":
    main()
