"""Independent validation: OS carriageway width, areal reconciliation, sensitivity."""
from __future__ import annotations

import json
import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from sidewalk import config as C
from sidewalk import dataio
from sidewalk import osm as OSM
from sidewalk import pipeline as P
from sidewalk import transect as T
from sidewalk import validate as V

OUT = pathlib.Path("outputs/tables")


def main():
    rl0 = dataio.load_roadlinks()
    sel = rl0[rl0.name.isin(C.ALL_ROADS)]
    b = sel.total_bounds
    bbox = (b[0] - 200, b[1] - 200, b[2] + 200, b[3] + 200)
    osm_ways = OSM.fetch_osm(bbox)
    topo, rl, os_pts, isl, n_cycle = P.prepare_layers(osm_ways)
    furniture = OSM.combine_furniture(os_pts.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]],
                                      OSM.fetch_osm_furniture(bbox))
    samples = pd.read_parquet(OUT / "samples.parquet")
    segments = pd.read_csv(OUT / "segments.csv")
    xs = pd.read_csv(OUT / "cross_sections.csv")
    ctxs = {r: T.build_context(r, rl, topo) for r in C.ALL_ROADS}

    # 1 ---- OS carriageway -------------------------------------------------
    chk = V.os_carriageway_check(xs, rl, ctxs)
    chk.to_csv(OUT / "validation_os_carriageway.csv", index=False)
    lk = V.os_carriageway_by_link(chk)
    lk.to_csv(OUT / "validation_os_carriageway_by_link.csv", index=False)
    sc = lk[(lk.formofway == "Single Carriageway") & (lk.n_xs >= 5)]
    os_stats = dict(n_links=int(len(sc)),
                    median_residual_m=float(sc.residual_m.median()),
                    mae_m=float(sc.residual_m.abs().mean()),
                    rmse_m=float(np.sqrt((sc.residual_m ** 2).mean())),
                    p90_abs_residual_m=float(np.percentile(sc.residual_m.abs(), 90)),
                    pct_within_1m=float(100 * (sc.residual_m.abs() < 1).mean()),
                    correlation=float(sc[["ours_median_m", "os_averagewidth_m"]].corr().iloc[0, 1]))
    print("[1] OS carriageway (link level, single carriageway):", {k: round(v, 3) for k, v in os_stats.items()})

    # 2 ---- areal reconciliation -------------------------------------------
    ar = V.areal_reconciliation(samples, segments, ctxs, topo)
    ar.to_csv(OUT / "validation_areal.csv", index=False)
    if len(ar):
        cov = ar.coverage.dropna()
        areal_stats = dict(n_segments=int(len(cov)), median_coverage=float(cov.median()),
                           mean_coverage=float(cov.mean()),
                           pct_above_90=float(100 * (cov > 0.90).mean()),
                           pct_above_95=float(100 * (cov > 0.95).mean()),
                           pct_below_80=float(100 * (cov < 0.80).mean()))
        print("[2] areal reconciliation (ribbon on footway):",
              {k: round(v, 3) for k, v in areal_stats.items()})
    else:
        areal_stats = {}

    # 3 ---- sensitivity ----------------------------------------------------
    frames = []
    for road in ["Brick Lane", "Fournier Street", "Commercial Street"]:
        s = V.sensitivity(P.run_road, road, topo, rl, furniture, osm_ways)
        s.insert(0, "road", road)
        frames.append(s)
    sens = pd.concat(frames, ignore_index=True)
    sens.to_csv(OUT / "validation_sensitivity.csv", index=False)
    piv = sens.pivot(index="setting", columns="road", values="median_width_delta_m")
    print("\n[3] sensitivity - change in median footway width (m) vs baseline:")
    print(piv.round(3).to_string())
    print("\n    max |change| across all settings and roads: "
          f"{sens.median_width_delta_m.abs().max():.3f} m")

    json.dump(dict(os_carriageway=os_stats, areal=areal_stats,
                   sensitivity_max_abs_delta_m=float(sens.median_width_delta_m.abs().max())),
              open(OUT / "validation_summary.json", "w"), indent=2)
    return chk, lk, ar, sens


if __name__ == "__main__":
    main()
