"""Regenerate the number-bearing tables in README.md and REPORT.md from the outputs.

Hand-maintained tables in prose go stale the moment the pipeline is re-run, and a
stale number in a deliverable is worse than no number. Everything between the
<!-- AUTO:name --> markers is generated from outputs/tables, so re-running the
pipeline and re-running this keeps the write-up honest by construction.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

TAB = pathlib.Path("outputs/tables")


def _f(v, nd=2):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def table_cross_section() -> str:
    cs = pd.read_csv(TAB / "road_cross_section.csv")
    cs = cs.sort_values("total_street_width_m", ascending=False)
    out = ["| Road | Left footway | Carriageway | Right footway | **Total street width** | "
           "Frontage to frontage | Median strip |",
           "|---|---:|---:|---:|---:|---:|---|"]
    for _, r in cs.iterrows():
        med = ("—" if not np.isfinite(r.median_strip_m_where_present)
               else f"{r.median_strip_m_where_present:.2f} m on {r.pct_with_median_strip:.0f}%")  # noqa: RUF001
        out.append(f"| {r.road} | {_f(r.left_footway_m)} | {_f(r.carriageway_m)} | "
                   f"{_f(r.right_footway_m)} | **{_f(r.total_street_width_m)}** | "
                   f"{_f(r.frontage_to_frontage_m)} | {med} |")
    return "\n".join(out)


def table_results() -> str:
    seg = pd.read_csv(TAB / "segments.csv")
    qa = pd.read_csv(TAB / "road_summary.csv")
    cs = pd.read_csv(TAB / "road_cross_section.csv").set_index("road")
    val = dict(zip(qa.road, qa.pct_valid, strict=False))
    chosen = {"Leman Street", "Wentworth Street", "Fournier Street"}
    rows = []
    for road, g in seg.groupby("road"):
        L = g.length_m.sum()
        rows.append(dict(road=road,
                         segs=len(g), m=int(round(L)),
                         med=g.width_median.median(), sig=g.sigma_m.median(),
                         carr=float(cs.at[road, "carriageway_m"]),
                         lt15=100 * g.loc[g.width_median < 1.5, "length_m"].sum() / L,
                         ge20=100 * g.loc[g.width_median >= 2.0, "length_m"].sum() / L,
                         valid=val[road], conf=g.confidence.median()))
    t = pd.DataFrame(rows).sort_values("med", ascending=False)
    out = ["| Road | Segments | Measured | Median footway | ±1σ | Median carriageway | "
           "% <1.5 m | % ≥2.0 m | QA retained | Confidence |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in t.iterrows():
        name = r.road + (" *(chosen)*" if r.road in chosen else "")
        out.append(f"| {name} | {r.segs} | {r.m:,} m | {_f(r.med)} m | {_f(r.sig)} | "
                   f"{_f(r.carr)} m | {_f(r.lt15, 1)} | {_f(r.ge20, 1)} | "
                   f"{_f(r.valid, 1)}% | {_f(r.conf)} |")
    return "\n".join(out)


def table_widest() -> str:
    v = pd.read_csv(TAB / "verification_verdicts.csv")
    w = v[v.group == "widest"]
    out = ["| # | Road | Side | Position | Width | σ | Confidence | Risk | Verdict |",
           "|---|---|---|---|---:|---:|---:|---:|---|"]
    for i, (_, r) in enumerate(w.iterrows(), 1):
        out.append(f"| {i} | {r.road} | {r.side} | {r.s_start:.0f}–{r.s_end:.0f} m | "
                   f"**{r.width:.2f} m** | {r.sigma:.2f} | {r.confidence:.2f} | "
                   f"{r.risk:.2f} | **{r.verdict}** |")
    return "\n".join(out)


def table_risky() -> str:
    v = pd.read_csv(TAB / "verification_verdicts.csv")
    w = v[v.group == "artefact_risk"]
    out = ["| Road | Side | Position | Width | σ | Confidence | Risk | Verdict |",
           "|---|---|---|---:|---:|---:|---:|---|"]
    for _, r in w.iterrows():
        out.append(f"| {r.road} | {r.side} | {r.s_start:.0f}–{r.s_end:.0f} m | "
                   f"{r.width:.2f} m | {r.sigma:.2f} | {r.confidence:.2f} | "
                   f"{r.risk:.2f} | **{r.verdict}** |")
    return "\n".join(out)


def table_headline() -> str:
    seg = pd.read_csv(TAB / "segments.csv")
    s = pd.read_parquet(TAB / "samples.parquet")
    meta = json.load(open(TAB / "run_metadata.json"))
    vs = json.load(open(TAB / "validation_summary.json"))
    ar = vs["areal"]
    oc = vs["os_carriageway"]
    rows = [
        ("Roads measured", "**10** (7 required + 3 chosen)"),
        ("Street length", f"**{meta['total_axis_length_m']:,.0f} m**"),
        ("Cross-sections measured", f"**{len(s):,}**, one per metre per side"),
        ("Retained after QA", f"**{100*s.valid.mean():.1f}%** — every rejection carries a named reason"),
        ("Segments", f"**{len(seg)}**, from change-point detection, not averaging"),
        ("Median segment uncertainty", f"**±{seg.sigma_m.median():.2f} m** (1σ)"),
        ("Agreement with OS's own carriageway width",
         f"**median {oc['median_residual_m']:+.2f} m, MAE {oc['mae_m']:.2f} m, "
         f"r = {oc['correlation']:.3f}** ({oc['n_links']} links)"),
        ("Areal reconciliation",
         f"**median {ar['median_coverage']:.3f}** of the measured ribbon lies on surveyed footway polygons"),
        ("Sensitivity",
         f"**≤{vs['sensitivity_max_abs_delta_m']:.2f} m** change in median width across 12 parameter settings"),
    ]
    return "\n".join(["| | |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows])


def line_subtitle() -> str:
    seg = pd.read_csv(TAB / "segments.csv")
    sm = pd.read_parquet(TAB / "samples.parquet")
    meta = json.load(open(TAB / "run_metadata.json"))
    return (f"Tower Hamlets &amp; Spitalfields · 10 roads · {meta['total_axis_length_m']:,.0f} m · "
            f"{len(sm):,} cross-sections · {len(seg)} segments").replace("&amp;", "&")


def notes_roadlevel() -> str:
    """The three per-road observations quoted in prose, derived not typed."""
    seg = pd.read_csv(TAB / "segments.csv")
    qa = pd.read_csv(TAB / "road_summary.csv").set_index("road")

    def share_below(road, thr=1.5):
        g = seg[seg.road == road]
        return 100 * g.loc[g.width_median < thr, "length_m"].sum() / g.length_m.sum()

    def share_at_least(road, thr=2.0):
        g = seg[seg.road == road]
        return 100 * g.loc[g.width_median >= thr, "length_m"].sum() / g.length_m.sum()

    gun = share_below("Gunthorpe Street")
    others = max(share_below(r) for r in seg.road.unique() if r != "Gunthorpe Street")
    fs = seg[seg.road == "Fournier Street"]
    whs = qa.at["Whitechapel High Street", "pct_valid"]
    return "\n".join([
        f"- **Gunthorpe Street is the finding that matters operationally.** {gun:.0f}% of its "
        f"measured footway length is below 1.5 m, and it is the only road here where that is "
        f"true — no other road exceeds {others:.0f}%.",
        f"- **Fournier Street, the control, did its job.** Median {fs.width_median.median():.2f} m, "
        f"σ {fs.sigma_m.median():.2f} m, {share_at_least('Fournier Street'):.0f}% of its length in "
        f"the “good” band and nothing below 1.5 m. A street of essentially constant width "
        f"measures essentially constant.",
        f"- **Whitechapel High Street has the lowest retention, {whs:.0f}%,** and that is the "
        f"right answer: a dual carriageway between two major junctions with a public park on "
        f"one side has no well-defined footway cross-section along much of its length.",
    ])


BLOCKS = {
    "subtitle": line_subtitle,
    "notes": notes_roadlevel,
    "headline": table_headline,
    "results": table_results,
    "cross_section": table_cross_section,
    "widest": table_widest,
    "risky": table_risky,
}


def sync(path: pathlib.Path) -> int:
    text = path.read_text()
    n = 0
    for name, fn in BLOCKS.items():
        pat = re.compile(rf"<!-- AUTO:{name} -->.*?<!-- /AUTO:{name} -->", re.S)
        if pat.search(text):
            block = f"<!-- AUTO:{name} -->\n{fn()}\n<!-- /AUTO:{name} -->"
            text = pat.sub(lambda _m, b=block: b, text)
            n += 1
    path.write_text(text)
    return n


if __name__ == "__main__":
    for f in (pathlib.Path("README.md"), pathlib.Path("REPORT.md")):
        if f.exists():
            print(f"{f}: {sync(f)} auto blocks refreshed")
